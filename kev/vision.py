"""An image as the start of the state, on the base's own vision tower.

Kev's bases (Qwen3.5 / Qwen3.8) are vision-language models, but DecisionModel loads
only their text backbone. The adapter and the pointer head live entirely in that
backbone, so they apply unchanged to the same module inside the full model.
VisionDecisionModel loads the full model (AutoModel: vision tower + language model,
no vocab head) and splices the image tokens immediately after `<|fim_prefix|>`
(the state delimiter; controller-heads.md / areal splice_vision):

    <|fim_prefix|> <|vision_start|> <|image_pad|> x N <|vision_end|> state text <|fim_middle|> ...

Upstream encode() has no image; this is the only splice. A record without an image
takes DecisionModel's path unchanged. Training shots are 490x504 and become N=240
pads on Qwen3.8-27B's processor (k4-plan-corpus; process_decision_image).

    tok, m = Checkpoint("jaredpalmer/kev-4b").load_vision("cuda", LoadOptions(dtype=torch.bfloat16))
    enc = admit(m, tok, rec, image=ImageOps.exif_transpose(Image.open("photo.jpg")))
    probs = m.probs(enc)

Over HTTP (kev.serve --vision) the picture travels in the request's `image` field as
a data URL or bare base64; decode_image turns it into the RGB, EXIF-rotated PIL image
encode() takes.
"""
import base64
import binascii
import hashlib
import io
import re

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, DynamicCache
from transformers.models.auto.image_processing_auto import AutoImageProcessor   # the top-level name requires torchvision

from .model import OPT_NONE, DecisionModel, probs_one, rows_of

# the bases' own processor allows 16.7 MP, ~16k tokens per picture (a 12 MP phone photo is 11,844); cap at ~1,024
MAX_IMAGE_PIXELS = 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024       # encoded file size a request may carry
MAX_SOURCE_PIXELS = 16_000_000           # checked from the header, before the pixels are decoded
# k4-plan-corpus / S1 training shots; Qwen3.8-27B preprocessor (patch 16, merge 2, t=1)
TRAIN_IMAGE_SIZE = (490, 504)
TRAIN_IMAGE_PADS = 240
_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,(.*)", re.S)


def decode_image(value):
    """A request's `image` string (a data URL, data:image/png;base64,..., or bare base64) -> RGB PIL image with its
    EXIF rotation applied. ValueError for anything that is not one still picture within the size limits."""
    m = _DATA_URL.fullmatch(value)
    if value.startswith("data:") and not m: raise ValueError("image data URL must look like data:image/<type>;base64,<data>")
    payload = m.group(1) if m else value
    if len(payload) > MAX_IMAGE_BYTES * 4 // 3 + 4: raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
    try: raw = base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError): raise ValueError("image is not valid base64") from None
    try: img = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError: raise ValueError(f"image has more than {MAX_SOURCE_PIXELS} pixels") from None
    except Exception: raise ValueError("image is not a picture Pillow can read") from None
    if img.width * img.height > MAX_SOURCE_PIXELS: raise ValueError(f"image has {img.width}x{img.height} pixels, more than {MAX_SOURCE_PIXELS}")
    if getattr(img, "n_frames", 1) != 1: raise ValueError("only still images are supported")
    return ImageOps.exif_transpose(img).convert("RGB")


def splice_image(enc, n_image_tokens, vision_start, image_pad, vision_end):
    """encode() output with an image block inserted immediately after <|fim_prefix|> (ids[0], the
    <state> delimiter). The block belongs to the state (segment 0), shifts every later index by
    its length, and keeps positions contiguous (the model replaces them with M-RoPE)."""
    block = [vision_start] + [image_pad] * n_image_tokens + [vision_end]
    k = len(block)
    return {**enc, "ids": enc["ids"][:1] + block + enc["ids"][1:],
            "seg": [0] * (k + 1) + enc["seg"][1:],
            "pos": list(range(k + 1)) + [p + k for p in enc["pos"][1:]],
            "opt": [OPT_NONE] * (k + 1) + enc["opt"][1:],
            "decide_idx": [d + k for d in enc["decide_idx"]],
            "opt_idx": [[o + k for o in oi] for oi in enc["opt_idx"]]}


class VisionDecisionModel(DecisionModel):
    accepts_images = True   # kev.serve takes a request's `image` only from a model that says so

    def __init__(self, name, tok, device, revision=None, **kw):
        super().__init__(name, tok, device, revision=revision, **kw)
        if not self.hybrid: raise ValueError(f"{name}: images use the row form, which Kev runs on the hybrid (Qwen3.5) bases only")
        cfg = self.vlm.config
        self.image_pad, self.vision_start, self.vision_end = cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id
        self.spatial_merge = cfg.vision_config.spatial_merge_size
        self.processor = AutoImageProcessor.from_pretrained(name, revision=revision, backend="pil")   # same pixels with or without torchvision
        size = self.processor.size
        self.processor.size = {"shortest_edge": size.shortest_edge, "longest_edge": min(size.longest_edge, MAX_IMAGE_PIXELS)}

    def backbone(self, name, *, revision=None, weights=None, **kw):
        """The language model inside the full vision-language model: the module DecisionModel.backbone loads on its own,
        so the adapter's keys match it unchanged.

        The checkpoint stores that module as `model.language_model.*`. Transformers' text-only rename rewrites those keys
        to `model.*` whenever a `qwen3_5_text` submodule is present (5.5.4 does this for the whole vision-language model),
        and the language weights are then missing. Pointing them at `language_model.*` first is the name Qwen3_5Model
        uses, and the text rename no longer matches."""
        if weights is not None:
            raise ValueError("load_vision applies an adapter to the base's vision-language model; a full-weight checkpoint has no adapter")
        self.vlm = AutoModel.from_pretrained(name, revision=revision, key_mapping={"^model.language_model.": "language_model."}, **kw)
        if not hasattr(self.vlm, "get_image_features"): raise ValueError(f"{name} has no vision tower")
        return self.vlm.language_model

    def encode(self, tok, rec, image=None, **kw):
        """DecisionModel.encode, plus `image` (one picture, e.g. a PIL image) immediately after <|fim_prefix|>.
        max_state limits the state text; the image tokens come on top of it."""
        enc = super().encode(tok, rec, **kw)
        if image is None: return enc
        pix = self.processor(images=[image], return_tensors="pt")
        if len(pix["image_grid_thw"]) != 1: raise ValueError(f"one image per record, got {len(pix['image_grid_thw'])}")
        n = int(pix["image_grid_thw"][0].prod()) // self.spatial_merge ** 2
        digest = hashlib.blake2b(pix["pixel_values"].numpy().tobytes(), digest_size=16).hexdigest()
        return {**splice_image(enc, n, self.vision_start, self.image_pad, self.vision_end),
                "pixel_values": pix["pixel_values"], "image_grid_thw": pix["image_grid_thw"], "image_digest": digest}

    def forward_rows_batch(self, encs, shared_prefix=False):
        if not any("image_grid_thw" in e for e in encs): return super().forward_rows_batch(encs, shared_prefix)
        if shared_prefix: raise ValueError("shared-prefix training does not carry images")
        return [self._image_rows(e) if "image_grid_thw" in e else super().forward_rows_batch([e])[0] for e in encs]

    @torch.no_grad()
    def probs(self, enc):
        """Probabilities per question. On a fused Ascend model with multiple questions, an image request runs the state
        once and the branches from its prefix; a single question runs as one graphed row."""
        if "image_grid_thw" not in enc: return super().probs(enc)
        if self._can_prefix_image(enc) and len(enc["decide_idx"]) > 1:
            return self.probs_and_prefix(enc, keep=False)[0]
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    @torch.no_grad()
    def probs_batch(self, encs, prefixes, keep):
        """Text requests keep the batched prefix and CUDA-graph path. Image requests use the prefix path when the fused
        Ascend layers are active (state once + branches, or branches from a cached prefix)."""
        image = [i for i, e in enumerate(encs) if "image_grid_thw" in e]
        if not image: return super().probs_batch(encs, prefixes, keep)
        text = [i for i in range(len(encs)) if i not in set(image)]
        ps, prefs = [None] * len(encs), [None] * len(encs)
        if text:
            tps, tprefs = super().probs_batch([encs[i] for i in text], [prefixes[i] for i in text], [keep[i] for i in text])
            for i, p, pref in zip(text, tps, tprefs): ps[i], prefs[i] = p, pref
        for i in image:
            if self._can_prefix_image(encs[i]):
                if prefixes[i] is not None:
                    ps[i], prefs[i] = self.probs_with_prefix(encs[i], prefixes[i]), prefixes[i]
                elif keep[i] or len(encs[i]["decide_idx"]) > 1:
                    ps[i], prefs[i] = self.probs_and_prefix(encs[i], keep=keep[i])
                else:
                    ps[i] = self.probs(encs[i])
            else:
                ps[i] = self.probs(encs[i])
        return ps, prefs

    def _can_prefix_image(self, enc):
        return str(self.device).startswith("npu") and getattr(self.lm.forward, "__module__", "") == "kev.npu_qwen35"

    def _image_state_inputs(self, enc):
        Ls = enc["seg"].count(0)
        grid = enc["image_grid_thw"].to(self.device)
        image = self.vlm.get_image_features(enc["pixel_values"].to(self.device), grid, return_dict=True).pooler_output[0]
        ids = torch.tensor([enc["ids"][:Ls]], device=self.device)
        is_image = ids == self.image_pad
        emb = self.vlm.get_input_embeddings()(ids)
        emb = emb.masked_scatter(is_image[..., None].expand_as(emb), image.to(emb.dtype))
        pos, deltas = self.vlm.get_rope_index(ids, mm_token_type_ids=is_image.int(), image_grid_thw=grid)
        return Ls, emb, pos, int(deltas[0, 0].item())

    def _image_rows(self, enc):
        """Logits per question of one record with an image: the rows of forward_rows_batch (state, image included, + one
        branch), with the image features computed once and the backbone inputs of each chunk built by the base's rules."""
        grid = enc["image_grid_thw"].to(self.device)
        image = self.vlm.get_image_features(enc["pixel_values"].to(self.device), grid, return_dict=True).pooler_output[0]

        def inputs(ids, att):   # the features in every row's <image_pad> slots; M-RoPE positions from the ids
            is_image = ids == self.image_pad
            emb = self.vlm.get_input_embeddings()(ids)
            emb = emb.masked_scatter(is_image[..., None].expand_as(emb), image.to(emb.dtype).repeat(len(ids), 1))
            pos, _ = self.vlm.get_rope_index(ids, mm_token_type_ids=is_image.int(), image_grid_thw=grid.repeat(len(ids), 1), attention_mask=att)
            return {"inputs_embeds": emb, "position_ids": pos}

        S, Sp, branches = rows_of(enc)
        hs = self._rows_hidden([(S + r["ids"], Sp + r["pos"]) for r in branches], inputs=inputs)
        return [self.head(h[len(S) + r["decide"]], h[torch.tensor([len(S) + o for o in r["opts"]], device=self.device)]) for h, r in zip(hs, branches)]

    @torch.no_grad()
    def prefix(self, enc, keep=True):
        if "image_grid_thw" not in enc: return super().prefix(enc)
        if not self._can_prefix_image(enc):
            raise NotImplementedError("the state-prefix cache does not carry images; use probs()")
        Ls, emb, pos, delta = self._image_state_inputs(enc)
        out = self.lm(inputs_embeds=emb, position_ids=pos, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        cache = out.past_key_values
        cache.rope_delta = delta
        if keep and getattr(self.lm, "graphs", None) is not None:
            cache = self.lm.graphs.snapshot_prefix(cache)
        return Ls, cache, None

    @torch.no_grad()
    def probs_and_prefix(self, enc, keep=True):
        if "image_grid_thw" not in enc: return super().probs_and_prefix(enc)
        if not self._can_prefix_image(enc):
            return [F.softmax(z, -1).cpu() for z in self.forward(enc)], None
        Ls, cache, h_state = self.prefix(enc, keep=keep)
        return self._image_branches_from_prefix(enc, cache), ((Ls, cache, h_state) if keep else None)

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        if "image_grid_thw" not in enc: return super().probs_with_prefix(enc, prefix)
        if not self._can_prefix_image(enc):
            raise NotImplementedError("the state-prefix cache does not carry images; use probs()")
        Ls, cache, _ = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        return self._image_branches_from_prefix(enc, cache)

    def _image_branches_from_prefix(self, enc, cache):
        S, _, rows = rows_of(enc)
        delta = getattr(cache, "rope_delta", 0)
        hs = self._rows_hidden([(r["ids"], [p + delta for p in r["pos"]]) for r in rows], cache=cache, prefix_len=len(S))
        ps = [F.softmax(self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)]), -1) for h, r in zip(hs, rows)]
        return list(torch.cat(ps).cpu().split([len(p) for p in ps]))
