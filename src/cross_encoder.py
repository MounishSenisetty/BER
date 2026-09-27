"""Character-level transformer cross-encoder for pair scoring (stage-2 feature).

The GBM sees ~100 hand-made similarity numbers per pair. The cross-encoder reads both records
character by character, jointly, with full attention between them. It can therefore learn noise
that no single feature describes: a token dropped here and a word appended there, reordered
address parts, a mistyped house number with a matching street, or a name turned into a domain.

* Input: `[CLS] s1 name | s1 address | record name | record address`. Each field is the normalised
  text (normalize.clean: transliteration + learned lexicon, lowercase ASCII), truncated or padded to
  a fixed width. Fixed field slots make batch assembly a pure numpy gather, and each slot gets its
  own segment embedding.
* Model: 4-layer pre-norm transformer encoder (d=192, 6 heads) trained FROM SCRATCH on the training
  split's candidate pairs. It uses no pretrained weights and no external data.
* Training data: every positive plus the hard negatives (stage-1 OOF p >= `train_min_p`), plus a
  sample of easy negatives.
* Leakage: two models, trained on disjoint halves of the entity folds. Each training pair is scored
  by the model that never saw its entity, so the stacked stage-2 model is trained on out-of-fold
  cross-encoder scores. Test pairs get the average of both models.
* Cost: only pairs with stage-1 p >= `min_p` are scored (the rest are certain non-matches), in
  fp16 on the GPU(s), with nn.DataParallel when there are two GPUs (Kaggle T4 x2).
"""
from __future__ import annotations

import io
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils import cpu_limit, LOG

PAD, CLS, UNK = 0, 1, 2
_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789"
_LUT = np.full(128, UNK, dtype=np.uint8)
for _i, _c in enumerate(_CHARS):
    _LUT[ord(_c)] = 3 + _i
VOCAB = 3 + len(_CHARS)


@dataclass
class CEConfig:
    name_len: int = 40
    addr_len: int = 72
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    ff: int = 768
    dropout: float = 0.1
    epochs: float = 3.0
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_train_pairs: int = 1_500_000
    train_min_p: float = 0.003           # negatives below this stage-1 p are "easy" ...
    easy_neg_frac: float = 0.05          # ... and only this fraction of them is used
    min_p: float = 0.01                  # pairs scored by the cross-encoder (train and test alike)
    seed: int = 42
    # pretrained backbone (Hugging Face id or local path, e.g. microsoft/deberta-v3-small); "" = the
    # from-scratch character model above
    backbone: str = ""
    max_len: int = 96                    # word-piece tokens per pair (pretrained backbone only)

    @property
    def seq_len(self) -> int:
        return 1 + 2 * (self.name_len + self.addr_len)


def torch_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "none"


# ----------------------------------------------------------------------------------------------
# text encoding
# ----------------------------------------------------------------------------------------------
def _encode_block(args):
    from .normalize import clean, set_lexicon
    texts, width, lex = args
    if lex is not None:
        set_lexicon(lex)
    out = np.zeros((len(texts), width), dtype=np.uint8)
    for i, t in enumerate(texts):
        b = clean(t).encode("ascii", "ignore")[:width]
        if b:
            out[i, :len(b)] = _LUT[np.frombuffer(b, dtype=np.uint8) & 127]
    return out


def encode_texts(texts: Sequence[str], width: int, n_jobs: int = -1, block: int = 100_000) -> np.ndarray:
    """(n, width) uint8 char ids of the normalised texts (PAD beyond the text)."""
    from .normalize import get_lexicon
    texts = [t if isinstance(t, str) else "" for t in texts]
    lex = get_lexicon()
    jobs = [(texts[i:i + block], width, lex) for i in range(0, len(texts), block)]
    n_jobs = cpu_limit() if n_jobs is None or n_jobs < 1 else n_jobs
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
            parts = list(ex.map(_encode_block, jobs))
    else:
        parts = [_encode_block(j) for j in jobs]
    return np.concatenate(parts) if parts else np.zeros((0, width), np.uint8)


def _clean_block(args):
    from .normalize import clean, set_lexicon
    texts, lex = args
    if lex is not None:
        set_lexicon(lex)
    return [clean(t) for t in texts]


def clean_texts(texts: Sequence[str], n_jobs: int = -1, block: int = 100_000) -> List[str]:
    """normalize.clean (transliteration + learned lexicon) over many texts, in parallel."""
    from .normalize import get_lexicon
    texts = [t if isinstance(t, str) else "" for t in texts]
    jobs = [(texts[i:i + block], get_lexicon()) for i in range(0, len(texts), block)]
    n_jobs = cpu_limit() if n_jobs is None or n_jobs < 1 else n_jobs
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
            parts = list(ex.map(_clean_block, jobs))
    else:
        parts = [_clean_block(j) for j in jobs]
    return [t for part in parts for t in part]


class TextTable:
    """Encoded name / address of a set of rows (Source 1 or records), looked up by row index:
    fixed-width char ids for the character model, "name ; address" strings for a pretrained one."""

    def __init__(self, names: Sequence[str], addrs: Sequence[str], cfg: CEConfig, n_jobs: int = -1):
        if cfg.backbone:
            n, a = clean_texts(names, n_jobs), clean_texts(addrs, n_jobs)
            self.text = np.asarray([f"{x} ; {y}" if y else x for x, y in zip(n, a)] + [None], dtype=object)[:-1]
        else:
            self.name = encode_texts(names, cfg.name_len, n_jobs)
            self.addr = encode_texts(addrs, cfg.addr_len, n_jobs)


class PairText:
    """(Source 1 text, record text) pairs for a pretrained backbone; indexable like an array."""

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self.a, self.b = a, b

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return PairText(self.a[idx], self.b[idx])


def encode_rows(frame, rows: np.ndarray, cfg: CEConfig, n_jobs: int = -1):
    """Encoded (name, addr) matrices for `rows` of a canonical frame, plus row -> position map."""
    rows = np.unique(rows)
    t = TextTable(frame["name"].to_numpy(dtype=object)[rows], frame["address"].to_numpy(dtype=object)[rows],
                  cfg, n_jobs)
    return rows, t


def assemble(t1: TextTable, i1: np.ndarray, t2: TextTable, i2: np.ndarray):
    if hasattr(t1, "text"):
        return PairText(t1.text[i1], t2.text[i2])
    n = len(i1)
    cls = np.full((n, 1), CLS, dtype=np.uint8)
    return np.concatenate([cls, t1.name[i1], t1.addr[i1], t2.name[i2], t2.addr[i2]], axis=1)


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
def _build_model(cfg: CEConfig):
    import torch
    import torch.nn as nn

    class CrossEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            L = cfg.seq_len
            self.tok = nn.Embedding(VOCAB, cfg.d_model, padding_idx=PAD)
            self.pos = nn.Embedding(L, cfg.d_model)
            seg = np.concatenate([[0], np.full(cfg.name_len, 1), np.full(cfg.addr_len, 2),
                                  np.full(cfg.name_len, 3), np.full(cfg.addr_len, 4)])
            self.register_buffer("seg_ids", torch.as_tensor(seg, dtype=torch.long), persistent=False)
            self.register_buffer("pos_ids", torch.arange(L, dtype=torch.long), persistent=False)
            self.seg = nn.Embedding(5, cfg.d_model)
            # alignment hints: does this position's char bigram / trigram occur on the other record
            # (any field / the same kind of field)? Gives the encoder the comparison it would otherwise
            # have to discover through attention alone, which makes training from scratch fast
            self.match = nn.Embedding(8, cfg.d_model)
            ftype = np.concatenate([[2], np.zeros(cfg.name_len), np.ones(cfg.addr_len),
                                    np.zeros(cfg.name_len), np.ones(cfg.addr_len)])
            self.register_buffer("ftype", torch.as_tensor(ftype, dtype=torch.long), persistent=False)
            self.split = 1 + cfg.name_len + cfg.addr_len
            layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff, cfg.dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))

        def _ngram(self, x, n):
            L = x.shape[1] - n + 1
            ids = x[:, :L].clone()
            ok = x[:, :L] > UNK
            for k in range(1, n):
                nxt = x[:, k:L + k]
                ids = ids * VOCAB + nxt
                ok &= (nxt > UNK) & (self.seg_ids[k:L + k] == self.seg_ids[:L])[None]
            ids = torch.where(ok, ids, torch.full_like(ids, -1))
            return torch.cat([ids, x.new_full((x.shape[0], n - 1), -1)], 1)

        def _flags(self, x):
            sp, ft = self.split, self.ftype
            same = ft[sp:][:, None] == ft[1:sp][None, :]
            bits = []
            for n in (3, 2):
                g = self._ngram(x, n)
                a, b = g[:, 1:sp], g[:, sp:]
                eq = (b[:, :, None] == a[:, None, :]) & (b[:, :, None] >= 0)
                bits.append(torch.cat([eq.any(1), eq.any(2)], 1))
                if n == 3:
                    eqs = eq & same[None]
                    bits.append(torch.cat([eqs.any(1), eqs.any(2)], 1))
            f = bits[0].long() + 2 * bits[1].long() + 4 * bits[2].long()
            return torch.cat([f.new_zeros((x.shape[0], 1)), f], 1)

        def forward(self, x):
            h = (self.tok(x) + self.pos(self.pos_ids)[None] + self.seg(self.seg_ids)[None]
                 + self.match(self._flags(x)))
            h = self.enc(h, src_key_padding_mask=(x == PAD))
            return self.head(self.norm(h[:, 0])).squeeze(-1)

    return CrossEncoder()


def _wrap(model, device):
    import torch
    model = model.to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


_TOKENIZERS: Dict[str, object] = {}


def _tokenizer(cfg: CEConfig):
    if cfg.backbone not in _TOKENIZERS:
        from transformers import AutoTokenizer
        _TOKENIZERS[cfg.backbone] = AutoTokenizer.from_pretrained(cfg.backbone)
    return _TOKENIZERS[cfg.backbone]


def _build_pretrained(cfg: CEConfig, weights: bool):
    """Sequence-classification head (1 logit) on a Hugging Face backbone. `weights=False` builds the
    architecture only (the fine-tuned weights come from the model bundle)."""
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForSequenceClassification
    if weights:
        m = AutoModelForSequenceClassification.from_pretrained(cfg.backbone, num_labels=1, ignore_mismatched_sizes=True)
    else:
        m = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(cfg.backbone, num_labels=1))

    class Head(nn.Module):                     # returns a plain logit tensor (DataParallel-friendly)
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kw = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
            return self.m(input_ids=input_ids, attention_mask=attention_mask, **kw).logits.squeeze(-1)

    return Head()


def _new_model(cfg: CEConfig, weights: bool = True):
    return _build_pretrained(cfg, weights) if cfg.backbone else _build_model(cfg)


def _batch(inputs, idx, cfg: CEConfig, device: str):
    import torch
    if isinstance(inputs, PairText):
        enc = _tokenizer(cfg)(list(inputs.a[idx]), list(inputs.b[idx]), truncation=True, max_length=cfg.max_len,
                              padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    return torch.as_tensor(inputs[idx].astype(np.int64), device=device)


def _call(model, xb):
    return model(**xb) if isinstance(xb, dict) else model(xb)


def train_model(X, y: np.ndarray, cfg: CEConfig, tag: str = "", X_val=None, y_val: Optional[np.ndarray] = None) -> bytes:
    """Train one cross-encoder on assembled pairs (char matrix or PairText); returns the serialized
    (fp16) state dict."""
    import torch
    device = torch_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = _wrap(_new_model(cfg), device)
    if cfg.backbone:                           # usual transformer recipe: no decay on biases / norms
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    else:
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "tok" not in n and "pos" not in n and "seg" not in n]
        no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "tok" not in n and "pos" not in n and "seg" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.98))
    n = len(y)
    bs = cfg.batch_size
    steps = max(1, int(math.ceil(cfg.epochs * n / bs)))

    def lr_at(s):
        if s < cfg.warmup_steps:
            return (s + 1) / cfg.warmup_steps
        return max(0.02, 0.5 * (1 + math.cos(math.pi * (s - cfg.warmup_steps) / max(1, steps - cfg.warmup_steps))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = torch.nn.BCEWithLogitsLoss()
    yt = y.astype(np.float32)
    model.train()
    t0, seen, run = time.time(), 0, 0.0
    perm = rng.permutation(n)
    pos = 0
    for s in range(steps):
        if pos + bs > n:
            perm, pos = rng.permutation(n), 0
        idx = np.sort(perm[pos:pos + bs])
        pos += bs
        xb = _batch(X, idx, cfg, device)
        yb = torch.as_tensor(yt[idx], device=device)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp):
            logit = _call(model, xb)
        loss = lossf(logit.float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        lv = loss.item()
        run = 0.98 * run + 0.02 * lv if s else lv
        seen += len(idx)
        if (s + 1) % max(1, steps // 10) == 0 or s == steps - 1:
            LOG.info("  CE%s step %d/%d  loss %.4f  (%.0f pairs/s)", tag, s + 1, steps, run, seen / (time.time() - t0))
    blob = io.BytesIO()
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in _unwrap(model).state_dict().items()}, blob)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    out = blob.getvalue()
    if X_val is not None and len(X_val):
        from sklearn.metrics import log_loss, roc_auc_score
        pv = 1 / (1 + np.exp(-predict_logits([out], X_val, cfg)))
        LOG.info("  CE%s held-out: logloss %.5f AUC %.5f (%d pairs)", tag, log_loss(y_val, pv, labels=[0, 1]),
                 roc_auc_score(y_val, pv) if 0 < y_val.sum() < len(y_val) else float("nan"), len(y_val))
    return out


_MODEL_CACHE: Dict[int, object] = {}


def _load(blob: bytes, cfg: CEConfig, device: str):
    import torch
    key = id(blob)
    if key not in _MODEL_CACHE:
        m = _new_model(cfg, weights=False)
        m.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))     # fp16 -> fp32 on copy
        m.eval()
        _MODEL_CACHE[key] = _wrap(m, device)
    return _MODEL_CACHE[key]


def predict_logits(blobs: List[bytes], X, cfg: CEConfig, batch: int = 4096) -> np.ndarray:
    """Mean logit of the given models over the assembled pairs X (char matrix or PairText)."""
    import torch
    device = torch_device()
    out = np.zeros(len(X), dtype=np.float32)
    if len(X) == 0 or not blobs:
        return out
    use_amp = device == "cuda"
    if cfg.backbone:
        batch = 512 if device == "cuda" else 32
    else:
        batch = batch if device == "cuda" else 256
    for blob in blobs:
        m = _load(blob, cfg, device)
        with torch.inference_mode():
            for a in range(0, len(X), batch):
                xb = _batch(X, np.arange(a, min(a + batch, len(X))), cfg, device)
                with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp):
                    out[a:a + batch] += _call(m, xb).float().cpu().numpy()
    return out / len(blobs)


# ----------------------------------------------------------------------------------------------
# stage-2 features from cross-encoder logits
# ----------------------------------------------------------------------------------------------
CE_FEATURES = ["ce_logit", "ce_rec_rank", "ce_rec_gap", "ce_ent_rank", "ce_ent_gap", "ce_x_p1"]


def ce_features(ce_logit: np.ndarray, p1: np.ndarray, s1: np.ndarray, rec: np.ndarray,
                n_s1: int, n_rec: int) -> Dict[str, np.ndarray]:
    """Cross-encoder features for the scored pairs (all arrays aligned to them): the logit and how it
    ranks among the record's / the entity's other scored candidates."""
    from .stage2 import GroupStats
    pc = (1 / (1 + np.exp(-np.clip(ce_logit, -30, 30)))).astype(np.float32)
    out = {"ce_logit": ce_logit.astype(np.float32)}
    for pre, g, n in (("rec", rec, n_rec), ("ent", s1, n_s1)):
        st = GroupStats(g, pc, n)
        other = np.where(st.rank == 1, st.top2[g], st.top1[g])
        out[f"ce_{pre}_rank"] = st.rank.astype(np.float32)
        out[f"ce_{pre}_gap"] = (pc - np.nan_to_num(other, nan=0.0)).astype(np.float32)
    out["ce_x_p1"] = np.sqrt(pc * np.clip(p1, 0, 1)).astype(np.float32)
    return out


def score_pairs(blobs: List[bytes], use: np.ndarray, s1_frame, rec_frame, s1_rows: np.ndarray,
                rec_rows: np.ndarray, cfg: CEConfig, n_jobs: int = -1, chunk: int = 1_000_000) -> np.ndarray:
    """Cross-encoder logits for pairs (s1_rows[k], rec_rows[k]). `use[k]` picks the model: 0 / 1, or 2 =
    the average of both (test pairs)."""
    u1, t1 = encode_rows(s1_frame, s1_rows, cfg, n_jobs)
    u2, t2 = encode_rows(rec_frame, rec_rows, cfg, n_jobs)
    i1, i2 = np.searchsorted(u1, s1_rows), np.searchsorted(u2, rec_rows)
    out = np.zeros(len(s1_rows), dtype=np.float32)
    t0 = time.time()
    for g in (0, 1, 2):
        idx = np.flatnonzero(use == g)
        models = [blobs[g]] if g < 2 else list(blobs)
        for a in range(0, len(idx), chunk):
            k = idx[a:a + chunk]
            out[k] = predict_logits(models, assemble(t1, i1[k], t2, i2[k]), cfg)
    LOG.info("  cross-encoder scored %d pairs in %.1fs", len(out), time.time() - t0)
    return out


def config_dict(cfg: CEConfig) -> dict:
    return asdict(cfg)
