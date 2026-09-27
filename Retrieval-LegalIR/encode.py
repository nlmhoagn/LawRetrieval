#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Trich vector cho toan bo chunk bang hiieu/halong_embedding.

Model: xlm-roberta base 278M, 768 chieu, max 512 token.
  - mean pooling theo attention mask (1_Pooling/config.json: mean=true, cls=false)
  - L2 normalize (module 2_Normalize) -> dot product = cosine
  - KHONG can prefix "query:"/"passage:" (config_sentence_transformers.json: prompts={})
  - train bang MatryoshkaLoss -> cat bot chieu roi normalize lai van dung duoc

RAM may nay chi con ~0.5GB nen vector duoc ghi thang ra memmap tren dia,
khong bao gio giu ca mang trong bo nho.

Usage:
    python encode.py --chunks ../Chunking-LegalIR/chunks.jsonl --out emb --limit 2000
    python encode.py --chunks ../Chunking-LegalIR/chunks.jsonl --out emb --dim 256
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = 'hiieu/halong_embedding'
MAX_LEN = 512
SUPER_BATCH = 4096      # gom lon roi sap theo do dai -> it padding thua


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def detect_pooling(model_id):
    """Doc 1_Pooling/config.json cua model thay vi cam cung 'mean'.

    halong/e5 dung mean, con ho BGE-M3 (bge-m3, Vietnamese_Embedding_V2) dung CLS.
    Lay trung binh token cua mot model huan luyen theo CLS se ra vector lech han,
    ma khong bao loi - hong am tham, kieu te nhat.
    """
    import json as _json
    cfg = None
    local = os.path.join(model_id, '1_Pooling', 'config.json')
    if os.path.exists(local):
        with open(local, encoding='utf-8') as f:
            cfg = _json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(model_id, '1_Pooling/config.json'),
                      encoding='utf-8') as f:
                cfg = _json.load(f)
        except Exception:
            pass
    if not cfg:
        print('  khong doc duoc 1_Pooling/config.json -> mac dinh mean')
        return 'mean', None
    if cfg.get('pooling_mode') in ('mean', 'cls'):
        mode = cfg['pooling_mode']
    elif cfg.get('pooling_mode_cls_token'):
        mode = 'cls'
    else:
        mode = 'mean'
    return mode, cfg.get('word_embedding_dimension') or cfg.get('embedding_dimension')


def count_lines(path):
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--chunks', '-c', required=True)
    p.add_argument('--out', '-o', default='emb', help='Thu muc dau ra')
    p.add_argument('--dim', type=int, default=768, help='Cat chieu Matryoshka (768/512/256/128)')
    p.add_argument('--batch', '-b', type=int, default=64)
    p.add_argument('--limit', '-n', type=int, default=0, help='Chi encode N chunk dau (de do thu)')
    p.add_argument('--fp32', action='store_true', help='Luu float32 thay vi float16')
    p.add_argument('--model', '-m', default=MODEL_ID,
                   help='Model id tren HuggingFace hoac duong dan model da fine-tune')
    args = p.parse_args()

    model_id = args.model

    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Nap {model_id} ...')
    pool_mode, model_dim = detect_pooling(model_id)
    print(f'  pooling: {pool_mode} | chieu goc: {model_dim}')
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float16 if dev == 'cuda' else torch.float32)
    model.to(dev).eval()

    total = args.limit or count_lines(args.chunks)
    dtype = np.float32 if args.fp32 else np.float16
    dim = args.dim
    if model_dim and args.dim == 768 and model_dim != 768:
        dim = model_dim          # khong cat Matryoshka khi model khong phai 768
        print(f'  --dim de mac dinh 768 nhung model ra {model_dim} -> dung {dim}')
    emb_path = os.path.join(args.out, f'emb_{dim}.f16' if dtype == np.float16 else f'emb_{dim}.f32')
    emb = np.memmap(emb_path, dtype=dtype, mode='w+', shape=(total, dim))

    print(f'Thiet bi: {dev} | {total:,} chunk -> {dim} chieu '
          f'({total * dim * np.dtype(dtype).itemsize / 1e9:.2f} GB tren dia)')

    tok_lens = []
    n_trunc = 0
    done = 0
    t0 = time.time()
    buf_txt, buf_idx = [], []

    def flush():
        nonlocal done, n_trunc
        if not buf_txt:
            return
        order = np.argsort([-len(t) for t in buf_txt])   # dai truoc -> OOM som neu co
        for s in range(0, len(order), args.batch):
            sel = order[s:s + args.batch]
            batch = [buf_txt[i] for i in sel]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors='pt')
            lens = enc['attention_mask'].sum(1)
            tok_lens.extend(lens.tolist())
            n_trunc += int((lens >= MAX_LEN).sum())
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc).last_hidden_state
                vec = (out[:, 0] if pool_mode == 'cls'
                       else mean_pool(out, enc['attention_mask']))
                if dim < vec.shape[1]:
                    vec = vec[:, :dim]                   # Matryoshka: cat roi chuan hoa lai
                vec = torch.nn.functional.normalize(vec.float(), p=2, dim=1)
            v = vec.cpu().numpy().astype(dtype)
            for j, i in enumerate(sel):
                emb[buf_idx[i]] = v[j]
        done += len(buf_txt)
        buf_txt.clear()
        buf_idx.clear()
        el = time.time() - t0
        print(f'  {done:,}/{total:,}  {done / el:.0f} chunk/s  '
              f'con lai ~{(total - done) / max(done / el, 1e-9) / 60:.0f} phut', flush=True)

    with open(args.chunks, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            buf_txt.append(json.loads(line)['text'])
            buf_idx.append(i)
            if len(buf_txt) >= SUPER_BATCH:
                flush()
    flush()

    emb.flush()
    del emb

    meta = {'model': model_id, 'dim': dim, 'n': total, 'dtype': str(np.dtype(dtype)),
            'pooling': pool_mode, 'normalized': True, 'max_len': MAX_LEN,
            'chunks_path': os.path.abspath(args.chunks), 'file': os.path.basename(emb_path)}
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    el = time.time() - t0
    tl = np.array(tok_lens)
    print()
    print(f'Xong {done:,} chunk trong {el / 60:.1f} phut ({done / el:.0f} chunk/s)')
    print(f'  Do dai token: median {int(np.median(tl))}, p90 {int(np.percentile(tl, 90))}, '
          f'p99 {int(np.percentile(tl, 99))}, max {int(tl.max())}')
    print(f'  Bi cat o {MAX_LEN} token: {n_trunc:,} chunk ({n_trunc / done * 100:.1f}%)')
    print(f'  -> {emb_path} ({os.path.getsize(emb_path) / 1e9:.2f} GB)')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
