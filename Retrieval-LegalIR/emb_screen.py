#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Sang loc embedder: chi encode ~30k chunk da nam trong tap ung vien, khong phai
ca 487k chunk cua corpus.

Encode day du mat ~70 phut moi model. O day chi can biet model nao xep chunk gold
len cao hon TRONG tap ung vien co san - phan "chon ra tap ung vien" bo qua duoc vi
BM25 da phu 99.2% gold roi.

So sanh cong bang: zero-shot voi zero-shot. halong-ft da fine-tune tren 6000 cap
nen dem no ra so voi model chua fine-tune la lech.

    python emb_screen.py --models BAAI/bge-m3 AITeamVN/Vietnamese_Embedding_V2
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from bm25 import BM25Index

PAIRS = 'pairs_val500_ftq.json'
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'LegalIR - Public Test-20260806T081424Z-1-001', 'LegalIR - Public Test')


def load_json(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def metrics(pairs, score_of):
    """score_of(qid) -> mang diem theo thu tu cand. Tra ve (AUC, Recall@5 muc doc)."""
    pos, neg, rec = [], [], []
    for q, v in pairs.items():
        if not v['cand']:
            rec.append(0.0)
            continue
        s = score_of(q)
        gold = set(v['gold'])
        for (d, _, _), sc in zip(v['cand'], s):
            (pos if d in gold else neg).append(sc)
        vals = {}
        for j in np.argsort(-s):
            vals.setdefault(v['cand'][j][0], []).append(float(s[j]))
        ds = sorted(((d, float(np.mean(x[:2]))) for d, x in vals.items()),
                    key=lambda t: -t[1])
        rec.append(len(gold & {d for d, _ in ds[:5]}) / len(gold))
    pos, neg = np.array(pos), np.array(neg)
    # AUC uoc luong bang lay mau, tranh ma tran 500x36000
    rng = np.random.default_rng(0)
    a = rng.choice(pos, 200000)
    b = rng.choice(neg, 200000)
    return float((a > b).mean() + 0.5 * (a == b).mean()), np.array(rec)


def from_memmap(emb_dir, pairs, qids, questions):
    """Diem dense tu thu muc vector da tinh san (halong goc / halong-ft)."""
    import torch
    from hybrid import Hybrid
    h = Hybrid('index', emb_dir, k1=2.5, b=0.9)
    qv = np.concatenate([h.encode_query([questions[q] for q in qids[i:i + 64]])
                         for i in range(0, len(qids), 64)])
    out = {}
    for i, q in enumerate(qids):
        cis = np.array([c[1] for c in pairs[q]['cand']], dtype=np.int64)
        if cis.size == 0:
            out[q] = np.zeros(0)
            continue
        out[q] = h.emb[cis].astype(np.float32) @ qv[i]
    del h
    torch.cuda.empty_cache()
    return out


def from_model(model_id, pairs, qids, questions, batch):
    """Encode moi cac chunk ung vien bang mot model bat ky."""
    import torch
    from sentence_transformers import SentenceTransformer

    bm = BM25Index('index')
    uniq = sorted({c[1] for v in pairs.values() for c in v['cand']})
    print(f'  {len(uniq)} chunk rieng can encode')
    texts = [bm.read_chunk(int(c))['text'] for c in uniq]
    pos = {c: i for i, c in enumerate(uniq)}

    t = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SentenceTransformer(model_id, trust_remote_code=True, device=dev)
    m.max_seq_length = 256
    if dev == 'cuda':
        m = m.half()          # fp32 cham gap doi, va khong can do chinh xac o day
    print(f'  nap model {time.time() - t:.0f}s | {dev} fp16 | pooling: '
          f'{[type(x).__name__ for x in m._modules.values()]}')

    # Do toc do that tren lo dau roi moi ngoai suy - dung doan tu lan chay khac
    t = time.time()
    m.encode(texts[:512], batch_size=batch, normalize_embeddings=True,
             show_progress_bar=False, convert_to_numpy=True)
    rate = 512 / max(time.time() - t, 1e-6)
    print(f'  do thu: {rate:.0f} chunk/s -> uoc {len(texts) / rate / 60:.1f} phut',
          flush=True)

    t = time.time()
    ce = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                  show_progress_bar=False, convert_to_numpy=True)
    print(f'  encode chunk: {time.time() - t:.0f}s ({len(texts) / (time.time() - t):.0f}/s)',
          flush=True)
    qe = m.encode([questions[q] for q in qids], batch_size=batch,
                  normalize_embeddings=True, show_progress_bar=False,
                  convert_to_numpy=True)

    out = {}
    for i, q in enumerate(qids):
        cis = [pos[c[1]] for c in pairs[q]['cand']]
        out[q] = ce[cis] @ qe[i] if cis else np.zeros(0)
    del m
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', default=PAIRS)
    p.add_argument('--models', nargs='*', default=['BAAI/bge-m3'])
    p.add_argument('--batch', type=int, default=64)
    args = p.parse_args()

    pairs = load_json(args.pairs)['items']
    train = load_json(os.path.join(BASE, 'train.json'))
    qids = list(pairs)
    questions = {q: train[q]['question'] for q in qids}
    print(f'{len(qids)} cau | {sum(len(v["cand"]) for v in pairs.values())} cap ung vien\n')

    rows = []
    for name, fn in [('halong goc (e5-base 278M)', lambda: from_memmap('emb', pairs, qids, questions)),
                     ('halong-ft (da fine-tune)', lambda: from_memmap('emb_ft', pairs, qids, questions))]:
        print(f'--- {name} ---')
        sc = fn()
        auc, rec = metrics(pairs, lambda q: sc[q])
        rows.append((name, auc, rec))
        print(f'  AUC {auc:.4f} | Recall@5 (dense thuan) {rec.mean():.4f}\n')

    for mid in args.models:
        print(f'--- {mid} (zero-shot) ---')
        try:
            sc = from_model(mid, pairs, qids, questions, args.batch)
        except Exception as e:
            print(f'  LOI: {type(e).__name__}: {str(e)[:200]}\n')
            continue
        auc, rec = metrics(pairs, lambda q: sc[q])
        rows.append((mid, auc, rec))
        print(f'  AUC {auc:.4f} | Recall@5 (dense thuan) {rec.mean():.4f}\n')

    print('=' * 72)
    print(f'{"model":<40}{"AUC":>9}{"Recall@5":>11}')
    print('=' * 72)
    for name, auc, rec in sorted(rows, key=lambda r: -r[1]):
        print(f'{name:<40}{auc:>9.4f}{rec.mean():>11.4f}')
    base = next((r for r in rows if r[0].startswith('halong goc')), None)
    if base and len(rows) > 2:
        print('\nSo voi halong GOC (cung la zero-shot, so sanh cong bang):')
        rng = np.random.default_rng(0)
        n = len(base[2])
        boot = rng.integers(0, n, size=(4000, n))
        for name, auc, rec in rows:
            if name == base[0]:
                continue
            d = rec - base[2]
            lo, hi = np.percentile(d[boot].mean(axis=1), [2.5, 97.5])
            print(f'  {name:<38}{d.mean():>+9.4f}  CI [{lo:+.4f}, {hi:+.4f}]')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
