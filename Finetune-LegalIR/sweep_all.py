#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Quet lai toan bo he so sau khi doi sang model fine-tune.

Moi tham so da tune truoc day deu tune cho MODEL CU, nen phai xem lai het.

Meo hieu qua: voi moi cau hoi chi cham diem MOT LAN (BM25 + dense tren pool lon
nhat), roi moi to hop pool/alpha/agg deu tinh lai tu bo diem do - gan nhu mien phi.
Nho vay quet 3 x 6 x 5 = 90 to hop chi ton bang mot lan chay binh thuong.

    pool  : so chunk ung vien lay tu BM25
    alpha : trong so DENSE khi hoa diem (con lai la BM25)
    agg   : gop chunk -> document (max / trung binh k chunk cao nhat)

Usage:
    python sweep_all.py --emb ../Retrieval-LegalIR/emb_ft -n 1000
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Retrieval-LegalIR'))
from hybrid import Hybrid, minmax          # noqa: E402

SPLIT_SEED = 42
POOLS = [500, 1000, 2000, 3000]
ALPHAS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
AGGS = [1, 2, 3, 5]          # trung binh k chunk cao nhat (1 = max)


def val_questions(train_path, split_path):
    with open(train_path, encoding='utf-8-sig') as f:
        train = json.load(f)
    if split_path and os.path.exists(split_path):
        with open(split_path, encoding='utf-8') as f:
            val = json.load(f)['val']
    else:
        qids = sorted(train.keys())
        random.Random(SPLIT_SEED).shuffle(qids)
        val = qids[-1000:]
    return [(train[q]['question'], {str(d) for d in train[q]['answer']}) for q in val]


def rank_docs(chunk_doc, idx, scores, k_agg, topk=5):
    """Gop diem chunk ve document bang trung binh k chunk cao nhat."""
    order = np.argsort(-scores)
    vals = {}
    for i in order:
        vals.setdefault(int(chunk_doc[idx[i]]), []).append(float(scores[i]))
    out = [(str(d), v[0] if k_agg == 1 else float(np.mean(v[:k_agg])))
           for d, v in vals.items()]
    out.sort(key=lambda x: -x[1])
    return {d for d, _ in out[:topk]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb', '-e', default='../Retrieval-LegalIR/emb_ft')
    p.add_argument('--index', default='../Retrieval-LegalIR/index')
    p.add_argument('--train', default='../LegalIR - Public Test/train.json')
    p.add_argument('--split', default='data/split.json')
    p.add_argument('--k1', type=float, default=2.5)
    p.add_argument('--b', type=float, default=0.9)
    p.add_argument('-n', type=int, default=1000)
    args = p.parse_args()

    qs = val_questions(args.train, args.split)[:args.n]
    h = Hybrid(args.index, args.emb, k1=args.k1, b=args.b)
    print(f'Model : {h.emeta["model"]}')
    print(f'BM25  : k1={args.k1} b={args.b}')
    print(f'Do tren {len(qs)} cau validation | '
          f'{len(POOLS)}x{len(ALPHAS)}x{len(AGGS)} = '
          f'{len(POOLS) * len(ALPHAS) * len(AGGS)} to hop')

    pool_max = max(POOLS)
    texts = [q for q, _ in qs]
    qv = np.concatenate([h.encode_query(texts[i:i + 64])
                         for i in range(0, len(texts), 64)])

    acc = {(p_, a, k): [] for p_ in POOLS for a in ALPHAS for k in AGGS}
    ceil = {p_: [] for p_ in POOLS}
    t0 = time.time()

    for qi, (q, gold) in enumerate(qs):
        sc, _ = h.bm25.score(q)
        nz = np.flatnonzero(sc)
        if nz.size == 0:
            for k in acc:
                acc[k].append(0.0)
            continue
        if nz.size > pool_max:
            nz = nz[np.argpartition(-sc[nz], pool_max - 1)[:pool_max]]
        order = nz[np.argsort(-sc[nz])]        # xep theo BM25 giam dan
        bm_all = sc[order]

        srt = np.argsort(order)
        vecs = np.empty((len(order), h.dim), dtype=np.float32)
        vecs[srt] = h.emb[order[srt]].astype(np.float32)
        dn_all = vecs @ qv[qi]

        for p_ in POOLS:
            n = min(p_, len(order))
            idx = order[:n]
            bmn = minmax(bm_all[:n])
            dnn = minmax(dn_all[:n])
            ceil[p_].append(
                len(gold & {str(d) for d in h.bm25.chunk_doc[idx]}) / len(gold))
            for a in ALPHAS:
                fused = a * dnn + (1 - a) * bmn
                for k in AGGS:
                    pred = rank_docs(h.bm25.chunk_doc, idx, fused, k)
                    acc[(p_, a, k)].append(len(gold & pred) / len(gold))
        if (qi + 1) % 200 == 0:
            print(f'  {qi + 1}/{len(qs)} ({time.time() - t0:.0f}s)', flush=True)

    print()
    print('TRAN UNG VIEN theo pool (gold co nam trong pool khong)')
    print('-' * 40)
    for p_ in POOLS:
        print(f'  pool {p_:>5} : {np.mean(ceil[p_]):.4f}')

    for k in AGGS:
        lab = 'max' if k == 1 else f'top{k} mean'
        print()
        print(f'AGG = {lab}')
        print('-' * (14 + 9 * len(ALPHAS)))
        print(f'{"pool":>6}' + ''.join(f'{f"a={a}":>9}' for a in ALPHAS))
        print('-' * (14 + 9 * len(ALPHAS)))
        for p_ in POOLS:
            row = f'{p_:>6}'
            for a in ALPHAS:
                row += f'{np.mean(acc[(p_, a, k)]):>9.4f}'
            print(row)

    best = max(acc, key=lambda k: np.mean(acc[k]))
    cur = np.mean(acc[(1000, 0.7, 3)])
    print()
    print('=' * 56)
    print(f'Dang dung : pool 1000, alpha 0.7, top3 mean -> {cur:.4f}')
    print(f'Tot nhat  : pool {best[0]}, alpha {best[1]}, '
          f'{"max" if best[2] == 1 else f"top{best[2]} mean"} -> {np.mean(acc[best]):.4f}'
          f'  ({np.mean(acc[best]) - cur:+.4f})')
    print('=' * 56)
    print(f'{time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
