#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Do model da fine-tune tren tap validation 1.000 cau (KHONG dung de huan luyen).

Moc da do san voi model goc:
    BM25 thuan     0.8112
    dense thuan    0.8307     <- con so can cai thien
    hybrid a=0.7   0.8633

LUU Y QUAN TRONG: alpha PHAI quet lai. Neu dense manh len thi ty le tron toi uu
chac chan dich chuyen (co the len 0.8-0.9). Do dense thuan roi giu nguyen alpha=0.7
la bo phi phan lon loi ich.

Usage:
    python eval_ft.py --emb ../Retrieval-LegalIR/emb_ft
    python eval_ft.py --emb ../Retrieval-LegalIR/emb_ft --base ../Retrieval-LegalIR/emb
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
from hybrid import Hybrid          # noqa: E402

SPLIT_SEED = 42
N_VAL = 1000
ALPHAS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def val_questions(train_path, split_path=None):
    with open(train_path, encoding='utf-8-sig') as f:
        train = json.load(f)
    if split_path and os.path.exists(split_path):
        with open(split_path, encoding='utf-8') as f:
            val = json.load(f)['val']
    else:
        qids = sorted(train.keys())
        random.Random(SPLIT_SEED).shuffle(qids)
        val = qids[-N_VAL:]
    return [(train[q]['question'], {str(d) for d in train[q]['answer']}) for q in val]


def evaluate(h, qs, label):
    texts = [q for q, _ in qs]
    t0 = time.time()
    qv = np.concatenate([h.encode_query(texts[i:i + 64])
                         for i in range(0, len(texts), 64)])

    print(f'\n{"=" * 60}')
    print(f'{label}   ({len(qs)} cau validation)')
    print('=' * 60)
    print(f'{"alpha":>7}{"":4}{"Recall@5":>10}{"Hit@5":>9}   ghi chu')
    print('-' * 60)

    best = (None, -1)
    results = {}
    for a in ALPHAS:
        rec, hit = [], []
        for i, (q, gold) in enumerate(qs):
            fus = 'bm25' if a == 0.0 else ('dense' if a == 1.0 else 'linear')
            pred = [d for d, _, _, _ in h.search_docs(q, 5, 1000, fus, a, qvec=qv[i])]
            n = len(gold & set(pred))
            rec.append(n / len(gold))
            hit.append(1.0 if n else 0.0)
        r = float(np.mean(rec))
        results[a] = r
        note = 'BM25 thuan' if a == 0.0 else ('dense thuan' if a == 1.0 else '')
        print(f'{a:>7.1f}{"":4}{r:>10.4f}{np.mean(hit):>9.4f}   {note}')
        if r > best[1]:
            best = (a, r)
    print('-' * 60)
    print(f'Tot nhat: alpha={best[0]} -> {best[1]:.4f}   ({time.time() - t0:.0f}s)')
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb', '-e', required=True, help='Thu muc vector cua model fine-tune')
    p.add_argument('--base', help='Thu muc vector model goc, de so sanh truc tiep')
    p.add_argument('--index', default='../Retrieval-LegalIR/index')
    p.add_argument('--train', default='../LegalIR - Public Test/train.json')
    p.add_argument('--split', default='data/split.json')
    p.add_argument('--k1', type=float, default=2.5)
    p.add_argument('--b', type=float, default=0.9)
    args = p.parse_args()

    if not os.path.exists(args.train):
        sys.exit(f'Khong tim thay {args.train} - truyen --train cho dung duong dan')

    qs = val_questions(args.train, args.split)
    print(f'Tap validation: {len(qs)} cau (seed {SPLIT_SEED}, khong dung de huan luyen)')

    ft = Hybrid(args.index, args.emb, k1=args.k1, b=args.b)
    print(f'Model fine-tune: {ft.emeta["model"]}')
    res_ft = evaluate(ft, qs, 'MODEL FINE-TUNE')

    if args.base:
        del ft
        bs = Hybrid(args.index, args.base, k1=args.k1, b=args.b)
        print(f'\nModel goc: {bs.emeta["model"]}')
        res_bs = evaluate(bs, qs, 'MODEL GOC')

        print(f'\n{"=" * 60}')
        print('SO SANH')
        print('=' * 60)
        print(f'{"alpha":>7}{"goc":>12}{"fine-tune":>12}{"delta":>10}')
        print('-' * 60)
        for a in ALPHAS:
            d = res_ft[a] - res_bs[a]
            print(f'{a:>7.1f}{res_bs[a]:>12.4f}{res_ft[a]:>12.4f}{d:>+10.4f}')
        print('-' * 60)
        bb, bf = max(res_bs.values()), max(res_ft.values())
        print(f'Tot nhat: {bb:.4f} -> {bf:.4f}  ({bf - bb:+.4f})')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
