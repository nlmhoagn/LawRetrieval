#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quet k1/b cua BM25 tren train.json.

k1 thap -> tu lap lai nhieu lan bao hoa nhanh hon (van ban dai nhoi tu khoa het loi the)
b  cao  -> phat chunk dai manh hon

Khong can build lai index: postings luu tf tho, doi k1/b chi doi mau so.

Usage:
    python sweep_bm25.py --index index --train "../LegalIR - Public Test/train.json" -n 200
"""

import argparse
import json
import random
import sys
import time

import numpy as np

from bm25 import BM25Index

K1S = [0.6, 0.9, 1.2, 1.5, 2.0]
BS = [0.4, 0.6, 0.75, 0.9]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', '-i', default='index')
    p.add_argument('--train', '-t', required=True)
    p.add_argument('-n', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--agg', default='top3')
    p.add_argument('--topk', '-k', type=int, default=5)
    args = p.parse_args()

    with open(args.train, encoding='utf-8-sig') as f:
        train = json.load(f)
    items = sorted(train.items())
    random.Random(args.seed).shuffle(items)
    items = items[:args.n]
    qs = [(v['question'], {str(d) for d in v['answer']}) for _, v in items]

    idx = BM25Index(args.index)
    K = args.topk
    t0 = time.time()
    grid = {}

    for k1 in K1S:
        for b in BS:
            idx.set_params(k1, b)
            rec = []
            for q, gold in qs:
                scores, _ = idx.score(q)
                pred = [d for d, _, _, _ in idx.rank_docs(scores, topk=K, agg=args.agg)]
                rec.append(len(gold & set(pred)) / len(gold))
            grid[(k1, b)] = float(np.mean(rec))
            print(f'  k1={k1:<4} b={b:<5} Recall@{K}={grid[(k1, b)]:.4f} '
                  f'({time.time() - t0:.0f}s)', flush=True)

    print()
    print('=' * 60)
    print(f'Recall@{K} tren {len(qs)} cau train (gop: {args.agg})')
    print('=' * 60)
    print('k1 \\ b  ' + ''.join(f'{b:>10}' for b in BS))
    print('-' * 60)
    best = max(grid, key=grid.get)
    for k1 in K1S:
        row = f'{k1:<8}'
        for b in BS:
            v = grid[(k1, b)]
            row += f'{v:>10.4f}' if (k1, b) != best else f'{v:>9.4f}*'
        print(row)
    print('-' * 60)
    print(f'Tot nhat: k1={best[0]}, b={best[1]} -> {grid[best]:.4f}')
    base = grid.get((1.5, 0.75))
    if base is not None:
        print(f'Mac dinh k1=1.5, b=0.75    -> {base:.4f}  '
              f'(chenh {grid[best] - base:+.4f})')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
