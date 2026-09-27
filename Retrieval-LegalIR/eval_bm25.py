#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do Recall@k cua BM25 tren train.json, theo dung cong thuc cham diem cua BTC:
    Recall  = trung binh |gold & pred| / |gold|
    Precision = trung binh |gold & pred| / |pred|

So sanh cac cach gop diem chunk -> document.

Usage:
    python eval_bm25.py --index index --train "../LegalIR - Public Test/train.json" -n 300
"""

import argparse
import json
import random
import sys
import time

import numpy as np

from bm25 import BM25Index

AGGS = ['max', 'top3', 'top3_log']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', '-i', default='index')
    p.add_argument('--train', '-t', required=True)
    p.add_argument('-n', type=int, default=300, help='So cau hoi lay mau (0 = tat ca)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--topk', '-k', type=int, default=5)
    args = p.parse_args()

    with open(args.train, encoding='utf-8-sig') as f:
        train = json.load(f)
    items = sorted(train.items())
    random.Random(args.seed).shuffle(items)
    if args.n:
        items = items[:args.n]

    idx = BM25Index(args.index)
    K = args.topk
    stats = {a: {'recall': [], 'prec': [], 'hit': [], 'rank': []} for a in AGGS}

    t0 = time.time()
    for i, (qid, v) in enumerate(items, 1):
        gold = {str(d) for d in v['answer']}
        scores, _ = idx.score(v['question'])
        for agg in AGGS:
            ranked = idx.rank_docs(scores, topk=max(K, 20), agg=agg)
            pred_all = [d for d, _, _, _ in ranked]
            pred = pred_all[:K]
            inter = len(gold & set(pred))
            stats[agg]['recall'].append(inter / len(gold))
            stats[agg]['prec'].append(inter / len(pred) if pred else 0.0)
            stats[agg]['hit'].append(1.0 if inter else 0.0)
            r = next((j + 1 for j, d in enumerate(pred_all) if d in gold), 0)
            stats[agg]['rank'].append(r)
        if i % 50 == 0:
            print(f'  {i}/{len(items)} cau ({time.time() - t0:.0f}s)', flush=True)

    n = len(items)
    print()
    print('=' * 74)
    print(f'BM25 chunk-level tren {n} cau hoi train (top-{K} van ban)')
    print('=' * 74)
    print(f'{"cach gop":<12}{"Recall@5":>11}{"Precision@5":>13}{"Hit@5":>9}{"MRR@20":>9}')
    print('-' * 74)
    for agg in AGGS:
        s = stats[agg]
        mrr = float(np.mean([1.0 / r if r else 0.0 for r in s['rank']]))
        print(f'{agg:<12}{np.mean(s["recall"]):>11.4f}{np.mean(s["prec"]):>13.4f}'
              f'{np.mean(s["hit"]):>9.4f}{mrr:>9.4f}')
    print('-' * 74)

    best = max(AGGS, key=lambda a: np.mean(stats[a]['recall']))
    ranks = stats[best]['rank']
    print(f'\nPhan bo thu hang cua gold (cach gop "{best}"):')
    for lo, hi, lab in ((1, 1, 'hang 1'), (2, 3, 'hang 2-3'), (4, 5, 'hang 4-5'),
                        (6, 10, 'hang 6-10'), (11, 20, 'hang 11-20')):
        c = sum(1 for r in ranks if r and lo <= r <= hi)
        print(f'  {lab:<12}: {c:>4} ({c / n * 100:5.1f}%)')
    miss = sum(1 for r in ranks if not r)
    print(f'  {"ngoai top20":<12}: {miss:>4} ({miss / n * 100:5.1f}%)')
    print(f'\nThoi gian: {time.time() - t0:.0f}s cho {n} cau '
          f'({(time.time() - t0) / n * 1000:.0f} ms/cau)')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
