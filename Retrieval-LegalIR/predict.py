#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Sinh bai nop tu file de bai (public-official.json / private-official.json).

Luon tra du 5 document_id: Recall la do do chinh, Precision chi dung de pha hoa,
nen tra thieu chi lam mat Recall chu khong duoc loi gi.

Usage:
    python predict.py --index index --questions "../LegalIR - Public Test/public-official.json" \
                      --out submission --train "../LegalIR - Public Test/train.json"
"""

import argparse
import collections
import json
import os
import sys
import time
import zipfile

import numpy as np

from bm25 import BM25Index

TOPK = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', '-i', default='index')
    p.add_argument('--questions', '-q', required=True, help='File de bai (answer = null)')
    p.add_argument('--out', '-o', default='submission', help='Ten file dau ra (khong duoi)')
    p.add_argument('--train', '-t', help='train.json - dung lam nguon do khi query khong khop gi')
    p.add_argument('--k1', type=float, default=3.0)
    p.add_argument('--b', type=float, default=0.75)
    p.add_argument('--agg', default='top3',
                   help='Gop chunk->doc: max | topN (vd top2, top3)')
    p.add_argument('--emb', help='Thu muc vector -> bat che do hybrid BM25 + dense')
    p.add_argument('--fusion', default='linear', choices=['linear', 'rrf'])
    p.add_argument('--alpha', type=float, default=0.7, help='Trong so DENSE (con lai la BM25)')
    p.add_argument('--pool', type=int, default=1000)
    p.add_argument('--rerank', help='Model cross-encoder -> bat tang 3')
    p.add_argument('--beta', type=float, default=0.6,
                   help='Trong so cross-encoder khi tron voi diem hybrid (1.0 = thay the han)')
    p.add_argument('--ndocs', type=int, default=20)
    p.add_argument('--mchunks', type=int, default=3)
    args = p.parse_args()

    with open(args.questions, encoding='utf-8-sig') as f:
        questions = json.load(f)

    # Nguon do: van ban hay la dap an nhat trong train. Chi dung khi BM25 tra ve
    # duoi 5 ket qua - tha doan bua con hon bo trong o, vi Recall khong phat.
    fallback = []
    if args.train:
        with open(args.train, encoding='utf-8-sig') as f:
            train = json.load(f)
        freq = collections.Counter(str(d) for v in train.values() for d in v['answer'])
        fallback = [d for d, _ in freq.most_common(20)]

    hyb = None
    rr = None
    if args.emb:
        from hybrid import Hybrid
        hyb = Hybrid(args.index, args.emb, k1=args.k1, b=args.b)
        idx = hyb.bm25
        print(f'Che do : HYBRID | fusion={args.fusion} alpha={args.alpha} '
              f'({int(args.alpha * 100)}% dense / {int((1 - args.alpha) * 100)}% BM25) '
              f'pool={args.pool}')
        if args.rerank:
            from rerank import Reranker, rerank_docs
            rr = Reranker(args.rerank, batch=16)
            print(f'Tang 3 : CROSS-ENCODER {args.rerank} | beta={args.beta} '
                  f'| {args.ndocs} van ban x {args.mchunks} chunk')
    else:
        idx = BM25Index(args.index, k1=args.k1, b=args.b)
        print('Che do : BM25 thuan')
    print(f'Index  : {idx.meta["n_chunks"]:,} chunk | k1={args.k1} b={args.b} agg={args.agg}')
    print(f'De bai : {len(questions)} cau hoi')

    qids = list(questions.keys())
    texts = [questions[q]['question'] for q in qids]

    qvecs = None
    if hyb is not None:
        t = time.time()
        qvecs = np.concatenate([hyb.encode_query(texts[i:i + 64])
                                for i in range(0, len(texts), 64)])
        print(f'  encode {len(texts)} cau hoi: {time.time() - t:.0f}s')

    submission = {}
    n_padded = 0
    t0 = time.time()
    for i, qid in enumerate(qids, 1):
        if rr is not None:
            ranked = rerank_docs(hyb, rr, questions[qid]['question'], topk=TOPK,
                                 pool=args.pool, alpha=args.alpha, agg=args.agg,
                                 n_docs=args.ndocs, m_chunks=args.mchunks,
                                 qvec=qvecs[i - 1], beta=args.beta)
            preds = [d for d, _, _, _ in ranked]
        elif hyb is not None:
            ranked = hyb.search_docs(questions[qid]['question'], topk=TOPK, pool=args.pool,
                                     fusion=args.fusion, alpha=args.alpha, agg=args.agg,
                                     qvec=qvecs[i - 1])
            preds = [d for d, _, _, _ in ranked]
        else:
            scores, _ = idx.score(questions[qid]['question'])
            preds = [d for d, _, _, _ in idx.rank_docs(scores, topk=TOPK, agg=args.agg)]
        if len(preds) < TOPK:
            n_padded += 1
            for d in fallback:
                if len(preds) >= TOPK:
                    break
                if d not in preds:
                    preds.append(d)
        submission[qid] = {'answer': [str(d) for d in preds[:TOPK]]}
        if i % 200 == 0:
            print(f'  {i}/{len(questions)} ({time.time() - t0:.0f}s)', flush=True)

    json_path = f'{args.out}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False)

    zip_path = f'{args.out}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # ten trong zip PHAI la submission.json va nam o goc
        zf.write(json_path, 'submission.json')

    lens = collections.Counter(len(v['answer']) for v in submission.values())
    uniq = len({d for v in submission.values() for d in v['answer']})
    print()
    print(f'Xong {len(submission)} cau trong {time.time() - t0:.0f}s')
    print(f'  So ID/cau      : ' + ', '.join(f'{k}: {v} cau' for k, v in sorted(lens.items())))
    print(f'  Van ban rieng  : {uniq}')
    print(f'  Cau phai do them: {n_padded}')
    print(f'  -> {json_path} ({os.path.getsize(json_path) / 1024:.0f} KB)')
    print(f'  -> {zip_path} ({os.path.getsize(zip_path) / 1024:.0f} KB)')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
