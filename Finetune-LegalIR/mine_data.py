#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Dao du lieu huan luyen cho fine-tune bi-encoder.

Van de: nhan cua BTC o muc DOCUMENT, con model hoc o muc CHUNK. Van ban gold co
median 33 chunk, phai chon 1 lam positive.
  -> chon chunk co diem hybrid cao nhat trong van ban gold (giam sat yeu).
  RUI RO: model hoc dong y voi chinh no. Chap nhan vi day la cach kha thi duy nhat
  khi khong co nhan muc chunk.

Hard negative: chunk thuoc van ban KHONG phai gold, lay tu hang SKIP_TOP tro di.
Bo qua vai hang dau vi corpus co 5.427 van ban chua tung la dap an - rat co the
chung van lien quan, chi la khong duoc gan nhan (false negative).

Chia tap: 1000 cau cuoi lam validation, KHONG dua vao huan luyen.

Usage:
    python mine_data.py --train "../LegalIR - Public Test/train.json" --out data
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
SKIP_TOP = 5        # bo qua N ung vien dau khi lay negative (tranh false negative)
N_NEG = 6           # so hard negative moi cau hoi
NEG_UNTIL = 80      # chi lay negative trong pham vi hang nay


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', '-t', required=True)
    p.add_argument('--index', default='../Retrieval-LegalIR/index')
    p.add_argument('--emb', default='../Retrieval-LegalIR/emb')
    p.add_argument('--out', '-o', default='data')
    p.add_argument('--k1', type=float, default=2.5)
    p.add_argument('--b', type=float, default=0.9)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--pool', type=int, default=1000)
    p.add_argument('--skip-top', type=int, default=SKIP_TOP,
                   help='Bo qua N ung vien dau khi lay negative. 0 = lay ca hang '
                        'cao nhat - chinh la nhom dang danh bai reranker hien tai.')
    p.add_argument('--neg', type=int, default=N_NEG, help='So hard negative moi cau')
    p.add_argument('--neg-until', type=int, default=NEG_UNTIL)
    p.add_argument('--pos-by', default='hybrid', choices=['hybrid', 'ce'],
                   help="Cach chon positive trong van ban gold. 'hybrid' = chunk co "
                        "diem hybrid cao nhat (model tu dong y voi chinh no). 'ce' = "
                        "cho cross-encoder cham lai TOP-N chunk hybrid roi chon - y "
                        "kien doc lap hon. Do tren 200 van ban: hai cach chon cung "
                        "chunk 64.5%% so lan.")
    p.add_argument('--pos-topn', type=int, default=5,
                   help='So chunk hybrid dan dau dua cho CE cham lai')
    p.add_argument('--reranker', default='models/reranker-ft')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.train, encoding='utf-8-sig') as f:
        train = json.load(f)

    qids = sorted(train.keys())
    random.Random(SPLIT_SEED).shuffle(qids)
    val_ids = set(qids[-N_VAL:])
    trn_ids = qids[:-N_VAL]
    print(f'Chia tap: {len(trn_ids)} huan luyen / {len(val_ids)} validation '
          f'(seed {SPLIT_SEED})')

    with open(os.path.join(args.out, 'split.json'), 'w', encoding='utf-8') as f:
        json.dump({'train': trn_ids, 'val': sorted(val_ids), 'seed': SPLIT_SEED}, f)

    h = Hybrid(args.index, args.emb, k1=args.k1, b=args.b)
    rr = None
    if args.pos_by == 'ce':
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'Retrieval-LegalIR'))
        from rerank import Reranker
        rr = Reranker(args.reranker, batch=16)
    print(f'Dao bang: {args.emb} | query encoder {h.qmodel}')
    print(f'Positive: {args.pos_by}' + (f' (CE cham lai {args.pos_topn} chunk dau)'
                                        if rr else ''))
    print(f'Negative: hang {args.skip_top}-{args.neg_until}, toi da {args.neg} moi cau')

    rows = []
    n_no_gold_chunk = 0
    n_short_neg = 0
    t0 = time.time()

    for i, qid in enumerate(trn_ids, 1):
        q = train[qid]['question']
        gold = {int(d) for d in train[qid]['answer']}
        order, fused = h.fused_chunk_scores(q, args.pool, 'linear', args.alpha)
        if order.size == 0:
            n_no_gold_chunk += 1
            continue
        rank = np.argsort(-fused)

        # positive: chunk diem cao nhat thuoc van ban gold
        gold_cis = [int(order[j]) for j in rank
                    if int(h.bm25.chunk_doc[order[j]]) in gold][:args.pos_topn]
        pos_ci = gold_cis[0] if gold_cis else None
        if rr is not None and len(gold_cis) > 1:
            txt = [h.bm25.read_chunk(c)['text'] for c in gold_cis]
            pos_ci = gold_cis[int(np.argmax(rr.score(q, txt)))]
        if pos_ci is None:
            # gold khong co chunk nao trong pool -> lay chunk dau tien cua van ban gold
            cand = np.flatnonzero(np.isin(h.bm25.chunk_doc, list(gold)))
            if cand.size == 0:
                n_no_gold_chunk += 1
                continue
            pos_ci = int(cand[0])

        # hard negative: van ban khong phai gold, tu hang SKIP_TOP den NEG_UNTIL
        negs, seen_doc = [], set()
        for j in rank[args.skip_top:args.neg_until]:
            ci = int(order[j])
            d = int(h.bm25.chunk_doc[ci])
            if d in gold or d in seen_doc:
                continue
            seen_doc.add(d)
            negs.append(ci)
            if len(negs) >= args.neg:
                break
        if len(negs) < args.neg:
            n_short_neg += 1
        if not negs:
            continue

        rows.append({'qid': qid, 'query': q, 'pos': pos_ci, 'negs': negs})
        if i % 500 == 0:
            print(f'  {i}/{len(trn_ids)} ({time.time() - t0:.0f}s)', flush=True)

    # ghi ra jsonl kem van ban thuc te
    out_path = os.path.join(args.out, 'pairs.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in rows:
            rec = {
                'qid': r['qid'],
                'query': r['query'],
                'positive': h.bm25.read_chunk(r['pos'])['text'],
                'negatives': [h.bm25.read_chunk(c)['text'] for c in r['negs']],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print()
    print(f'Xong {len(rows)} cap trong {time.time() - t0:.0f}s')
    print(f'  Cau bi bo (khong tim duoc chunk gold): {n_no_gold_chunk}')
    print(f'  Cau co it hon {args.neg} negative        : {n_short_neg}')
    print(f'  -> {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)')
    print(f'  -> {os.path.join(args.out, "split.json")}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
