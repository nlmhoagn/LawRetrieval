#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Ket hop mat chu (BM25) va ngu nghia (halong_embedding).

Kien truc: BM25 lay pool -> dense cham lai pool -> hoa diem -> gop ve document.

Vi sao khong dense search toan cuc: do tren train, pool 1000 chunk dau cua BM25
da chua 99.0% gold (pool 2000 -> 99.67%). Nen viec con lai la XEP LAI THU TU chu
khong phai tim them. Cach nay chi doc ~1000 vector tu memmap (~3MB/query) thay vi
nap ca 0.75GB vector vao RAM - quan trong voi may chi con 0.5GB trong.

Hoa diem:
  rrf    - Reciprocal Rank Fusion, chi dung THU HANG nen mien nhiem chenh lech thang do
  linear - chuan hoa min-max trong pool roi tron theo alpha

Usage:
    python hybrid.py --query "lệ phí làm căn cước công dân" --topk 5
    python hybrid.py --eval "../LegalIR - Public Test/train.json" -n 300
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from bm25 import BM25Index

MODEL_ID = 'hiieu/halong_embedding'
RRF_K = 60


def minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


class Hybrid:
    def __init__(self, index_dir='index', emb_dir='emb', k1=3.0, b=0.75, device=None,
                 qmodel=None):
        self.bm25 = BM25Index(index_dir, k1=k1, b=b)
        with open(os.path.join(emb_dir, 'meta.json'), encoding='utf-8') as f:
            self.emeta = json.load(f)
        dt = np.float16 if '16' in self.emeta['dtype'] else np.float32
        self.dim = self.emeta['dim']
        self.emb = np.memmap(os.path.join(emb_dir, self.emeta['file']), dtype=dt,
                             mode='r', shape=(self.emeta['n'], self.dim))
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Encode cau hoi PHAI dung dung model da encode van ban, neu khong hai ben
        # nam o hai khong gian vector khac nhau. Truoc day cho cung MODEL_ID nen khi
        # doi sang emb_ft (sinh boi halong-ft) thi cau hoi van dung model goc.
        self.qmodel = qmodel or self.emeta.get('model') or MODEL_ID
        if ('/' in self.qmodel or '\\' in self.qmodel) and not os.path.exists(self.qmodel):
            base_name = os.path.basename(os.path.normpath(self.qmodel))
            alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Finetune-LegalIR', 'models', base_name)
            if os.path.exists(alt):
                self.qmodel = os.path.abspath(alt)
        # Cau hoi phai gop token y het cach da gop cho van ban, neu khong hai ben
        # lech khong gian. emb/meta.json ghi lai cach gop luc encode.
        self.pooling = self.emeta.get('pooling', 'mean')
        self.tok = AutoTokenizer.from_pretrained(self.qmodel)
        self.model = AutoModel.from_pretrained(
            self.qmodel, dtype=torch.float16 if self.dev == 'cuda' else torch.float32)
        self.model.to(self.dev).eval()

    def encode_query(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=512,
                       return_tensors='pt').to(self.dev)
        with torch.no_grad():
            h = self.model(**enc).last_hidden_state
            if self.pooling == 'cls':
                v = h[:, 0]
            else:
                m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            if self.dim < v.shape[1]:
                v = v[:, :self.dim]
            v = torch.nn.functional.normalize(v.float(), p=2, dim=1)
        return v.cpu().numpy()

    def fused_chunk_scores(self, query, pool=1000, fusion='rrf', alpha=0.5, qvec=None):
        """Tra ve (chunk_idx, diem_hoa) cho cac chunk trong pool.

        qvec: vector cau hoi tinh san (dung khi chay lo, khoi encode lai tung cau).
        """
        sc, _ = self.bm25.score(query)
        nz = np.flatnonzero(sc)
        if nz.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if nz.size > pool:
            nz = nz[np.argpartition(-sc[nz], pool - 1)[:pool]]
        order = nz[np.argsort(-sc[nz])]          # da sap theo BM25 giam dan
        bm = sc[order]

        # doc vector cua pool: sap theo chi so de memmap doc tuan tu hon
        srt = np.argsort(order)
        vecs = np.empty((len(order), self.dim), dtype=np.float32)
        vecs[srt] = self.emb[order[srt]].astype(np.float32)
        dense = vecs @ (self.encode_query([query])[0] if qvec is None else qvec)

        if fusion == 'dense':
            return order, dense
        if fusion == 'bm25':
            return order, bm
        if fusion == 'linear':
            return order, alpha * minmax(dense) + (1 - alpha) * minmax(bm)

        # rrf: thu hang 0-based cua tung he thong
        r_bm = np.empty(len(order), dtype=np.float64)
        r_bm[np.argsort(-bm)] = np.arange(len(order))
        r_dn = np.empty(len(order), dtype=np.float64)
        r_dn[np.argsort(-dense)] = np.arange(len(order))
        return order, (1.0 / (RRF_K + 1 + r_bm) + 1.0 / (RRF_K + 1 + r_dn)).astype(np.float32)

    @staticmethod
    def _agg_score(vals, agg):
        """max | topN | max+W (trung binh N chunk cao nhat). vals da sap giam dan."""
        agg = agg.replace(' ', '')
        if agg == 'max':
            return vals[0]
        if agg.startswith('max+'):
            return vals[0] + float(agg[4:]) * (vals[1] if len(vals) > 1 else 0.0)
        if agg.startswith('top') and agg[3:].isdigit():
            return float(np.mean(vals[:int(agg[3:])]))
        raise ValueError(f'agg khong hop le: {agg}')

    def search_docs(self, query, topk=5, pool=1000, fusion='rrf', alpha=0.5, agg='top3',
                    qvec=None):
        order, fused = self.fused_chunk_scores(query, pool, fusion, alpha, qvec)
        if order.size == 0:
            return []
        rank = np.argsort(-fused)
        best, vals = {}, {}
        for i in rank:
            d = int(self.bm25.chunk_doc[order[i]])
            if d not in best:
                best[d] = int(order[i])
                vals[d] = []
            vals[d].append(float(fused[i]))
        out = [(str(d), self._agg_score(v, agg), best[d], len(v))
               for d, v in vals.items()]
        out.sort(key=lambda x: -x[1])
        return out[:topk]


def run_eval(h, train_path, n, seed, pool, topk):
    with open(train_path, encoding='utf-8-sig') as f:
        train = json.load(f)
    items = sorted(train.items())
    random.Random(seed).shuffle(items)
    items = items[:n]
    qs = [(v['question'], {str(d) for d in v['answer']}) for _, v in items]

    configs = [('bm25 (goc)', 'bm25', 0.0), ('dense (thuan)', 'dense', 0.0),
               ('rrf', 'rrf', 0.0)] + \
              [(f'linear a={a}', 'linear', a) for a in (0.3, 0.5, 0.7)]

    print(f'{len(qs)} cau train (seed {seed}), pool {pool} chunk, top-{topk} van ban')
    print('-' * 56)
    print(f'{"cau hinh":<18}{"Recall@5":>11}{"Hit@5":>9}{"delta":>10}')
    print('-' * 56)
    base = None
    for lab, fus, a in configs:
        rec, hit = [], []
        t0 = time.time()
        for q, gold in qs:
            pred = [d for d, _, _, _ in h.search_docs(q, topk=topk, pool=pool,
                                                      fusion=fus, alpha=a)]
            inter = len(gold & set(pred))
            rec.append(inter / len(gold))
            hit.append(1.0 if inter else 0.0)
        r = float(np.mean(rec))
        if base is None:
            base = r
        print(f'{lab:<18}{r:>11.4f}{np.mean(hit):>9.4f}{r - base:>+10.4f}'
              f'   ({time.time() - t0:.0f}s)')
    print('-' * 56)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', default='index')
    p.add_argument('--emb', default='emb')
    p.add_argument('--query', '-q')
    p.add_argument('--eval', '-e', help='train.json de do')
    p.add_argument('-n', type=int, default=300)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--pool', type=int, default=1000)
    p.add_argument('--topk', '-k', type=int, default=5)
    p.add_argument('--fusion', default='rrf', choices=['rrf', 'linear', 'bm25', 'dense'])
    p.add_argument('--alpha', type=float, default=0.5)
    args = p.parse_args()

    h = Hybrid(args.index, args.emb)
    if args.eval:
        run_eval(h, args.eval, args.n, args.seed, args.pool, args.topk)
    elif args.query:
        for r, (d, s, ci, n) in enumerate(
                h.search_docs(args.query, args.topk, args.pool, args.fusion, args.alpha), 1):
            c = h.bm25.read_chunk(ci)
            print(f'#{r}  {s:.4f}  doc {d:<8} {c["path"]}  ({n} chunk)')
            print(f'    {" ".join(c["text"].split(chr(10))[1:])[:150]}')
    else:
        p.error('Can --query hoac --eval')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
