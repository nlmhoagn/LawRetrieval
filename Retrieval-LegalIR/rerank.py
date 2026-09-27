#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Tang 3: cross-encoder rerank.

Chuoi day du:
    BM25 (pool 1000 chunk)
      -> hybrid linear alpha=0.7  (bi-encoder, vector tinh san)
      -> lay top-N van ban, moi van ban up to M chunk manh nhat
      -> cross-encoder cham lai tung cap (cau hoi, chunk)
      -> gop ve document -> top-5

Khac biet voi bi-encoder: bi-encoder nen ca chunk thanh 1 vector TRUOC khi thay cau
hoi, nen khong "chu y" duoc vao dung doan duoc hoi. Cross-encoder doc cau hoi va
chunk CUNG LUC nen lam duoc, doi lai phai chay model o query time tren tung ung vien.

Vi sao lay theo VAN BAN chu khong phai top-K chunk: can du 5 van ban rieng o dau ra,
ma top-50 chunk co the chi phu vai van ban.

Usage:
    python rerank.py --eval "../LegalIR - Public Test/train.json" -n 300
    python rerank.py --model AITeamVN/Vietnamese_Reranker --bench
"""

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hybrid import Hybrid

DEFAULT_MODEL = 'AITeamVN/Vietnamese_Reranker'
MAX_LEN = 512


class Reranker:
    def __init__(self, model_id=DEFAULT_MODEL, device=None, batch=16):
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch = batch
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            # Vai repo chi co sentencepiece.bpe.model, thieu tokenizer.json nen
            # AutoTokenizer khong suy ra duoc lop. Kien truc la xlm-roberta thi
            # chi dinh thang, doc sentencepiece truc tiep.
            from transformers import XLMRobertaTokenizerFast
            self.tok = XLMRobertaTokenizerFast.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float16 if self.dev == 'cuda' else torch.float32,
            low_cpu_mem_usage=True)
        self.model.to(self.dev)
        self.model.eval()
        self.model_id = model_id

    def score(self, query, texts):
        out = np.empty(len(texts), dtype=np.float32)
        for s in range(0, len(texts), self.batch):
            part = texts[s:s + self.batch]
            enc = self.tok([query] * len(part), part, padding=True, truncation=True,
                           max_length=MAX_LEN, return_tensors='pt').to(self.dev)
            with torch.no_grad():
                logits = self.model(**enc).logits
            out[s:s + len(part)] = logits[:, 0].float().cpu().numpy()
        return out


class EnsembleReranker:
    """Gop nhieu cross-encoder, lay trung binh diem. Do duoc +0.006 Recall@5."""

    def __init__(self, model_ids, device=None, batch=16):
        self.rerankers = [Reranker(m, device, batch) for m in model_ids]
        self.model_id = 'ensemble[' + ','.join(m.split('/')[-1] for m in model_ids) + ']'
        print(f'Ensemble reranker: {len(self.rerankers)} model')

    def score(self, query, texts):
        scores = np.stack([rr.score(query, texts) for rr in self.rerankers])
        return scores.mean(axis=0)


def _agg(v, agg):
    """Gop diem cac chunk cua mot van ban thanh diem van ban. v da sap giam dan.

        max        - chi lay chunk manh nhat
        topN       - trung binh N chunk manh nhat
        max+W      - chunk manh nhat cong W lan chunk thu hai

    'max+W' co ly khi reranker da du gioi: van ban gold thuong chi co DUNG MOT
    doan chua cau tra loi, doan thu hai chi tam tam. Lay trung binh se pha loang
    tin hieu do, con van ban sai co hai doan deu deu lai noi len. Do tren 500 cau
    validation sach: max+0.4 cho 0.9377 so voi 0.9277 cua top2.
    """
    agg = agg.replace(' ', '')   # chong loi 'max + 0.4' bi fallback am tham
    if agg == 'max':
        return v[0]
    if agg.startswith('max+'):
        return v[0] + float(agg[4:]) * (v[1] if len(v) > 1 else 0.0)
    if agg.startswith('top') and agg[3:].isdigit():
        return float(np.mean(v[:int(agg[3:])]))
    raise ValueError(f"agg khong hop le: '{agg}'. Dung 'max', 'topN', hoac 'max+W'.")


def rerank_docs(h, rr, query, topk=5, pool=1000, alpha=0.7, agg='top3',
                n_docs=20, m_chunks=3, qvec=None, beta=1.0):
    """Hybrid lay ung vien -> cross-encoder cham lai -> top-k van ban.

    beta: trong so cua cross-encoder khi tron voi diem hybrid.
          1.0 = thay the han bang cross-encoder, 0.0 = giu nguyen hybrid.
    """
    order, fused = h.fused_chunk_scores(query, pool, 'linear', alpha, qvec)
    if order.size == 0:
        return []

    # gom chunk theo van ban, giu thu tu fused giam dan
    per_doc, fmap = {}, {}
    for i in np.argsort(-fused):
        ci = int(order[i])
        d = int(h.bm25.chunk_doc[ci])
        per_doc.setdefault(d, []).append(ci)
        fmap[ci] = float(fused[i])
        if len(per_doc) > n_docs and len(per_doc[d]) == 1:
            per_doc.pop(d)
            break

    cand = [(d, ci) for d, cis in per_doc.items() for ci in cis[:m_chunks]]
    if not cand:
        return []
    texts = [h.bm25.read_chunk(ci)['text'] for _, ci in cand]
    scores = rr.score(query, texts)

    if beta < 1.0:
        fh = np.array([fmap[ci] for _, ci in cand], dtype=np.float32)
        from hybrid import minmax
        scores = beta * minmax(scores) + (1 - beta) * minmax(fh)

    vals, best = {}, {}
    for (d, ci), s in sorted(zip(cand, scores), key=lambda x: -x[1]):
        vals.setdefault(d, []).append(float(s))
        best.setdefault(d, ci)
    out = [(str(d), _agg(v, agg), best[d], len(v)) for d, v in vals.items()]
    out.sort(key=lambda x: -x[1])
    return out[:topk]


def run_eval(h, rr, train_path, n, seed, pool, alpha, n_docs, m_chunks):
    with open(train_path, encoding='utf-8-sig') as f:
        train = json.load(f)
    items = sorted(train.items())
    random.Random(seed).shuffle(items)
    items = items[:n]
    qs = [(v['question'], {str(d) for d in v['answer']}) for _, v in items]

    print(f'{len(qs)} cau train (seed {seed}) | pool {pool} | alpha {alpha} | '
          f'rerank {n_docs} van ban x {m_chunks} chunk')
    print(f'reranker: {rr.model_id}')
    print('-' * 60)

    rec_h, rec_r, hit_h, hit_r = [], [], [], []
    t0 = time.time()
    for i, (q, gold) in enumerate(qs, 1):
        ph = [d for d, _, _, _ in h.search_docs(q, 5, pool, 'linear', alpha)]
        pr = [d for d, _, _, _ in rerank_docs(h, rr, q, 5, pool, alpha,
                                              n_docs=n_docs, m_chunks=m_chunks)]
        rec_h.append(len(gold & set(ph)) / len(gold))
        rec_r.append(len(gold & set(pr)) / len(gold))
        hit_h.append(1.0 if gold & set(ph) else 0.0)
        hit_r.append(1.0 if gold & set(pr) else 0.0)
        if i % 50 == 0:
            print(f'  {i}/{len(qs)}  ({time.time() - t0:.0f}s)', flush=True)

    print()
    print(f'{"cau hinh":<26}{"Recall@5":>11}{"Hit@5":>9}{"delta":>10}')
    print('-' * 60)
    print(f'{"hybrid (khong rerank)":<26}{np.mean(rec_h):>11.4f}{np.mean(hit_h):>9.4f}'
          f'{0.0:>+10.4f}')
    print(f'{"+ cross-encoder":<26}{np.mean(rec_r):>11.4f}{np.mean(hit_r):>9.4f}'
          f'{np.mean(rec_r) - np.mean(rec_h):>+10.4f}')
    print('-' * 60)
    print(f'{(time.time() - t0) / len(qs) * 1000:.0f} ms/cau')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', default='index')
    p.add_argument('--emb', default='emb')
    p.add_argument('--model', '-m', default=DEFAULT_MODEL)
    p.add_argument('--eval', '-e')
    p.add_argument('--query', '-q')
    p.add_argument('--bench', action='store_true', help='Do toc do cross-encoder')
    p.add_argument('-n', type=int, default=300)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--pool', type=int, default=1000)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--ndocs', type=int, default=20)
    p.add_argument('--mchunks', type=int, default=3)
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()

    h = Hybrid(args.index, args.emb)
    rr = Reranker(args.model, batch=args.batch)

    if args.bench:
        q = 'Lệ phí làm căn cước công dân là bao nhiêu'
        order, fused = h.fused_chunk_scores(q, args.pool, 'linear', args.alpha)
        texts = [h.bm25.read_chunk(int(i))['text'] for i in order[np.argsort(-fused)[:60]]]
        rr.score(q, texts[:8])                       # warmup
        t = time.time()
        rr.score(q, texts)
        el = time.time() - t
        print(f'{args.model}: {len(texts)} cap trong {el:.2f}s '
              f'-> {len(texts) / el:.0f} cap/s')
        print(f'  uoc 1000 cau x 60 cap: {60000 / (len(texts) / el) / 60:.0f} phut')
        if torch.cuda.is_available():
            print(f'  VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
        return 0

    if args.eval:
        run_eval(h, rr, args.eval, args.n, args.seed, args.pool, args.alpha,
                 args.ndocs, args.mchunks)
    elif args.query:
        for r, (d, s, ci, n) in enumerate(
                rerank_docs(h, rr, args.query, 5, args.pool, args.alpha,
                            n_docs=args.ndocs, m_chunks=args.mchunks), 1):
            c = h.bm25.read_chunk(ci)
            print(f'#{r}  {s:7.3f}  doc {d:<8} {c["path"]}')
    else:
        p.error('Can --eval, --query hoac --bench')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
