#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Quet alpha (va beta, lambda) ma chi chay cross-encoder MOT lan.

Meo: pool 2000 chunk do BM25 chon, KHONG phu thuoc alpha. Nen chi can dump diem
BM25 va diem dense THO cho ca pool, roi tinh lai
    fused = alpha*minmax(dense) + (1-alpha)*minmax(bm25)
cho alpha bat ky o ngoai. Diem cross-encoder cua mot cap (cau hoi, chunk) cung
khong phu thuoc alpha - chi can cham HOP cua cac tap ung vien sinh boi moi alpha.

    python sweep_alpha.py dump  -n 500 -o pool_val500.npz
    python sweep_alpha.py score -d pool_val500.npz -o sc_union.json
    python sweep_alpha.py sweep -d pool_val500.npz -s sc_union.json
"""

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'LegalIR - Public Test-20260806T081424Z-1-001', 'LegalIR - Public Test')
TRAIN = os.path.join(BASE, 'train.json')
SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                     'Finetune-LegalIR', 'data', 'split.json')
def _pick_reranker():
    """Uu tien ban fp16 neu co (nap nhanh hon), khong thi dung ban goc.
    Hai ban cho ket qua giong nhau tung bit - xem ghi chu trong prior.py."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'Finetune-LegalIR', 'models')
    fp16 = os.path.join(base, 'reranker-ft-fp16')
    return fp16 if os.path.isdir(fp16) else os.path.join(base, 'reranker-ft')


RERANKER = _pick_reranker()

POOL = 2000
NDOCS = 50
MCHUNKS = 2
ALPHAS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_LEN = 512


def load_json(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def mm(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


# ------------------------------------------------------------------ dump

def cmd_dump(args):
    from hybrid import Hybrid

    train = load_json(TRAIN)
    val = [str(q) for q in load_json(SPLIT)['val'] if str(q) in train]
    val = val[args.offset:args.offset + args.n] if args.n else val[args.offset:]
    h = Hybrid('index', args.emb, k1=2.5, b=0.9)
    print(f'{len(val)} cau | query encoder: {h.qmodel}')

    texts = [train[q]['question'] for q in val]
    qv = np.concatenate([h.encode_query(texts[i:i + 64])
                         for i in range(0, len(texts), 64)])

    idxs, bms, dns, offs = [], [], [], [0]
    t0 = time.time()
    for i, q in enumerate(val):
        sc, _ = h.bm25.score(texts[i])
        nz = np.flatnonzero(sc)
        if nz.size > POOL:
            nz = nz[np.argpartition(-sc[nz], POOL - 1)[:POOL]]
        order = nz[np.argsort(-sc[nz])]
        srt = np.argsort(order)
        vecs = np.empty((len(order), h.dim), dtype=np.float32)
        vecs[srt] = h.emb[order[srt]].astype(np.float32)
        idxs.append(order.astype(np.int32))
        bms.append(sc[order].astype(np.float32))
        dns.append((vecs @ qv[i]).astype(np.float32))
        offs.append(offs[-1] + len(order))
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(val)} ({time.time() - t0:.0f}s)', flush=True)

    np.savez_compressed(
        args.out,
        qids=np.array(val), offs=np.array(offs, dtype=np.int64),
        idx=np.concatenate(idxs), bm25=np.concatenate(bms), dense=np.concatenate(dns),
        chunk_doc=h.bm25.chunk_doc)
    with open(args.out + '.gold.json', 'w', encoding='utf-8') as f:
        json.dump({q: [str(d) for d in train[q]['answer']] for q in val}, f)
    print(f'\n{len(val)} cau, {offs[-1]} dong pool -> {args.out} '
          f'({os.path.getsize(args.out) / 1e6:.0f} MB)')
    return 0


# ------------------------------------------------------------------ chung

def _load(dump):
    # Phai vat chat hoa ra dict: NpzFile giai nen LAI ca mang moi lan truy cap khoa,
    # ma _cands doc 4 khoa cho tung cau tung alpha -> cham gap hang tram lan.
    with np.load(dump, allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    gold = load_json(dump + '.gold.json')
    return z, gold


def _cands(z, i, alpha, ndocs=NDOCS, mchunks=MCHUNKS):
    """Ung vien cua cau thu i o muc alpha: [(doc_id, chunk_idx, fused_score)]."""
    s, e = z['offs'][i], z['offs'][i + 1]
    if e <= s:
        return []
    idx = z['idx'][s:e]
    fused = (alpha * mm(z['dense'][s:e]) + (1 - alpha) * mm(z['bm25'][s:e])
             if alpha not in (0.0, 1.0) else
             (mm(z['dense'][s:e]) if alpha == 1.0 else mm(z['bm25'][s:e])))
    per_doc = {}
    out = []
    for j in np.argsort(-fused):
        ci = int(idx[j])
        d = int(z['chunk_doc'][ci])
        if d not in per_doc:
            if len(per_doc) >= ndocs:
                break
            per_doc[d] = 0
        if per_doc[d] < mchunks:
            per_doc[d] += 1
            out.append((str(d), ci, float(fused[j])))
    return out


# ------------------------------------------------------------------ score

def cmd_score(args):
    from rerank import Reranker

    z, gold = _load(args.dump)
    qids = list(z['qids'])
    union = {}
    for i, q in enumerate(qids):
        u = set()
        for a in ALPHAS:
            u.update(ci for _, ci, _ in _cands(z, i, a))
        union[q] = sorted(u)
    tot = sum(len(v) for v in union.values())
    print(f'{len(qids)} cau | hop ung vien qua {len(ALPHAS)} muc alpha: '
          f'{tot} cap ({tot / len(qids):.0f}/cau)')

    from bm25 import BM25Index
    bm = BM25Index('index')
    train = load_json(TRAIN)

    rr = Reranker(args.model, batch=args.batch)
    out, t0 = {}, time.time()
    for i, q in enumerate(qids):
        cis = union[q]
        if not cis:
            out[q] = {}
            continue
        txt = [bm.read_chunk(int(c))['text'] for c in cis]
        sc = rr.score(train[q]['question'], txt)
        out[q] = {str(c): float(s) for c, s in zip(cis, sc)}
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {i + 1}/{len(qids)} ({el:.0f}s, con ~{el / (i + 1) * (len(qids) - i - 1) / 60:.0f}p)',
                  flush=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'scores': out}, f)
    print(f'\nxong {time.time() - t0:.0f}s -> {args.out}')
    return 0


# ------------------------------------------------------------------ sweep

def _prior_weights(exclude):
    train = load_json(TRAIN)
    freq = collections.Counter(str(d) for q, v in train.items() if q not in exclude
                               for d in v['answer'])
    mx = math.log1p(max(freq.values()))
    return {d: math.log1p(c) / mx for d, c in freq.items()}


def cmd_sweep(args):
    z, gold = _load(args.dump)
    ce = load_json(args.scores)['scores']
    qids = list(z['qids'])
    n = len(qids)
    W = _prior_weights(set(map(str, load_json(SPLIT)['val'])))

    def recall(alpha, beta, lam):
        out = []
        for i, q in enumerate(qids):
            cand = _cands(z, i, alpha)
            if not cand:
                out.append(0.0)
                continue
            fh = np.array([c[2] for c in cand])
            cs = np.array([ce[q].get(str(c[1]), 0.0) for c in cand])
            s = beta * mm(cs) + (1 - beta) * mm(fh) if beta < 1.0 else mm(cs)
            vals = {}
            for j in np.argsort(-s):
                vals.setdefault(cand[j][0], []).append(float(s[j]))
            ds = sorted(((d, float(np.mean(x[:MCHUNKS]))) for d, x in vals.items()),
                        key=lambda t: -t[1])
            v = mm(np.array([x for _, x in ds]))
            if lam:
                v = v + lam * np.array([W.get(d, 0.0) for d, _ in ds])
            top = [ds[j][0] for j in np.argsort(-v)[:5]]
            g = set(gold[q])
            out.append(len(g & set(top)) / len(g))
        return np.array(out)

    half = n // 2
    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(4000, n))

    ra, rb, rl = args.ref
    ref = recall(ra, rb, rl)
    print(f'{n} cau | moc alpha={ra} beta={rb} lam={rl} -> {ref.mean():.4f}\n')

    # Quet luoi. Chon tren NUA DAU, kiem chung tren NUA SAU - cung quy trinh
    # muc 5 RESULTS.md, vi luoi cang lon thi cang de chon trung nhieu.
    rows = []
    for a in args.alphas:
        for b in args.betas:
            for l in args.lams:
                r = recall(a, b, l)
                rows.append((a, b, l, r))

    print(f'{"alpha":>6}{"beta":>6}{"lam":>6}{"Recall@5":>11}{"delta":>9}'
          f'{"nua dau":>9}{"nua sau":>9}{"boot 95% CI":>22}')
    print('-' * 78)
    for a, b, l, r in rows:
        d = r - ref
        bs = d[boot].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        star = ' *' if (a, b, l) == (ra, rb, rl) else ''
        print(f'{a:>6}{b:>6}{l:>6}{r.mean():>11.4f}{d.mean():>+9.4f}'
              f'{r[:half].mean():>9.4f}{r[half:].mean():>9.4f}'
              f'{f"[{lo:+.4f}, {hi:+.4f}]":>22}{star}')
    print('-' * 78)

    best = max(rows, key=lambda t: t[3][:half].mean())
    a, b, l, r = best
    d = r - ref
    bs = d[boot].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'\nChon tren NUA DAU : alpha={a} beta={b} lam={l} -> {r[:half].mean():.4f}')
    print(f'Kiem chung NUA SAU: {r[half:].mean():.4f}  (moc nua sau {ref[half:].mean():.4f})')
    print(f'Ca {n} cau        : {r.mean():.4f} ({d.mean():+.4f}), '
          f'CI [{lo:+.4f}, {hi:+.4f}], thang {(bs > 0).mean() * 100:.1f}%')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('dump')
    a.add_argument('-n', type=int, default=500)
    a.add_argument('--offset', type=int, default=0,
                   help='Bo qua N cau dau. Dung de lay nua SAU cua tap validation - nua dau da bi dung de chon tham so nhieu lan nen khong con trung thuc.')
    a.add_argument('--emb', default='emb_ft')
    a.add_argument('-o', '--out', required=True)
    a.set_defaults(fn=cmd_dump)

    b = sub.add_parser('score')
    b.add_argument('-d', '--dump', required=True)
    b.add_argument('--model', default=RERANKER)
    b.add_argument('--batch', type=int, default=16)
    b.add_argument('-o', '--out', required=True)
    b.set_defaults(fn=cmd_score)

    c = sub.add_parser('sweep')
    c.add_argument('-d', '--dump', required=True)
    c.add_argument('-s', '--scores', required=True)
    c.add_argument('--alphas', type=float, nargs='+', default=ALPHAS)
    c.add_argument('--betas', type=float, nargs='+', default=[0.5],
                   help='Truyen nhieu gia tri de quet luoi alpha x beta')
    c.add_argument('--lams', type=float, nargs='+', default=[0.2])
    c.add_argument('--ref', type=float, nargs=3, default=[0.7, 0.5, 0.2],
                   metavar=('A', 'B', 'L'), help='Cau hinh moc de so delta')
    c.set_defaults(fn=cmd_sweep)

    args = p.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
