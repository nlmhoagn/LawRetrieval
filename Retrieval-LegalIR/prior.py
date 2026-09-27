#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Prior do pho bien: cong them diem cho van ban hay la dap an trong train.

Vi sao co ly: BM25 va dense deu chi nhin quan he cau hoi <-> noi dung, khong he
biet van ban nao thuc te hay duoc hoi. Nhan train cho khong tin hieu do. Do tren
1000 cau held-out: van ban tung la dap an >=20 lan co xac suat lam dap an cao
gap 50 lan van ban chua tung, con van ban chua tung chi bang 0.40 lan.

    diem' = diem + lam * log1p(freq) / log1p(max_freq)

diem da min-max trong pool ung vien nen nam trong [0,1]; lam vi the la "toi da
duoc cong bao nhieu phan cua toan thang diem".

Chay 3 buoc (dump mot lan, quet lam offline gan nhu mien phi):

    python prior.py dump  -q val    -o cands_val.json
    python prior.py sweep -c cands_val.json
    python prior.py dump  -q "../LegalIR - Public Test-.../public-official.json" -o cands_pub.json
    python prior.py submit -c cands_pub.json --lam <chon o buoc sweep> -o submission_prior
"""

import argparse
import collections
import json
import math
import os
import sys
import time
import zipfile

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'LegalIR - Public Test-20260806T081424Z-1-001', 'LegalIR - Public Test')

def _find_train():
    candidates = [
        os.path.join(BASE, 'train.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LegalIR - Public Test', 'train.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train.json'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return os.path.join(BASE, 'train.json')

TRAIN = _find_train()
PUBLIC = os.path.join(BASE, 'public-official.json')
SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                     'Finetune-LegalIR', 'data', 'split.json')


def _pick_reranker():
    """Uu tien ban fp16 (nap nhanh hon, doc 1.14 GB thay vi 2.27 GB), khong co
    thi dung ban goc do train_ce.py sinh ra.

    Ket qua hai ban giong nhau TUNG BIT - da do tren 2.911 diem, sai lech 0.0.
    Vi rerank.py cast sang fp16 luc nap len GPU du checkpoint tren dia la gi,
    ma ban fp16 chinh la ket qua cua dung phep cast do.
    """
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'Finetune-LegalIR', 'models')
    fp16 = os.path.join(base, 'reranker-ft-fp16')
    return fp16 if os.path.isdir(fp16) else os.path.join(base, 'reranker-ft')


RERANKER = _pick_reranker()

# Cau hinh tot nhat hien tai (RUNNING.md muc 8, commit af45536: public 0.9456)
CFG = dict(k1=2.5, b=0.9, pool=2000, alpha=0.7, agg='max+0.4',
           ndocs=50, mchunks=2, beta=0.5)
NCAND = 50          # so ung vien luu lai de con cho prior xep lai
TOPK = 5


def load_json(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


# ------------------------------------------------------------------ prior

def build_prior(train, exclude_qids=frozenset()):
    """Trong so [0,1] theo do pho bien. exclude_qids: cau hoi KHONG duoc dem,
    dung khi do tren validation - neu dem ca cau dang cham thi dap an cua no
    tu cong cho chinh minh, delta se ao."""
    freq = collections.Counter(
        str(d) for qid, v in train.items() if qid not in exclude_qids
        for d in v['answer'])
    mx = math.log1p(max(freq.values()))
    return {d: math.log1p(c) / mx for d, c in freq.items()}, freq


def apply_prior(cand, w, lam, topk=TOPK):
    """cand: [[doc_id, score], ...] da sap giam dan. Tra ve top-k doc_id."""
    if not cand:
        return []
    v = np.array([s for _, s in cand], dtype=np.float64)
    lo, hi = v.min(), v.max()
    v = (v - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(v)
    if lam:
        v = v + lam * np.array([w.get(d, 0.0) for d, _ in cand])
    return [cand[j][0] for j in np.argsort(-v)[:topk]]


# ------------------------------------------------------------------ dump

def cmd_dump(args):
    from hybrid import Hybrid
    from rerank import Reranker, EnsembleReranker, rerank_docs

    if args.questions == 'val':
        train = load_json(TRAIN)
        val = set(map(str, load_json(SPLIT)['val']))
        questions = {q: train[q] for q in train if q in val}
        print(f'Nguon  : validation {len(questions)} cau (tu split.json)')
    else:
        questions = load_json(args.questions)
        print(f'Nguon  : {args.questions} | {len(questions)} cau')

    # Nap cross-encoder TRUOC: checkpoint 2.17 GB can map tron mot lan, phai gianh
    # duoc lo commit lien tuc luc con rong nhat. Nap bi-encoder truoc thi den luot
    # no chi con manh vun -> "paging file too small".
    reranker_paths = [p.strip() for p in args.reranker.split(',') if p.strip()]
    if len(reranker_paths) > 1:
        rr = EnsembleReranker(reranker_paths, batch=args.batch)
    else:
        rr = Reranker(reranker_paths[0], batch=args.batch)
    CFG['alpha'], CFG['beta'], CFG['agg'] = args.alpha, args.beta, args.agg
    if hasattr(args, 'pool') and args.pool:
        CFG['pool'] = args.pool
    if hasattr(args, 'ndocs') and args.ndocs:
        CFG['ndocs'] = args.ndocs
    if hasattr(args, 'mchunks') and args.mchunks:
        CFG['mchunks'] = args.mchunks
    h = Hybrid(args.index, args.emb, k1=CFG['k1'], b=CFG['b'])
    print(f'BM25   : k1={CFG["k1"]} b={CFG["b"]} | pool {CFG["pool"]}')
    print(f'Dense  : {args.emb} alpha={CFG["alpha"]}')
    print(f'CE     : {rr.model_id} beta={CFG["beta"]} '
          f'| {CFG["ndocs"]} van ban x {CFG["mchunks"]} chunk')

    qids = list(questions)
    if args.limit:
        qids = qids[:args.limit]
        print(f'  (--limit {args.limit}: chi chay {len(qids)} cau de thu)')
    texts = [questions[q]['question'] for q in qids]
    t = time.time()
    qvecs = np.concatenate([h.encode_query(texts[i:i + 64])
                            for i in range(0, len(texts), 64)])
    print(f'  encode {len(texts)} cau hoi: {time.time() - t:.0f}s')

    out = {}
    t0 = time.time()
    for i, qid in enumerate(qids):
        ranked = rerank_docs(h, rr, texts[i], topk=NCAND, pool=CFG['pool'],
                             alpha=CFG['alpha'], agg=CFG['agg'],
                             n_docs=CFG['ndocs'], m_chunks=CFG['mchunks'],
                             qvec=qvecs[i], beta=CFG['beta'])
        out[qid] = [[d, float(s)] for d, s, _, _ in ranked]
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f'  {i + 1}/{len(qids)} ({el:.0f}s, con ~{el / (i + 1) * (len(qids) - i - 1) / 60:.0f} phut)',
                  flush=True)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'cfg': CFG, 'ncand': NCAND, 'cands': out}, f)
    nc = np.mean([len(v) for v in out.values()])
    print(f'\nXong {len(out)} cau trong {time.time() - t0:.0f}s | '
          f'trung binh {nc:.1f} ung vien/cau -> {args.out}')
    return 0


# ------------------------------------------------------------------ sweep

def cmd_sweep(args):
    d = load_json(args.cands)
    cands = d['cands']
    train = load_json(TRAIN)
    val_ids = set(map(str, load_json(SPLIT)['val']))

    # prior dem tu 6000 cau train, KHONG dem 1000 cau dang cham
    w, freq = build_prior(train, exclude_qids=val_ids)
    print(f'Prior  : dem tu {len(train) - len(val_ids)} cau train '
          f'(loai {len(val_ids)} cau validation) | {len(freq)} van ban')

    gold = {q: {str(x) for x in train[q]['answer']} for q in cands}
    n = len(cands)
    ceil = np.mean([len(gold[q] & {c[0] for c in cands[q]}) / len(gold[q]) for q in cands])
    print(f'Do tren: {n} cau | tran top-{d["ncand"]} = {ceil:.4f}\n')

    def recall(lam):
        return np.array([
            len(gold[q] & set(apply_prior(cands[q], w, lam))) / len(gold[q])
            for q in cands])

    base = recall(0.0)
    half = n // 2
    keys = list(cands)
    idx1 = np.array([i for i in range(n) if i < half])
    idx2 = np.array([i for i in range(n) if i >= half])

    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(4000, n))

    print(f'goc (lam=0): Recall@5 = {base.mean():.4f}   '
          f'[nua dau {base[idx1].mean():.4f} | nua sau {base[idx2].mean():.4f}]')
    print('-' * 74)
    print(f'{"lam":>6}{"Recall@5":>11}{"delta":>10}{"nua dau":>10}{"nua sau":>10}'
          f'{"boot 95% CI":>24}')
    print('-' * 74)
    rows = []
    for lam in args.lams:
        r = recall(lam)
        dl = r - base
        bs = dl[boot].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append((lam, r.mean(), dl.mean(), dl[idx1].mean(), dl[idx2].mean(), lo, hi))
        print(f'{lam:>6}{r.mean():>11.4f}{dl.mean():>+10.4f}{dl[idx1].mean():>+10.4f}'
              f'{dl[idx2].mean():>+10.4f}{f"[{lo:+.4f}, {hi:+.4f}]":>24}')
    print('-' * 74)

    # chon tren nua dau, kiem chung nua sau - dung quy trinh muc 5 RESULTS.md
    best = max(rows, key=lambda x: x[3])
    print(f'\nChon tren NUA DAU : lam={best[0]}  (delta nua dau {best[3]:+.4f})')
    print(f'Kiem chung NUA SAU: delta {best[4]:+.4f}   '
          f'{"XAC NHAN" if best[4] > 0 else "KHONG XAC NHAN - dung dung"}')
    print(f'Ca {n} cau        : {best[1]:.4f} ({best[2]:+.4f}), '
          f'CI [{best[5]:+.4f}, {best[6]:+.4f}]')
    return 0


# ------------------------------------------------------------------ submit

def cmd_submit(args):
    d = load_json(args.cands)
    cands = d['cands']
    train_path = getattr(args, 'train', None) or TRAIN
    train = load_json(train_path)

    # Bai nop that: dem tren CA 7000 cau train. Public test roi nhau voi train
    # nen khong co ro ri, va nhieu du lieu hon thi tan suat it nhieu hon.
    w, freq = build_prior(train)
    print(f'Prior  : dem tu ca {len(train)} cau train | {len(freq)} van ban '
          f'| lam={args.lam}')

    fallback = [dd for dd, _ in freq.most_common(20)]
    sub, n_pad, n_moved = {}, 0, 0
    for qid, cand in cands.items():
        before = [c[0] for c in cand[:TOPK]]
        preds = apply_prior(cand, w, args.lam)
        if set(preds) != set(before):
            n_moved += 1
        if len(preds) < TOPK:
            n_pad += 1
            for dd in fallback:
                if len(preds) >= TOPK:
                    break
                if dd not in preds:
                    preds.append(dd)
        sub[qid] = {'answer': [str(x) for x in preds[:TOPK]]}

    json_path = f'{args.out}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sub, f, ensure_ascii=False)
    zip_path = f'{args.out}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, 'submission.json')

    lens = collections.Counter(len(v['answer']) for v in sub.values())
    print(f'\n{len(sub)} cau | ' + ', '.join(f'{k} ID: {v} cau' for k, v in sorted(lens.items())))
    print(f'  Van ban rieng   : {len({x for v in sub.values() for x in v["answer"]})}')
    print(f'  Cau bi doi top-5: {n_moved} ({n_moved / len(sub) * 100:.1f}%)')
    print(f'  Cau phai do them: {n_pad}')
    print(f'  -> {zip_path} ({os.path.getsize(zip_path) / 1024:.0f} KB)')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    d = sub.add_parser('dump', help='Chay pipeline, luu top-50 ung vien kem diem')
    d.add_argument('--questions', '-q', required=True, help='"val" hoac duong dan file de bai')
    d.add_argument('--out', '-o', required=True)
    d.add_argument('--index', default='index')
    d.add_argument('--emb', default='emb_ft')
    d.add_argument('--reranker', default=RERANKER,
                   help='Duong dan den cross-encoder. Truyen nhieu duong dan ngan '
                        'bang dau phay de dung ensemble: path1,path2')
    d.add_argument('--batch', type=int, default=16,
                   help='Batch size cho cross-encoder. Tang len tren A100.')
    d.add_argument('--limit', type=int, default=0, help='Chi chay N cau dau - de thu nhanh')
    d.add_argument('--alpha', type=float, default=CFG['alpha'])
    d.add_argument('--agg', default=CFG['agg'],
                   help="Gop chunk->van ban: max | topN | max+W. 'max+0.4' do duoc "
                        "tot nhat sau khi reranker da fine-tune.")
    d.add_argument('--beta', type=float, default=CFG['beta'],
                   help='Trong so cross-encoder. Quet lai moi khi doi model dense: '
                        'dense manh hon thi beta toi uu giam xuong.')
    d.add_argument('--pool', type=int, default=0,
                   help='BM25 pool size. 0 = dung default tu CFG.')
    d.add_argument('--ndocs', type=int, default=0,
                   help='So van ban dua vao cross-encoder. 0 = dung default.')
    d.add_argument('--mchunks', type=int, default=0,
                   help='So chunk moi van ban dua vao CE. 0 = dung default.')
    d.set_defaults(fn=cmd_dump)

    s = sub.add_parser('sweep', help='Quet lam tren validation')
    s.add_argument('--cands', '-c', required=True)
    s.add_argument('--lams', type=float, nargs='+',
                   default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8])
    s.set_defaults(fn=cmd_sweep)

    b = sub.add_parser('submit', help='Ap prior len dump public -> submission.zip')
    b.add_argument('--cands', '-c', required=True)
    b.add_argument('--lam', type=float, required=True)
    b.add_argument('--out', '-o', default='submission_prior')
    b.add_argument('--train', default=None, help='Duong dan train.json de dem tan suat')
    b.set_defaults(fn=cmd_submit)

    args = p.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
