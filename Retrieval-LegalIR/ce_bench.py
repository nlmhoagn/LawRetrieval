#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
So sanh cac cross-encoder tren CUNG mot bo ung vien.

Tach lam 3 buoc de doi model khong phai chay lai tang hybrid:

    pairs  - chay BM25 + dense, dump ung vien (50 van ban x 2 chunk) cho N cau val
    score  - nap mot reranker, cham diem toan bo cap (cau hoi, chunk), luu ra file
    eval   - tron voi diem hybrid theo beta, gop ve document, do Recall@5

Vi cac model cham tren cung bo ung vien va cung tap cau hoi nen so sanh duoc theo
CAP: bootstrap tren hieu so tung cau, nhieu thap hon nhieu so voi so sanh doc lap.

    python ce_bench.py pairs -n 500 -o pairs_val500.json
    python ce_bench.py score -p pairs_val500.json --model ../Finetune-LegalIR/models/reranker-ft --kind seq  -o sc_ft.json
    python ce_bench.py score -p pairs_val500.json --model Qwen/Qwen3-Reranker-0.6B --kind qwen3 -o sc_qwen06.json
    python ce_bench.py eval  -p pairs_val500.json -s sc_ft.json sc_qwen06.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'LegalIR - Public Test-20260806T081424Z-1-001', 'LegalIR - Public Test')
TRAIN = os.path.join(BASE, 'train.json')
SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                     'Finetune-LegalIR', 'data', 'split.json')

CFG = dict(k1=2.5, b=0.9, pool=2000, alpha=0.7, ndocs=50, mchunks=2)
MAX_LEN = 512


def load_json(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


# ------------------------------------------------------------------ pairs

def cmd_pairs(args):
    """Chay tang hybrid, dump ung vien. Khong nap cross-encoder nao."""
    import numpy as np
    from hybrid import Hybrid

    train = load_json(TRAIN)
    val = [str(q) for q in load_json(SPLIT)['val'] if str(q) in train]
    val = val[args.offset:args.offset + args.n] if args.n else val[args.offset:]
    print(f'{len(val)} cau validation')

    h = Hybrid('index', args.emb, k1=CFG['k1'], b=CFG['b'])
    texts = [train[q]['question'] for q in val]
    qvecs = np.concatenate([h.encode_query(texts[i:i + 64])
                            for i in range(0, len(texts), 64)])

    out = {}
    t0 = time.time()
    for i, qid in enumerate(val):
        order, fused = h.fused_chunk_scores(texts[i], CFG['pool'], 'linear',
                                            CFG['alpha'], qvecs[i])
        if order.size == 0:
            out[qid] = {'question': texts[i], 'gold': [str(d) for d in train[qid]['answer']],
                        'cand': []}
            continue
        per_doc, fmap = {}, {}
        for j in np.argsort(-fused):
            ci = int(order[j])
            d = int(h.bm25.chunk_doc[ci])
            per_doc.setdefault(d, []).append(ci)
            fmap[ci] = float(fused[j])
            if len(per_doc) > CFG['ndocs'] and len(per_doc[d]) == 1:
                per_doc.pop(d)
                break
        cand = [[str(d), ci, fmap[ci]]
                for d, cis in per_doc.items() for ci in cis[:CFG['mchunks']]]
        out[qid] = {'question': texts[i],
                    'gold': [str(d) for d in train[qid]['answer']], 'cand': cand}
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f'  {i + 1}/{len(val)} ({el:.0f}s, con ~{el / (i + 1) * (len(val) - i - 1) / 60:.0f}p)',
                  flush=True)

    # tran: gold co nam trong tap ung vien khong
    ceil = np.mean([len(set(v['gold']) & {c[0] for c in v['cand']}) / len(v['gold'])
                    for v in out.values()])
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'cfg': CFG, 'items': out}, f)
    npair = sum(len(v['cand']) for v in out.values())
    print(f'\n{len(out)} cau, {npair} cap ({npair / len(out):.0f}/cau), '
          f'tran ung vien {ceil:.4f} -> {args.out}')
    return 0


# ------------------------------------------------------------------ score

def _texts_for(pairs, idx_dir='index'):
    """Doc noi dung chunk cho moi cap. Chi can line_offsets, khong can model."""
    from bm25 import BM25Index
    bm = BM25Index(idx_dir)
    cache = {}
    for v in pairs['items'].values():
        for _, ci, _ in v['cand']:
            if ci not in cache:
                cache[ci] = bm.read_chunk(int(ci))['text']
    return cache


def _segmenter():
    """Bo tach tu tieng Viet. PhoBERT bat buoc dau vao da tach ('muc_luong co_so')
    vi BPE cua no xay tren van ban da tach - khong tach thi tu bi vo vun."""
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize


class SeqScorer:
    """Cross-encoder kieu phan loai cap (XLM-R, BGE-m3, ViRanker, PhoRanker...)."""
    kind = 'seq'

    def __init__(self, model_id, batch, maxlen=MAX_LEN, segment=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.batch = torch, batch
        self.maxlen = maxlen
        self.seg = _segmenter() if segment else None
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            from transformers import XLMRobertaTokenizerFast
            self.tok = XLMRobertaTokenizerFast.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float16 if self.dev == 'cuda' else torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True).to(self.dev).eval()

    def score(self, query, texts):
        out = np.empty(len(texts), dtype=np.float32)
        if self.seg is not None:
            query = self.seg(query)
        for s in range(0, len(texts), self.batch):
            part = texts[s:s + self.batch]
            if self.seg is not None:
                part = [self.seg(t) for t in part]
            enc = self.tok([query] * len(part), part, padding=True, truncation=True,
                           max_length=self.maxlen, return_tensors='pt').to(self.dev)
            with self.torch.no_grad():
                out[s:s + len(part)] = self.model(**enc).logits[:, 0].float().cpu().numpy()
        return out


class YesNoScorer:
    """Reranker kieu sinh: diem = logit cua token 'yes' o vi tri cuoi.

    Dung cho bge-reranker-v2-gemma va Qwen3-Reranker. Hai ho khac nhau o khuon
    prompt nen tach qua `kind`.
    """

    def __init__(self, model_id, batch, kind):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.batch, self.kind = torch, batch, kind
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tok.padding_side = 'left'
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16 if self.dev == 'cuda' else torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True).to(self.dev).eval()
        self.yes = self.tok.convert_tokens_to_ids('yes')
        self.no = self.tok.convert_tokens_to_ids('no')
        self.yes_cap = self.tok('Yes', add_special_tokens=False)['input_ids'][0]
        if self.yes is None or self.no is None:      # tokenizer khong co token roi
            self.yes = self.tok('yes', add_special_tokens=False)['input_ids'][-1]
            self.no = self.tok('no', add_special_tokens=False)['input_ids'][-1]

    # Khuon prompt chinh thuc cua Qwen3-Reranker. Sai khuon nay (vd bo khoi <think>)
    # la diem lech han - model duoc huan luyen de tra loi dung o vi tri do.
    Q3_PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements '
              'based on the Query and the Instruct provided. Note that the answer can '
              'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    Q3_SUF = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    Q3_INSTR = ('Given a legal question in Vietnamese, retrieve the legal document '
                'passage that answers the question')

    # bge-reranker-v2-gemma: cau chi dan nam o CUOI, sau cap A/B - model duoc huan
    # luyen de doan token ngay sau cau do. Dat len dau thi diem gan nhu vo nghia
    # (do thu: AUC 0.67 so voi 0.95 cua model dang dung).
    GEMMA_INSTR = ("Given a query A and a passage B, determine whether the passage "
                   "contains an answer to the query by providing a prediction of "
                   "either 'Yes' or 'No'.")

    def _prompt(self, query, doc):
        if self.kind == 'gemma':
            return f'A: {query}\nB: {doc}\n{self.GEMMA_INSTR}'
        return (self.Q3_PRE + f'<Instruct>: {self.Q3_INSTR}\n<Query>: {query}\n'
                f'<Document>: {doc}' + self.Q3_SUF)

    def _fit_doc(self, doc, budget):
        """Cat bot noi dung chunk TRUOC khi ghep prompt.

        Neu de truncation cua tokenizer lo viec nay thi no cat tu duoi len, an mat
        khoi <think> o cuoi -> model khong biet phai tra loi o dau.
        """
        ids = self.tok(doc, add_special_tokens=False)['input_ids']
        if len(ids) <= budget:
            return doc
        return self.tok.decode(ids[:budget], skip_special_tokens=True)

    def score(self, query, texts):
        out = np.empty(len(texts), dtype=np.float32)
        # gemma can BOS o dau chuoi; Qwen3 da co san dau <|im_start|> trong khuon
        special = (self.kind == 'gemma')
        overhead = len(self.tok(self._prompt(query, ''),
                                add_special_tokens=special)['input_ids'])
        budget = max(64, MAX_LEN - overhead)
        for s in range(0, len(texts), self.batch):
            part = [self._fit_doc(t, budget) for t in texts[s:s + self.batch]]
            prompts = [self._prompt(query, t) for t in part]
            enc = self.tok(prompts, padding=True, return_tensors='pt',
                           add_special_tokens=special).to(self.dev)
            with self.torch.no_grad():
                lg = self.model(**enc).logits[:, -1, :].float()
            if self.kind == 'gemma':
                v = lg[:, self.yes_cap]      # dung token 'Yes' nhu ban goc cua BAAI
            else:
                # log P(yes) chuan hoa tren {yes, no} - dung cach cham chinh thuc
                two = self.torch.stack([lg[:, self.no], lg[:, self.yes]], dim=1)
                v = self.torch.log_softmax(two, dim=1)[:, 1]
            out[s:s + len(part)] = v.cpu().numpy()
        return out


def cmd_score(args):
    pairs = load_json(args.pairs)
    print(f'{len(pairs["items"])} cau, '
          f'{sum(len(v["cand"]) for v in pairs["items"].values())} cap')
    print('doc noi dung chunk...', flush=True)
    texts = _texts_for(pairs)
    print(f'  {len(texts)} chunk rieng')

    print(f'nap {args.model} (kind={args.kind}, batch={args.batch})...', flush=True)
    t = time.time()
    sc = (SeqScorer(args.model, args.batch, args.maxlen, args.segment)
          if args.kind == 'seq'
          else YesNoScorer(args.model, args.batch, args.kind))
    print(f'  {time.time() - t:.0f}s')

    out, t0 = {}, time.time()
    items = list(pairs['items'].items())
    for i, (qid, v) in enumerate(items):
        if not v['cand']:
            out[qid] = []
            continue
        out[qid] = [float(x) for x in
                    sc.score(v['question'], [texts[c[1]] for c in v['cand']])]
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  {i + 1}/{len(items)} ({el:.0f}s, con ~{el / (i + 1) * (len(items) - i - 1) / 60:.0f}p)',
                  flush=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'kind': args.kind, 'scores': out}, f)
    print(f'\nxong {time.time() - t0:.0f}s -> {args.out}')
    return 0


# ------------------------------------------------------------------ eval

def _minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def _recall(pairs, scores, beta, agg_n=2, topk=5):
    rec = []
    for qid, v in pairs['items'].items():
        gold = set(v['gold'])
        if not v['cand']:
            rec.append(0.0)
            continue
        ce = np.array(scores[qid], dtype=np.float64)
        fh = np.array([c[2] for c in v['cand']], dtype=np.float64)
        s = beta * _minmax(ce) + (1 - beta) * _minmax(fh) if beta < 1.0 else _minmax(ce)
        vals = {}
        for j in np.argsort(-s):
            vals.setdefault(v['cand'][j][0], []).append(float(s[j]))
        docs = sorted(((d, float(np.mean(x[:agg_n]))) for d, x in vals.items()),
                      key=lambda z: -z[1])[:topk]
        rec.append(len(gold & {d for d, _ in docs}) / len(gold))
    return np.array(rec)


def cmd_eval(args):
    pairs = load_json(args.pairs)
    n = len(pairs['items'])
    ceil = np.mean([len(set(v['gold']) & {c[0] for c in v['cand']}) / len(v['gold'])
                    for v in pairs['items'].values()])
    base = _recall(pairs, {q: [c[2] for c in v['cand']]
                           for q, v in pairs['items'].items()}, beta=0.0)
    print(f'{n} cau | tran ung vien {ceil:.4f} | khong rerank {base.mean():.4f}\n')

    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(4000, n))
    half = n // 2

    results = {}
    for path in args.scores:
        d = load_json(path)
        name = os.path.basename(path).replace('.json', '')
        print(f'--- {name}  ({d["model"]}, {d["kind"]}) ---')
        print(f'{"beta":>6}{"Recall@5":>11}{"vs khong rerank":>18}{"nua dau":>10}{"nua sau":>10}')
        best = None
        for beta in args.betas:
            r = _recall(pairs, d['scores'], beta)
            print(f'{beta:>6}{r.mean():>11.4f}{r.mean() - base.mean():>+18.4f}'
                  f'{r[:half].mean():>10.4f}{r[half:].mean():>10.4f}')
            if best is None or r[:half].mean() > best[1][:half].mean():
                best = (beta, r)
        results[name] = best
        print(f'  -> chon beta={best[0]} tren nua dau, nua sau {best[1][half:].mean():.4f}\n')

    if len(results) > 1:
        names = list(results)
        ref = names[0]
        print('=' * 62)
        print(f'SO SANH THEO CAP (moc: {ref})')
        print('=' * 62)
        for nm in names[1:]:
            d = results[nm][1] - results[ref][1]
            bs = d[boot].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            win = (bs > 0).mean() * 100
            print(f'{nm:<22}{d.mean():>+9.4f}  CI [{lo:+.4f}, {hi:+.4f}]  thang {win:.1f}%')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('pairs')
    a.add_argument('-n', type=int, default=500)
    a.add_argument('--offset', type=int, default=0,
                   help='Bo qua N cau dau. --offset 500 lay nua SAU cua tap validation, phan chua tung dung de chon tham so.')
    a.add_argument('--emb', default='emb_ft')
    a.add_argument('-o', '--out', required=True)
    a.set_defaults(fn=cmd_pairs)

    b = sub.add_parser('score')
    b.add_argument('-p', '--pairs', required=True)
    b.add_argument('--model', required=True)
    b.add_argument('--kind', default='seq', choices=['seq', 'qwen3', 'gemma'])
    b.add_argument('--batch', type=int, default=16)
    b.add_argument('--maxlen', type=int, default=MAX_LEN,
                   help='PhoBERT chi co 258 vi tri -> phai dat 256, khong thi CUDA assert')
    b.add_argument('--segment', action='store_true',
                   help='Tach tu tieng Viet truoc khi tokenize (bat buoc voi PhoBERT)')
    b.add_argument('-o', '--out', required=True)
    b.set_defaults(fn=cmd_score)

    c = sub.add_parser('eval')
    c.add_argument('-p', '--pairs', required=True)
    c.add_argument('-s', '--scores', nargs='+', required=True)
    c.add_argument('--betas', type=float, nargs='+',
                   default=[0.3, 0.5, 0.7, 0.9, 1.0])
    c.set_defaults(fn=cmd_eval)

    args = p.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
