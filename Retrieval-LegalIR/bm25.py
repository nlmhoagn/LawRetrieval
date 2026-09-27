#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
BM25 chunk-level tren inverted index nam tren dia.

Vi sao khong dung ma tran sparse trong RAM: 487k chunk x ~50M postings se ton
vai tram MB thuong truc. O day index duoc ghi ra dia duoi dang mang memmap, luc
query chi doc dung postings cua vai tu trong cau hoi -> RAM gan nhu bang 0.

Build 2 luot:
  luot 1: dem document frequency -> tu dien + offset cua tung tu
  luot 2: ghi postings (chunk_id, tf) vao dung o da chua san

Usage:
    python bm25.py build --chunks ../Chunking-LegalIR/chunks.jsonl --index index/
    python bm25.py query --index index/ "mức lương cơ sở là bao nhiêu"
    python bm25.py query --index index/ "..." --topk 10 --group-doc
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

K1 = 1.5
B = 0.75
MIN_DF = 2          # bo tu chi xuat hien 1 chunk (loi go, rac OCR)
MAX_DF_RATIO = 0.7  # bo tu xuat hien o >70% chunk (idf ~ 0 ma ton postings)

TOKEN_RE = re.compile(r'\w+', re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------- build

def build(chunks_path, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    t0 = time.time()

    # ---- luot 1: tu dien + df + do dai chunk + offset dong -------------
    print('Luot 1/2: quet tu dien...')
    df = {}
    doclen = []
    line_offsets = []
    chunk_doc = []      # doc_id cua tung chunk, de gop chunk->doc khong can doc lai file
    with open(chunks_path, 'rb') as f:
        offset = 0
        for line in f:
            line_offsets.append(offset)
            offset += len(line)
            rec = json.loads(line)
            chunk_doc.append(int(rec['doc_id']))
            text = rec['text']
            toks = tokenize(text)
            doclen.append(len(toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
            n = len(doclen)
            if n % 100000 == 0:
                print(f'  {n:,} chunk, tu dien tho {len(df):,}', flush=True)

    N = len(doclen)
    max_df = int(N * MAX_DF_RATIO)
    vocab = {t: i for i, t in enumerate(
        sorted(t for t, d in df.items() if MIN_DF <= d <= max_df))}
    dfs = np.zeros(len(vocab), dtype=np.int32)
    for t, i in vocab.items():
        dfs[i] = df[t]
    dropped_rare = sum(1 for d in df.values() if d < MIN_DF)
    dropped_common = sum(1 for d in df.values() if d > max_df)
    del df

    doclen = np.asarray(doclen, dtype=np.int32)
    avgdl = float(doclen.mean())
    nnz = int(dfs.sum())

    print(f'  {N:,} chunk | tu dien {len(vocab):,} '
          f'(bo {dropped_rare:,} tu hiem, {dropped_common:,} tu qua pho bien)')
    print(f'  postings: {nnz:,} (~{nnz * 6 / 1e6:.0f} MB tren dia) | avgdl {avgdl:.1f}')

    # offsets[i]..offsets[i+1] la vung postings cua tu i
    offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    np.cumsum(dfs, out=offsets[1:])

    # ---- luot 2: ghi postings ------------------------------------------
    print('Luot 2/2: ghi postings...')
    post = np.memmap(os.path.join(index_dir, 'postings.i32'), dtype=np.int32,
                     mode='w+', shape=(nnz,))
    tfs = np.memmap(os.path.join(index_dir, 'tfs.u16'), dtype=np.uint16,
                    mode='w+', shape=(nnz,))
    cursor = offsets[:-1].copy()

    with open(chunks_path, 'rb') as f:
        for idx, line in enumerate(f):
            toks = tokenize(json.loads(line)['text'])
            counts = {}
            for t in toks:
                i = vocab.get(t)
                if i is not None:
                    counts[i] = counts.get(i, 0) + 1
            for i, c in counts.items():
                p = cursor[i]
                post[p] = idx
                tfs[p] = min(c, 65535)
                cursor[i] = p + 1
            if (idx + 1) % 100000 == 0:
                print(f'  {idx + 1:,}/{N:,} chunk', flush=True)

    post.flush()
    tfs.flush()
    del post, tfs

    np.save(os.path.join(index_dir, 'offsets.npy'), offsets)
    np.save(os.path.join(index_dir, 'df.npy'), dfs)
    np.save(os.path.join(index_dir, 'doclen.npy'), doclen)
    np.save(os.path.join(index_dir, 'line_offsets.npy'),
            np.asarray(line_offsets, dtype=np.int64))
    np.save(os.path.join(index_dir, 'chunk_doc.npy'),
            np.asarray(chunk_doc, dtype=np.int32))
    with open(os.path.join(index_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for t in sorted(vocab, key=vocab.get):
            f.write(t + '\n')
    with open(os.path.join(index_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'n_chunks': N, 'vocab_size': len(vocab), 'nnz': nnz,
                   'avgdl': avgdl, 'k1': K1, 'b': B,
                   'chunks_path': os.path.abspath(chunks_path)}, f, indent=2)

    size = sum(os.path.getsize(os.path.join(index_dir, f))
               for f in os.listdir(index_dir))
    print(f'Xong sau {time.time() - t0:.0f}s | index {size / 1e6:.0f} MB tai {index_dir}')


# ---------------------------------------------------------------- query

class BM25Index:
    def __init__(self, index_dir, k1=K1, b=B):
        self.dir = index_dir
        self.k1, self.b = k1, b
        with open(os.path.join(index_dir, 'meta.json'), encoding='utf-8') as f:
            self.meta = json.load(f)
        self.offsets = np.load(os.path.join(index_dir, 'offsets.npy'))
        self.df = np.load(os.path.join(index_dir, 'df.npy'))
        self.doclen = np.load(os.path.join(index_dir, 'doclen.npy'))
        self.line_offsets = np.load(os.path.join(index_dir, 'line_offsets.npy'))
        self.chunk_doc = np.load(os.path.join(index_dir, 'chunk_doc.npy'))
        # tong so chunk cua tung van ban trong CA corpus - dung de phat van ban dai
        self.doc_nchunks = np.bincount(self.chunk_doc)
        with open(os.path.join(index_dir, 'vocab.txt'), encoding='utf-8') as f:
            self.vocab = {t.rstrip('\n'): i for i, t in enumerate(f)}
        n = self.meta['nnz']
        self.post = np.memmap(os.path.join(index_dir, 'postings.i32'),
                              dtype=np.int32, mode='r', shape=(n,))
        self.tfs = np.memmap(os.path.join(index_dir, 'tfs.u16'),
                             dtype=np.uint16, mode='r', shape=(n,))
        N = self.meta['n_chunks']
        # idf kieu BM25 co lam tron duoi, tranh gia tri am voi tu qua pho bien
        self.idf = np.log(1.0 + (N - self.df + 0.5) / (self.df + 0.5)).astype(np.float32)
        self.set_params(self.k1, self.b)

    def set_params(self, k1, b):
        """Doi k1/b khong can build lai index: postings luu tf tho, chi mau so doi."""
        self.k1, self.b = k1, b
        self.norm = (k1 * (1 - b + b * self.doclen / self.meta['avgdl'])).astype(np.float32)

    def score(self, query):
        scores = np.zeros(self.meta['n_chunks'], dtype=np.float32)
        hit_terms = []
        for t in tokenize(query):
            i = self.vocab.get(t)
            if i is None:
                continue
            hit_terms.append(t)
            s, e = self.offsets[i], self.offsets[i + 1]
            ids = np.asarray(self.post[s:e], dtype=np.int64)
            tf = np.asarray(self.tfs[s:e], dtype=np.float32)
            scores[ids] += self.idf[i] * (tf * (self.k1 + 1)) / (tf + self.norm[ids])
        return scores, hit_terms

    def read_chunk(self, idx):
        # Giu san file handle thay vi mo lai moi lan. Truoc day ham nay mo lai file
        # chunks.jsonl (592 MB) cho TUNG chunk - mot lan sinh bai nop goi 100.000 lan,
        # khien GPU phai cho dia thay vi chay het cong suat.
        f = getattr(self, '_chunks_fh', None)
        if f is None:
            f = self._chunks_fh = open(self.meta['chunks_path'], 'rb')
        f.seek(int(self.line_offsets[idx]))
        return json.loads(f.readline())

    def __del__(self):
        f = getattr(self, '_chunks_fh', None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    def search(self, query, topk=10):
        scores, hit_terms = self.score(query)
        k = min(topk, int((scores > 0).sum()))
        if k == 0:
            return [], hit_terms
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        out = []
        for idx in top:
            c = self.read_chunk(int(idx))
            c['score'] = float(scores[idx])
            out.append(c)
        return out, hit_terms

    def rank_docs(self, scores, topk=5, pool=3000, agg='max'):
        """Gop diem chunk ve document. Chi dung mang trong RAM, khong doc file.

        agg: max        - diem cao nhat trong cac chunk cua van ban
             top3       - trung binh 3 chunk cao nhat (do bot may man cua max)
             top3_log   - top3 chia log(so chunk), phat van ban qua dai
        Tra ve [(doc_id_str, diem, chunk_idx_manh_nhat, so_chunk_trong_pool)].
        """
        nz = np.flatnonzero(scores)
        if nz.size == 0:
            return []
        if nz.size > pool:
            part = np.argpartition(-scores[nz], pool - 1)[:pool]
            nz = nz[part]
        order = nz[np.argsort(-scores[nz])]      # chunk theo diem giam dan

        best_chunk, vals = {}, {}
        for idx in order:
            d = int(self.chunk_doc[idx])
            if d not in best_chunk:
                best_chunk[d] = int(idx)
                vals[d] = []
            vals[d].append(float(scores[idx]))

        ranked = []
        for d, v in vals.items():
            if agg == 'max':
                s = v[0]
            else:
                s = float(np.mean(v[:3]))
                if agg == 'top3_log':
                    # chia cho tong so chunk cua van ban trong corpus, KHONG phai
                    # so chunk lot pool - dung so pool se thuong nham van ban
                    # chi vua du lot vao pool voi 1 chunk.
                    s /= np.log1p(float(self.doc_nchunks[d]))
            ranked.append((str(d), s, best_chunk[d], len(v)))
        ranked.sort(key=lambda x: -x[1])
        return ranked[:topk]

    def search_docs(self, query, topk=5, agg='max'):
        scores, hit_terms = self.score(query)
        ranked = self.rank_docs(scores, topk=topk, agg=agg)
        out = [(s, did, self.read_chunk(ci), n) for did, s, ci, n in ranked]
        return out, hit_terms


def print_results(results, hit_terms, query):
    print('=' * 78)
    print(f'QUERY: {query}')
    print(f'Tu khop tu dien: {", ".join(hit_terms) if hit_terms else "(khong co)"}')
    print('=' * 78)
    if not results:
        print('Khong tim thay ket qua.')
        return
    for r, c in enumerate(results, 1):
        print(f'#{r}  score {c["score"]:.2f}   doc {c["doc_id"]}   [tang {c["tier"]}, {c["n_words"]} tu]')
        print(f'    path: {c["path"]}')
        body = c['text'].split('\n', 1)
        snippet = (body[1] if len(body) > 1 else c['text'])[:220].replace('\n', ' ')
        print(f'    {snippet}...')
        print()


def print_doc_results(ranked, hit_terms, query):
    print('=' * 78)
    print(f'QUERY: {query}   (gop chunk -> document)')
    print(f'Tu khop tu dien: {", ".join(hit_terms) if hit_terms else "(khong co)"}')
    print('=' * 78)
    for r, (score, did, best, nchunk) in enumerate(ranked, 1):
        print(f'#{r}  score {score:.2f}   doc {did}   ({nchunk} chunk trong pool)')
        print(f'    path chunk manh nhat: {best["path"]}')
        print()


def main():
    p = argparse.ArgumentParser(description='BM25 chunk-level cho corpus LegalIR.')
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='Dung index tu chunks.jsonl')
    b.add_argument('--chunks', '-c', required=True)
    b.add_argument('--index', '-i', default='index')

    q = sub.add_parser('query', help='Tim kiem')
    q.add_argument('--index', '-i', default='index')
    q.add_argument('query', nargs='+')
    q.add_argument('--topk', '-k', type=int, default=10)
    q.add_argument('--group-doc', '-g', action='store_true',
                   help='Gop chunk ve document thay vi liet ke tung chunk')
    q.add_argument('--agg', default='max', choices=['max', 'top3', 'top3_log'])

    args = p.parse_args()
    if args.cmd == 'build':
        build(args.chunks, args.index)
    else:
        query = ' '.join(args.query)
        idx = BM25Index(args.index)
        if args.group_doc:
            ranked, hits = idx.search_docs(query, topk=args.topk, agg=args.agg)
            print_doc_results(ranked, hits, query)
        else:
            res, hits = idx.search(query, topk=args.topk)
            print_results(res, hits, query)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
