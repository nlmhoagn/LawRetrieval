#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Chay toan bo pipeline bang mot lenh.

    python run_all.py --data "LegalIR - Public Test"

Bon buoc, moi buoc tao ra thu buoc sau can. Buoc nao da co san pham thi tu bo qua,
nen chay lai an toan - dut giua chung khong phai lam lai tu dau.

    1. chunk corpus          1m31s   ->  chunks.jsonl        592 MB
    2. index BM25             202s   ->  index/              299 MB
    3. trich vector       36 phut*   ->  emb/                714 MB
    4. sinh bai nop            74s   ->  submission.zip       21 KB
    5. kiem tra bai nop

    * can GPU. Tren CPU se lau hon rat nhieu.

Thu muc du lieu can co:
    <data>/selected-contexts/     8.532 file context_*.json
    <data>/public-official.json   cau hoi can tra loi
    <data>/train.json             tuy chon, dung lam nguon do khi BM25 tra < 5 ket qua
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNK_DIR = os.path.join(ROOT, 'Chunking-LegalIR')
RETR_DIR = os.path.join(ROOT, 'Retrieval-LegalIR')
VALID_DIR = os.path.join(ROOT, 'Validator-Task-LegalIR')


def run(cmd, cwd, label):
    print(f'\n{"=" * 70}\n{label}\n{"=" * 70}', flush=True)
    print('$ ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, '-u'] + cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f'\nBUOC THAT BAI: {label} (exit {r.returncode})')
    print(f'-> {time.time() - t0:.0f}s', flush=True)


def find_contexts(data_dir):
    """Tim thu muc chua context_*.json - zip goc giai nen ra long hai cap."""
    cand = os.path.join(data_dir, 'selected-contexts')
    if not os.path.isdir(cand):
        sys.exit(f'Khong tim thay {cand}')
    if any(f.startswith('context_') for f in os.listdir(cand)):
        return cand
    for n in os.listdir(cand):
        sub = os.path.join(cand, n)
        if os.path.isdir(sub) and any(x.startswith('context_') for x in os.listdir(sub)):
            return sub
    sys.exit(f'Khong tim thay file context_*.json trong {cand}')


def main():
    p = argparse.ArgumentParser(
        description='Chay toan bo pipeline LegalIR bang mot lenh.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', '-d', required=True,
                   help='Thu muc du lieu cua BTC')
    p.add_argument('--out', '-o', default='submission',
                   help='Ten bai nop dau ra (khong duoi)')
    p.add_argument('--questions', '-q', default='public-official.json',
                   help='File de bai trong thu muc du lieu')
    p.add_argument('--k1', type=float, default=2.5)
    p.add_argument('--b', type=float, default=0.9)
    p.add_argument('--alpha', type=float, default=0.7,
                   help='Trong so DENSE khi hoa diem (con lai la BM25)')
    p.add_argument('--pool', type=int, default=1000)
    p.add_argument('--force', action='store_true',
                   help='Lam lai tu dau, ke ca buoc da co san pham')
    p.add_argument('--skip-encode', action='store_true',
                   help='Bo qua dense, chi dung BM25 (nhanh nhung kem ~0.05 Recall)')
    args = p.parse_args()

    data = os.path.abspath(args.data)
    ctx = find_contexts(data)
    questions = os.path.join(data, args.questions)
    train = os.path.join(data, 'train.json')
    if not os.path.exists(questions):
        sys.exit(f'Khong tim thay de bai: {questions}')

    chunks = os.path.join(CHUNK_DIR, 'chunks.jsonl')
    index = os.path.join(RETR_DIR, 'index')
    emb = os.path.join(RETR_DIR, 'emb')

    print(f'Du lieu   : {data}')
    print(f'Corpus    : {ctx}')
    print(f'De bai    : {questions}')
    print(f'Cau hinh  : k1={args.k1} b={args.b} alpha={args.alpha} pool={args.pool}')

    # --- 1. chunk ---------------------------------------------------------
    if args.force or not os.path.exists(chunks):
        run(['chunker.py', '--contexts', ctx, '--out', 'chunks.jsonl'],
            CHUNK_DIR, 'BUOC 1/5 - Chunk corpus (~1m30s)')
    else:
        print(f'\n[bo qua buoc 1] da co {chunks}')

    # --- 2. index BM25 ----------------------------------------------------
    if args.force or not os.path.exists(os.path.join(index, 'meta.json')):
        run(['bm25.py', 'build', '--chunks', chunks, '--index', 'index'],
            RETR_DIR, 'BUOC 2/5 - Index BM25 (~3 phut)')
    else:
        print(f'[bo qua buoc 2] da co {index}')

    # --- 3. trich vector --------------------------------------------------
    use_dense = not args.skip_encode
    if use_dense:
        if args.force or not os.path.exists(os.path.join(emb, 'meta.json')):
            run(['encode.py', '--chunks', chunks, '--out', 'emb'],
                RETR_DIR, 'BUOC 3/5 - Trich vector halong_embedding (~36 phut, can GPU)')
        else:
            print(f'[bo qua buoc 3] da co {emb}')

    # --- 4. sinh bai nop --------------------------------------------------
    cmd = ['predict.py', '--index', 'index', '--questions', questions,
           '--k1', str(args.k1), '--b', str(args.b), '--out', args.out]
    if use_dense:
        cmd += ['--emb', 'emb', '--alpha', str(args.alpha), '--pool', str(args.pool)]
    if os.path.exists(train):
        cmd += ['--train', train]
    run(cmd, RETR_DIR, 'BUOC 4/5 - Sinh bai nop (~1-2 phut)')

    # --- 5. kiem tra ------------------------------------------------------
    zip_path = os.path.join(RETR_DIR, f'{args.out}.zip')
    run([os.path.join(VALID_DIR, 'validate_submission.py'), zip_path,
         '--ref', questions, '--corpus', ctx],
        RETR_DIR, 'BUOC 5/5 - Kiem tra bai nop')

    print(f'\n{"=" * 70}')
    print(f'XONG. Bai nop: {zip_path}')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
