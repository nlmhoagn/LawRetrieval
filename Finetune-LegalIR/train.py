#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Fine-tune hiieu/halong_embedding tren du lieu cuoc thi.

LOSS
  CachedMultipleNegativesRankingLoss boc trong MatryoshkaLoss.

  MNRL keo positive lai gan, day negative ra xa. No dung ca negative TRONG BATCH -
  moi mau khac trong batch deu la negative mien phi, nen batch cang lon cang tot.

  Ban "Cached" (GradCache) chia thanh hai luot nen batch size khong con rang buoc
  voi VRAM: batch 64 tren 6GB thay vi batch 8. Cham hon ~2x moi buoc nhung nhieu
  negative hon han.

  Matryoshka giu lai vi model goc duoc train bang no - bo di thi mat kha nang cat
  768->256, va no cung co tac dung chinh quy hoa.

RANG BUOC PHAN CUNG
  6.4 GB VRAM. Model 278M + AdamW (fp32 master + 2 trang thai momentum) da ton
  ~3.3 GB truoc khi tinh gradient va activation.

Usage:
    python train.py --smoke                      # 30 buoc, do VRAM va toc do
    python train.py                              # train that, 1 epoch
    python train.py --batch 32 --mini 8          # neu OOM thi ha xuong
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import (CachedMultipleNegativesRankingLoss,
                                          MatryoshkaLoss,
                                          MultipleNegativesRankingLoss)
from transformers import TrainerCallback

BASE_MODEL = 'hiieu/halong_embedding'
MATRYOSHKA_DIMS = [768, 512, 256, 128]
LORA_TARGETS = ['query', 'key', 'value']


def merge_and_strip(module):
    """Gop delta LoRA vao trong so goc roi go lop boc, tra ve nn.Linear thuan.

    Can lam vay de model luu ra nap duoc bang AutoModel binh thuong - encode.py
    khong phai biet gi ve LoRA.
    """
    from peft.tuners.lora import LoraLayer
    for name, child in list(module.named_children()):
        if isinstance(child, LoraLayer):
            child.merge()
            setattr(module, name, child.base_layer)
        else:
            merge_and_strip(child)

log = logging.getLogger('ft')


def setup_logging(logfile):
    os.makedirs(os.path.dirname(logfile) or '.', exist_ok=True)
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, encoding='utf-8')):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False


class ProgressCallback(TrainerCallback):
    """In tien do moi N buoc: loss, lr, VRAM, toc do, thoi gian con lai."""

    def __init__(self, total_steps):
        self.total = total_steps
        self.t0 = None
        self.last_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        log.info(f'Bat dau huan luyen: {self.total} buoc')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or 'loss' not in logs:
            return
        step = state.global_step
        el = time.time() - self.t0
        sps = step / el if el > 0 else 0
        eta = (self.total - step) / sps / 60 if sps > 0 else 0
        vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        pct = 100 * step / max(self.total, 1)
        self.last_loss = logs['loss']
        log.info(
            f'buoc {step:>5}/{self.total} ({pct:5.1f}%)  '
            f'loss {logs["loss"]:.4f}  '
            f'lr {logs.get("learning_rate", 0):.2e}  '
            f'VRAM {vram:.2f}GB  '
            f'{sps:.2f} buoc/s  con ~{eta:.0f} phut'
        )

    def on_train_end(self, args, state, control, **kwargs):
        el = time.time() - self.t0
        log.info(f'Ket thuc sau {el / 60:.1f} phut, loss cuoi {self.last_loss}')


def load_pairs(path, n_neg, limit=0):
    rows, skipped = [], 0
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and len(rows) >= limit:
                break
            r = json.loads(line)
            negs = r['negatives'][:n_neg]
            if len(negs) < n_neg:       # bo cau thieu negative de cot dong deu nhau
                skipped += 1
                continue
            item = {'anchor': r['query'], 'positive': r['positive']}
            for j, ng in enumerate(negs):
                item[f'negative_{j + 1}'] = ng
            rows.append(item)
    return Dataset.from_list(rows), skipped


def main():
    p = argparse.ArgumentParser(
        description='Fine-tune bi-encoder cho LegalIR.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', '-d', default='data/pairs.jsonl')
    p.add_argument('--base', default=BASE_MODEL, help='Model nen de fine-tune')
    p.add_argument('--out', '-o', default='models/halong-ft')
    p.add_argument('--log', default='logs/train.log')
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--batch', type=int, default=64,
                   help='Batch that (nho GradCache nen khong bi VRAM chan)')
    p.add_argument('--mini', type=int, default=8,
                   help='Mini-batch cua GradCache - day moi la thu an VRAM')
    p.add_argument('--neg', type=int, default=3, help='So hard negative moi cau hoi')
    p.add_argument('--maxlen', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--no-cache', action='store_true',
                   help='Dung MNRL thuong thay vi GradCache (can batch nho)')
    p.add_argument('--lora', action='store_true',
                   help='Fine-tune bang LoRA thay vi toan bo trong so. Bat buoc voi '
                        'model 568M tro len: full fine-tune can ~7.9 GB chi rieng '
                        'trong so + trang thai AdamW, vuot 6.4 GB VRAM.')
    p.add_argument('--rank', type=int, default=16, help='Hang cua LoRA')
    p.add_argument('--no-matryoshka', action='store_true',
                   help='Bo MatryoshkaLoss (chi nen giu neu model goc duoc train bang no)')
    p.add_argument('--smoke', action='store_true',
                   help='Chay 30 buoc de do VRAM/toc do, khong luu model')
    args = p.parse_args()

    setup_logging(args.log)
    log.info('=' * 68)
    log.info('FINE-TUNE BI-ENCODER - DSC 2026 Task 1 LegalIR')
    log.info('=' * 68)

    if torch.cuda.is_available():
        log.info(f'GPU: {torch.cuda.get_device_name(0)} '
                 f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
    else:
        log.warning('KHONG CO GPU - se rat cham')

    ds, skipped = load_pairs(args.data, args.neg, limit=400 if args.smoke else 0)
    log.info(f'Du lieu: {len(ds)} mau ({skipped} cau bi bo vi thieu negative)')
    log.info(f'Cot: {ds.column_names}')

    log.info(f'Nap model goc: {args.base}')
    model = SentenceTransformer(args.base)
    model.max_seq_length = args.maxlen
    out_dim = model.get_sentence_embedding_dimension()
    log.info(f'Chieu dau ra: {out_dim}')

    if args.lora:
        from peft import LoraConfig, inject_adapter_in_model
        backbone = model[0].auto_model
        inject_adapter_in_model(
            LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                       target_modules=LORA_TARGETS, bias='none'),
            backbone)
        backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        tr = sum(x.numel() for x in backbone.parameters() if x.requires_grad)
        tot = sum(x.numel() for x in backbone.parameters())
        log.info(f'LoRA r={args.rank} tren {LORA_TARGETS} + gradient checkpointing')
        log.info(f'  train {tr / 1e6:.1f}M / {tot / 1e6:.0f}M tham so ({100 * tr / tot:.2f}%)')

    if args.no_cache:
        inner = MultipleNegativesRankingLoss(model)
        log.info('Loss: MultipleNegativesRankingLoss (khong GradCache)')
    else:
        inner = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini)
        log.info(f'Loss: CachedMultipleNegativesRankingLoss (GradCache, mini={args.mini})')
    if args.no_matryoshka:
        loss = inner
        log.info('Khong boc Matryoshka')
    else:
        dims = [d for d in ([out_dim] + MATRYOSHKA_DIMS) if d <= out_dim]
        dims = sorted(set(dims), reverse=True)
        loss = MatryoshkaLoss(model, inner, matryoshka_dims=dims)
        log.info(f'Boc trong MatryoshkaLoss, dims={dims}')

    steps = 30 if args.smoke else int(len(ds) * args.epochs / args.batch)
    seqs = args.batch * (2 + args.neg)
    log.info(f'Batch {args.batch} x (1 anchor + 1 positive + {args.neg} negative) '
             f'= {seqs} chuoi x {args.maxlen} token moi buoc')
    log.info(f'lr {args.lr}  warmup {args.warmup}  epochs {args.epochs}  -> {steps} buoc')

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        fp16=True,
        max_steps=30 if args.smoke else -1,
        logging_steps=5 if args.smoke else 10,
        save_strategy='no',
        report_to=[],
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=ds, loss=loss,
        callbacks=[ProgressCallback(steps)])

    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        log.error('HET VRAM. Thu ha --mini (vd 4) hoac --maxlen 192, '
                  'hoac giam --neg xuong 2.')
        raise
    el = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log.info('-' * 68)
    log.info(f'VRAM dinh: {peak:.2f} GB')

    if args.smoke:
        full_ds, _ = load_pairs(args.data, args.neg)
        full_steps = int(len(full_ds) * args.epochs / args.batch)
        log.info(f'Chay day du se la {full_steps} buoc -> '
                 f'uoc {full_steps * el / 30 / 60:.0f} phut')
        log.info('Smoke test xong, KHONG luu model.')
    else:
        if args.lora:
            log.info('Gop LoRA vao trong so goc ...')
            merge_and_strip(model[0].auto_model)
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        log.info(f'Da luu model: {args.out}')
        log.info('')
        log.info('Buoc tiep theo:')
        log.info(f'  cd ../Retrieval-LegalIR')
        log.info(f'  python encode.py --chunks ../Chunking-LegalIR/chunks.jsonl '
                 f'--out emb_ft --model ../Finetune-LegalIR/{args.out}')
        log.info(f'  cd ../Finetune-LegalIR && python eval_ft.py --emb ../Retrieval-LegalIR/emb_ft')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
