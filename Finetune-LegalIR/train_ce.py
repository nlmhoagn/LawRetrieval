#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Fine-tune cross-encoder (tang 3) bang LoRA.

VI SAO PHAI DUNG LoRA
  Cross-encoder la xlm-roberta-LARGE, 568M tham so. Full fine-tune can:
      fp32 master + grad + Adam(m,v) = 568M x 16 byte = 9.1 GB
  trong khi VRAM chi co 6.4 GB. LoRA chi train adapter (~5M tham so) nen optimizer
  state gan nhu bang 0; model goc nam fp16 chi ton 1.1 GB.

VI SAO FINE-TUNE CROSS-ENCODER LA DANG GIA NHAT
  Do tren validation: beta toi uu la 0.5, khong phai 1.0. Nghia la cross-encoder
  off-the-shelf CO tin hieu nhung chua du manh de cam lai - chu ky dien hinh cua
  lech mien. Fine-tune chua dung cho do.
  Ngoai ra no KHONG can encode lai corpus (khac bi-encoder, moi lan thu mat 36 phut).

DU LIEU
  Dung lai 5.999 cap da dao. Voi BCE: (query, positive)=1, (query, negative)=0.

Usage:
    python train_ce.py --smoke              # 30 buoc, do VRAM va toc do
    python train_ce.py                      # train that
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from datasets import Dataset

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from peft import LoraConfig, inject_adapter_in_model
from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                 CrossEncoderTrainingArguments)
from sentence_transformers.cross_encoder.losses import (BinaryCrossEntropyLoss,
                                                        CachedMultipleNegativesRankingLoss)
from transformers import TrainerCallback

BASE_MODEL = 'AITeamVN/Vietnamese_Reranker'
# CHI nham attention. Neu them 'dense' thi LoRA bam ca FFN 1024->4096 va 4096->1024,
# activation phinh len -> OOM ngay o 6.4 GB VRAM.
LORA_TARGETS = ['query', 'key', 'value']

log = logging.getLogger('ce')


def setup_logging(logfile):
    os.makedirs(os.path.dirname(logfile) or '.', exist_ok=True)
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(logfile, encoding='utf-8')):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False


class ProgressCallback(TrainerCallback):
    def __init__(self, total):
        self.total = total
        self.t0 = None

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
        log.info(f'buoc {step:>5}/{self.total} ({100*step/max(self.total,1):5.1f}%)  '
                 f'loss {logs["loss"]:.4f}  lr {logs.get("learning_rate", 0):.2e}  '
                 f'VRAM {vram:.2f}GB  {sps:.2f} buoc/s  con ~{eta:.0f} phut')


def merge_and_strip(module):
    """Gop delta LoRA vao trong so goc roi go lop boc, tra ve nn.Linear thuan.

    Can lam vay de model luu ra load duoc bang AutoModelForSequenceClassification
    binh thuong - rerank.py khong phai biet gi ve LoRA.
    """
    from peft.tuners.lora import LoraLayer
    for name, child in list(module.named_children()):
        if isinstance(child, LoraLayer):
            child.merge()
            setattr(module, name, child.base_layer)
        else:
            merge_and_strip(child)


def build_dataset(path, n_neg, listwise, limit=0):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            negs = r['negatives'][:n_neg]
            if len(negs) < n_neg:
                continue
            if listwise:
                item = {'query': r['query'], 'positive': r['positive']}
                for j, ng in enumerate(negs):
                    item[f'negative_{j + 1}'] = ng
                rows.append(item)
            else:
                rows.append({'query': r['query'], 'passage': r['positive'], 'label': 1.0})
                for ng in negs:
                    rows.append({'query': r['query'], 'passage': ng, 'label': 0.0})
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', default='data/pairs.jsonl')
    p.add_argument('--out', '-o', default='models/reranker-ft')
    p.add_argument('--log', default='logs/train_ce.log')
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--neg', type=int, default=4)
    p.add_argument('--maxlen', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4, help='LoRA chiu duoc lr cao hon')
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--listwise', action='store_true',
                   help='Dung CachedMultipleNegativesRankingLoss thay vi BCE')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed. Doi seed de train nhieu checkpoint cho ensemble.')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()

    # Set seed for reproducibility and ensemble diversity
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    setup_logging(args.log)
    log.info('=' * 68)
    log.info('FINE-TUNE CROSS-ENCODER (LoRA) - DSC 2026 Task 1')
    log.info('=' * 68)
    if torch.cuda.is_available():
        log.info(f'GPU: {torch.cuda.get_device_name(0)} '
                 f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')

    ds = build_dataset(args.data, args.neg, args.listwise,
                       limit=300 if args.smoke else 0)
    log.info(f'Du lieu: {len(ds)} dong | cot {ds.column_names}')

    log.info(f'Nap model goc: {BASE_MODEL}')
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=args.maxlen)

    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                      target_modules=LORA_TARGETS, bias='none')
    # inject_adapter_in_model gan LoRA TAI CHO, giu nguyen class va chu ky forward.
    # get_peft_model se boc bang PeftModelForSequenceClassification, ma lop boc do
    # doi chu ky forward -> sentence-transformers goi vao la vo.
    inject_adapter_in_model(lora, model.model)
    for name, prm in model.model.named_parameters():
        prm.requires_grad = ('lora_' in name) or ('classifier' in name)

    # bat buoc voi 6.4 GB VRAM: doi thoi gian lay bo nho
    model.model.enable_input_require_grads()
    model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    log.info('Da bat gradient checkpointing')
    trainable = sum(p_.numel() for p_ in model.model.parameters() if p_.requires_grad)
    total = sum(p_.numel() for p_ in model.model.parameters())
    log.info(f'LoRA r={args.rank} tren {LORA_TARGETS}')
    log.info(f'  train {trainable/1e6:.1f}M / {total/1e6:.0f}M tham so '
             f'({100*trainable/total:.2f}%)')

    if args.listwise:
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=4)
        log.info('Loss: CachedMultipleNegativesRankingLoss (listwise)')
    else:
        loss = BinaryCrossEntropyLoss(model)
        log.info('Loss: BinaryCrossEntropyLoss (pointwise)')

    eff = args.batch * args.accum
    steps = 30 if args.smoke else int(len(ds) * args.epochs / eff)
    log.info(f'batch {args.batch} x accum {args.accum} = {eff} | maxlen {args.maxlen} '
             f'| lr {args.lr} -> {steps} buoc')

    targs = CrossEncoderTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        fp16=True,
        max_steps=30 if args.smoke else -1,
        logging_steps=5 if args.smoke else 20,
        save_strategy='no',
        report_to=[],
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=ds, loss=loss,
                                  callbacks=[ProgressCallback(steps)])
    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        log.error('HET VRAM. Thu --batch 4 --accum 4, hoac --maxlen 256.')
        raise
    el = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log.info('-' * 68)
    log.info(f'Xong sau {el/60:.1f} phut | VRAM dinh {peak:.2f} GB')

    if args.smoke:
        full = build_dataset(args.data, args.neg, args.listwise)
        fs = int(len(full) * args.epochs / eff)
        log.info(f'Chay day du: {fs} buoc -> uoc {fs * el / 30 / 60:.0f} phut')
        log.info('Smoke test xong, KHONG luu model.')
        return 0

    # gop LoRA vao model goc roi luu nguyen ban -> rerank.py dung duoc ngay
    log.info('Gop LoRA vao trong so goc ...')
    merge_and_strip(model.model)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    log.info(f'Da luu: {args.out}')
    log.info('')
    log.info('Buoc tiep theo - do lai tren validation:')
    log.info(f'  python sweep_ce.py --rerank {args.out}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
