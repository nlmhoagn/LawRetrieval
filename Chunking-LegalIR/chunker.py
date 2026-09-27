#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Chunker 3 tang cho corpus van ban phap luat.

Tang 1: cat theo "Dieu N"                     (85.9% van ban co cau truc nay)
Tang 2: Dieu qua dai -> cat tiep theo "Khoan" (^N.), van dai -> cua so co dinh
Tang 3: van ban khong co "Dieu"               -> cua so co dinh + overlap

Sau do: gop chunk qua ngan vao chunk ke, va prepend header
(ten van ban > Dieu > Khoan) vao moi chunk de chunk tu dung duoc mot minh.

Usage:
    python chunker.py --demo 43443 --contexts <selected-contexts/>
    python chunker.py --contexts <selected-contexts/> --out chunks.jsonl
"""

import argparse
import json
import os
import re
import sys

# --- nguong ---------------------------------------------------------------
TARGET_WORDS = 350      # Dieu dai hon nguong nay se bi cat nho
MIN_WORDS = 32          # chunk ngan hon nguong nay se bi gop vao chunk ke
WINDOW_WORDS = 256      # cua so co dinh cho tang 3
OVERLAP = 0.25          # ty le chong lan cua cua so co dinh
HEAD_MIN_WORDS = 30     # phan dau van ban (truoc Dieu 1) dai hon thi giu lai

RE_DIEU = re.compile(r'(?m)^\s*(Điều\s+\d+[a-zđ]?)\s*[.:]?')
RE_KHOAN = re.compile(r'(?m)^\s*(\d+)\.\s')
RE_CHUONG = re.compile(r'(?m)^\s*(Chương\s+[IVXLCDM\d]+)')


def clean(text):
    """Bo \\r cua bang HTML, gom khoang trang va dong trong."""
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def title_of(doc):
    """Lay tieu de: uu tien 'name', khong co thi suy tu slug trong link."""
    name = doc.get('name')
    if name:
        return name.replace('-', ' ').strip()
    link = doc.get('link', '')
    m = re.search(r'/([^/]+?)-\d+\.aspx', link)
    if m:
        return m.group(1).replace('-', ' ').strip()
    return ''


def split_fixed(text, window=WINDOW_WORDS, overlap=OVERLAP):
    """Cua so co dinh theo tu, co chong lan."""
    words = text.split()
    if len(words) <= window:
        return [text] if words else []
    stride = max(1, int(window * (1 - overlap)))
    out = []
    for i in range(0, len(words), stride):
        piece = words[i:i + window]
        if len(piece) < MIN_WORDS and out:
            break
        out.append(' '.join(piece))
        if i + window >= len(words):
            break
    return out


def split_by_khoan(text):
    """Cat mot Dieu thanh cac Khoan. Dong tieu de Dieu di theo Khoan dau tien."""
    marks = list(RE_KHOAN.finditer(text))
    if len(marks) < 2:
        return None
    bounds = [m.start() for m in marks] + [len(text)]
    head = text[:bounds[0]].strip()
    parts = []
    for i, m in enumerate(marks):
        body = text[bounds[i]:bounds[i + 1]].strip()
        parts.append((f'Khoản {m.group(1)}', body))
    if head:
        label, body = parts[0]
        parts[0] = (label, head + '\n' + body)
    return parts


def merge_labels(first, last):
    """Gop nhan cua hai manh lien nhau: 'Khoan 1' + 'Khoan 3' -> 'Khoan 1-3'."""
    if first == last:
        return first
    mf = re.match(r'^(.*?)(\d+)$', first)
    ml = re.match(r'^(.*?)(\d+)$', last)
    if mf and ml and mf.group(1) == ml.group(1):
        return f'{mf.group(1)}{mf.group(2)}-{ml.group(2)}'
    return f'{first} + {last}'


def merge_small(pieces):
    """Gop cac manh qua ngan vao manh ke truoc (hoac ke sau neu la manh dau).

    Nhan duoc gop theo, neu khong header se noi sai pham vi chunk.
    """
    out = []          # [[label_dau, label_cuoi, body], ...]
    for label, body in pieces:
        if out and len(body.split()) < MIN_WORDS:
            out[-1][1] = label
            out[-1][2] += '\n' + body
        else:
            out.append([label, label, body])
    if len(out) > 1 and len(out[0][2].split()) < MIN_WORDS:
        first, _, body = out.pop(0)
        out[0] = [first, out[0][1], body + '\n' + out[0][2]]
    return [(merge_labels(a, b), body) for a, b, body in out]


def chunk_document(doc_id, doc):
    """Tra ve list chunk cua mot van ban."""
    text = clean(doc.get('passage', ''))
    title = title_of(doc)

    if not text.split():
        # 20 van ban trong corpus co passage rong - van phai index bang tieu de,
        # vi 6 trong so do la dap an gold trong train.
        if not title:
            return []
        return [{'doc_id': doc_id, 'chunk_id': f'{doc_id}#t0', 'tier': 0,
                 'path': 'title-only', 'n_words': len(title.split()),
                 'text': title}]

    marks = list(RE_DIEU.finditer(text))

    # ---- Tang 3: khong co "Dieu" -> cua so co dinh ----------------------
    if not marks:
        pieces = [(f'Phần {i + 1}', p) for i, p in enumerate(split_fixed(text))]
        tier_of = lambda _: 3
    else:
        # ---- Tang 1: cat theo "Dieu" ------------------------------------
        bounds = [m.start() for m in marks] + [len(text)]
        pieces = []
        head = text[:bounds[0]].strip()
        if len(head.split()) >= HEAD_MIN_WORDS:
            pieces.append(('Phần mở đầu', head))
        for i, m in enumerate(marks):
            pieces.append((m.group(1), text[bounds[i]:bounds[i + 1]].strip()))
        pieces = merge_small(pieces)

        # ---- Tang 2: Dieu qua dai -> cat theo Khoan, van dai -> cua so ---
        expanded = []
        for label, body in pieces:
            if len(body.split()) <= TARGET_WORDS:
                expanded.append((label, body, 1))
                continue

            # Dong tieu de cua Dieu ("Dieu 4. Hieu luc thi hanh") phai di theo
            # MOI manh con, neu khong manh thu 2 tro di mat het ngu canh.
            heading = body.split('\n', 1)[0].strip()

            def with_heading(piece):
                return piece if piece.startswith(heading) else f'{heading}\n{piece}'

            khoan = split_by_khoan(body)
            if khoan:
                for klabel, kbody in merge_small(khoan):
                    if len(kbody.split()) <= TARGET_WORDS:
                        expanded.append((f'{label} > {klabel}', with_heading(kbody), 2))
                    else:
                        for j, w in enumerate(split_fixed(kbody)):
                            expanded.append((f'{label} > {klabel} > phần {j + 1}',
                                             with_heading(w), 2))
            else:
                for j, w in enumerate(split_fixed(body)):
                    expanded.append((f'{label} > phần {j + 1}', with_heading(w), 2))
        pieces = [(l, b) for l, b, _ in expanded]
        tiers = [t for _, _, t in expanded]
        tier_of = lambda i: tiers[i]

    # ---- header: chunk phai tu dung duoc mot minh ------------------------
    chunks = []
    for i, (label, body) in enumerate(pieces):
        header = ' > '.join(x for x in (title, label) if x)
        chunks.append({
            'doc_id': doc_id,
            'chunk_id': f'{doc_id}#{i}',
            'tier': tier_of(i),
            'path': label,
            'n_words': len(body.split()),
            'text': f'{header}\n{body}' if header else body,
        })
    return chunks


# ---------------------------------------------------------------- demo/CLI

def load_doc(ctx_dir, doc_id):
    path = os.path.join(ctx_dir, f'context_{doc_id}.json')
    if not os.path.exists(path):
        sys.exit(f'Khong tim thay: {path}')
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def demo(ctx_dir, doc_id, preview):
    doc = load_doc(ctx_dir, doc_id)
    raw = doc.get('passage', '')
    chunks = chunk_document(doc_id, doc)

    print('=' * 78)
    print(f'VAN BAN {doc_id}')
    print('=' * 78)
    print(f'name       : {doc.get("name", "(KHONG CO)")}')
    print(f'do dai goc : {len(raw):,} ky tu / {len(raw.split()):,} tu')
    print(f'sau clean  : {len(clean(raw)):,} ky tu / {len(clean(raw).split()):,} tu')
    print(f'so chunk   : {len(chunks)}')
    by_tier = {}
    for c in chunks:
        by_tier[c['tier']] = by_tier.get(c['tier'], 0) + 1
    print(f'theo tang  : ' + ', '.join(f'tang {t}: {n}' for t, n in sorted(by_tier.items())))
    print()
    print(f'{"#":<4}{"tang":<6}{"tu":<7}path')
    print('-' * 78)
    for c in chunks:
        print(f'{c["chunk_id"].split("#")[1]:<4}{c["tier"]:<6}{c["n_words"]:<7}{c["path"]}')

    print()
    print('=' * 78)
    print(f'NOI DUNG {min(preview, len(chunks))} CHUNK DAU')
    print('=' * 78)
    for c in chunks[:preview]:
        print(f'--- {c["chunk_id"]}  [tang {c["tier"]}, {c["n_words"]} tu]  {c["path"]}')
        body = c['text']
        print(body[:600] + ('\n  [...]' if len(body) > 600 else ''))
        print()


def build_all(ctx_dir, out_path, limit=0):
    files = sorted(f for f in os.listdir(ctx_dir)
                   if f.startswith('context_') and f.endswith('.json'))
    if limit:
        files = files[:limit]
    n_chunks = 0
    stats = {}
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, fn in enumerate(files, 1):
            doc_id = fn[len('context_'):-len('.json')]
            with open(os.path.join(ctx_dir, fn), encoding='utf-8-sig') as f:
                doc = json.load(f)
            for c in chunk_document(doc_id, doc):
                out.write(json.dumps(c, ensure_ascii=False) + '\n')
                n_chunks += 1
                stats[c['tier']] = stats.get(c['tier'], 0) + 1
            if i % 500 == 0:
                print(f'  {i}/{len(files)} van ban -> {n_chunks:,} chunk', flush=True)
    print(f'Xong: {len(files)} van ban -> {n_chunks:,} chunk -> {out_path}')
    print('Theo tang: ' + ', '.join(f'tang {t}: {n:,}' for t, n in sorted(stats.items())))


def main():
    p = argparse.ArgumentParser(description='Chunker 3 tang cho corpus LegalIR.')
    p.add_argument('--contexts', '-c', required=True, help='Thu muc selected-contexts')
    p.add_argument('--demo', '-d', help='In chi tiet chunk cua mot doc_id')
    p.add_argument('--preview', type=int, default=3, help='So chunk in noi dung o che do demo')
    p.add_argument('--out', '-o', help='Chunk toan bo corpus ra file .jsonl')
    p.add_argument('--limit', '-n', type=int, default=0,
                   help='Chi xu ly N van ban dau tien (0 = tat ca)')
    args = p.parse_args()

    ctx = args.contexts
    if not any(f.startswith('context_') for f in os.listdir(ctx)):
        for n in os.listdir(ctx):
            sub = os.path.join(ctx, n)
            if os.path.isdir(sub) and any(x.startswith('context_') for x in os.listdir(sub)):
                ctx = sub
                break

    if args.demo:
        demo(ctx, args.demo, args.preview)
    elif args.out:
        build_all(ctx, args.out, args.limit)
    else:
        p.error('Can --demo <doc_id> hoac --out <chunks.jsonl>')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
