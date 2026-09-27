#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: Legal Information Retrieval (LegalIR)
Validator local cho file nộp bài.

Kiem tra submission.zip / submission.json truoc khi nop len he thong,
theo dung cac rang buoc ma chuong trinh cham diem cua BTC ap dung.

Usage:
    python validate_submission.py submission.zip
    python validate_submission.py submission.json --ref public-official.json
    python validate_submission.py submission.zip --ref public-official.json --corpus selected-contexts/

Exit code:
    0 - hop le (co the con canh bao)
    1 - khong hop le
    2 - loi su dung (khong doc duoc file dau vao)
"""

import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter

MAX_DOCS_PER_QUESTION = 5
MAX_EXAMPLES = 5
SUBMISSION_FILENAME = 'submission.json'


# ---------------------------------------------------------------- reporting

class Report:
    """Gom toan bo loi/canh bao roi in mot lan, thay vi dung o loi dau tien."""

    def __init__(self):
        self.errors = []    # [(code, title, [chi tiet...])]
        self.warnings = []
        self.info = []

    def error(self, code, title, details=None):
        self.errors.append((code, title, details or []))

    def warn(self, code, title, details=None):
        self.warnings.append((code, title, details or []))

    def note(self, line):
        self.info.append(line)

    @property
    def ok(self):
        return not self.errors

    def render(self):
        out = []
        for line in self.info:
            out.append(f'  {line}')
        if self.info:
            out.append('')

        for label, items in (('LOI', self.errors), ('CANH BAO', self.warnings)):
            for code, title, details in items:
                out.append(f'[{label} {code}] {title}')
                for d in details[:MAX_EXAMPLES]:
                    out.append(f'    - {d}')
                if len(details) > MAX_EXAMPLES:
                    out.append(f'    ... va {len(details) - MAX_EXAMPLES} truong hop khac')
                out.append('')

        out.append('=' * 72)
        if self.errors:
            out.append(f'KET QUA: KHONG HOP LE  ({len(self.errors)} loai loi, '
                       f'{len(self.warnings)} canh bao)')
            out.append('Bai nop se bi tu choi. Vui long sua cac loi tren truoc khi nop.')
        elif self.warnings:
            out.append(f'KET QUA: HOP LE  (co {len(self.warnings)} canh bao)')
            out.append('File duoc chap nhan. Hay doc ky cac canh bao tren: '
                       'W001/W002 lam mat diem cua nhung cau lien quan.')
        else:
            out.append('KET QUA: HOP LE')
        out.append('=' * 72)
        return '\n'.join(out)


# ---------------------------------------------------------------- json utils

def _no_duplicate_keys(pairs):
    """object_pairs_hook: bat key trung lap (json.load mac dinh se nuot mat)."""
    seen = {}
    dups = []
    for k, v in pairs:
        if k in seen:
            dups.append(k)
        seen[k] = v
    if dups:
        seen['__duplicate_keys__'] = dups
    return seen


def load_json_bytes(raw, report, where):
    """Parse JSON tu bytes, phat hien BOM va key trung lap."""
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as e:
        report.error('E003', f'{where}: file khong phai UTF-8 hop le.',
                     [f'Vi tri byte {e.start}. Hay luu file voi encoding UTF-8.'])
        return None

    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as e:
        report.error('E003', f'{where}: khong parse duoc JSON.',
                     [f'Dong {e.lineno}, cot {e.colno}: {e.msg}'])
        return None


# ---------------------------------------------------------------- input read

def read_submission(path, report):
    """Doc submission tu .zip hoac .json. Tra ve bytes noi dung submission.json."""
    if not os.path.exists(path):
        print(f'Khong tim thay file: {path}', file=sys.stderr)
        sys.exit(2)

    if path.lower().endswith('.json'):
        report.warn('W004',
                    'Ban dang kiem tra truc tiep file .json.',
                    ['He thong chi nhan submission.zip. Nho nen file thanh zip '
                     'voi submission.json nam o thu muc GOC cua zip.'])
        with open(path, 'rb') as f:
            return f.read()

    if not zipfile.is_zipfile(path):
        report.error('E001', 'File nop khong phai dinh dang .zip hop le.')
        return None

    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith('/')]
        junk = [n for n in names
                if n.startswith('__MACOSX/') or os.path.basename(n) in ('.DS_Store', 'Thumbs.db')]
        real = [n for n in names if n not in junk]

        if junk:
            report.warn('W003', 'Zip chua file rac cua he dieu hanh (bo qua khi cham).',
                        junk)

        if SUBMISSION_FILENAME not in real:
            nested = [n for n in real if os.path.basename(n) == SUBMISSION_FILENAME]
            if nested:
                report.error('E001',
                             f'{SUBMISSION_FILENAME} nam trong thu muc con, khong phai goc zip.',
                             [f'Tim thay tai: {n}' for n in nested] +
                             ['Cach sua: mo thu muc chua submission.json roi nen '
                              'CHINH FILE do, khong nen ca thu muc.'])
            else:
                report.error('E001',
                             f'Khong tim thay {SUBMISSION_FILENAME} trong zip.',
                             [f'Zip dang chua: {", ".join(real[:10]) or "(rong)"}'])
            return None

        extra = [n for n in real if n != SUBMISSION_FILENAME]
        if extra:
            report.error('E002',
                         f'Zip phai chua DUY NHAT {SUBMISSION_FILENAME}.',
                         [f'File thua: {n}' for n in extra])

        return zf.read(SUBMISSION_FILENAME)


def load_reference(path, report):
    """Doc file de bai (public-official.json / private-official.json) lay danh sach question_id."""
    if path is None:
        return None
    if not os.path.exists(path):
        print(f'Khong tim thay file de bai: {path}', file=sys.stderr)
        sys.exit(2)
    with open(path, 'rb') as f:
        data = load_json_bytes(f.read(), report, 'File de bai')
    if data is None:
        sys.exit(2)
    if not isinstance(data, dict):
        print('File de bai khong phai JSON object.', file=sys.stderr)
        sys.exit(2)
    data.pop('__duplicate_keys__', None)
    return set(data.keys())


def load_corpus_ids(path):
    """Lay tap document_id tu thu muc selected-contexts (theo ten file context_<id>.json)."""
    if path is None:
        return None
    if not os.path.isdir(path):
        print(f'Khong tim thay thu muc corpus: {path}', file=sys.stderr)
        sys.exit(2)

    root = path
    entries = os.listdir(root)
    # Zip goc giai nen ra selected-contexts/selected-contexts/ - tu do vao mot cap.
    if not any(n.startswith('context_') for n in entries):
        for n in entries:
            sub = os.path.join(root, n)
            if os.path.isdir(sub) and any(x.startswith('context_') for x in os.listdir(sub)):
                root = sub
                entries = os.listdir(root)
                break

    ids = {n[len('context_'):-len('.json')]
           for n in entries
           if n.startswith('context_') and n.endswith('.json')}
    if not ids:
        print(f'Thu muc corpus khong chua file context_*.json: {path}', file=sys.stderr)
        sys.exit(2)
    return ids


# ---------------------------------------------------------------- validation

def validate(sub, ref_ids, corpus_ids, report):
    """Kiem tra noi dung submission da parse. Tra ve dict thong ke."""
    if not isinstance(sub, dict):
        report.error('E004',
                     'submission.json phai la mot JSON object o cap ngoai cung.',
                     [f'Dang doc duoc kieu: {type(sub).__name__}',
                      'Dinh dang dung: {"147194": {"answer": ["177504", "740"]}}'])
        return {}

    dup_qids = sub.pop('__duplicate_keys__', None)
    if dup_qids:
        report.error('E013', 'Co question_id bi lap lai trong submission.json.',
                     sorted(set(dup_qids)))

    if not sub:
        report.error('E004', 'submission.json rong.')
        return {}

    # --- doi chieu tap question_id voi de bai
    sub_ids = set(sub.keys())
    if ref_ids is not None:
        missing = sorted(ref_ids - sub_ids)
        extra = sorted(sub_ids - ref_ids)
        if missing:
            report.error('E005',
                         f'Thieu {len(missing)}/{len(ref_ids)} question_id so voi de bai.',
                         missing)
        if extra:
            report.error('E006',
                         f'Co {len(extra)} question_id khong ton tai trong de bai.',
                         extra)

    # --- kiem tra tung cau
    bad_value, no_answer, bad_answer_type = [], [], []
    bad_id_type, empty_id, duplicated, unknown_id = [], [], [], []
    empty_answer, over_limit = [], []
    length_dist = Counter()

    for qid in sorted(sub.keys()):
        value = sub[qid]

        if not isinstance(value, dict):
            bad_value.append(f'"{qid}": nhan duoc {type(value).__name__}, can object '
                             f'dang {{"answer": [...]}}')
            continue

        value.pop('__duplicate_keys__', None)

        if 'answer' not in value:
            no_answer.append(f'"{qid}": thieu key "answer" (co: {list(value.keys())})')
            continue

        answer = value['answer']
        if answer is None:
            bad_answer_type.append(f'"{qid}": "answer" = null. Day la gia tri mac dinh cua '
                                   f'file de bai, ban chua dien ket qua.')
            continue
        if isinstance(answer, str):
            bad_answer_type.append(f'"{qid}": "answer" la chuoi "{answer}", phai la list. '
                                   f'Dung ["{answer}"].')
            continue
        if not isinstance(answer, list):
            bad_answer_type.append(f'"{qid}": "answer" kieu {type(answer).__name__}, phai la list.')
            continue

        length_dist[len(answer)] += 1

        if len(answer) == 0:
            empty_answer.append(f'"{qid}": danh sach rong -> Recall = Precision = 0 cho cau nay.')
        elif len(answer) > MAX_DOCS_PER_QUESTION:
            over_limit.append(f'"{qid}": tra ve {len(answer)} document_id (toi da '
                              f'{MAX_DOCS_PER_QUESTION}) -> Recall = Precision = 0 cho cau nay.')

        for doc in answer:
            if not isinstance(doc, str):
                bad_id_type.append(f'"{qid}": document_id {doc!r} kieu {type(doc).__name__}, '
                                   f'phai la chuoi. Dung "{doc}".')
            elif not doc.strip():
                empty_id.append(f'"{qid}": chua document_id rong hoac toan khoang trang.')

        str_ids = [d for d in answer if isinstance(d, str)]
        dups = [d for d, c in Counter(str_ids).items() if c > 1]
        if dups:
            duplicated.append(f'"{qid}": document_id bi lap: {", ".join(sorted(dups))}')

        if corpus_ids is not None:
            for d in str_ids:
                if d.strip() and d not in corpus_ids:
                    unknown_id.append(f'"{qid}": document_id "{d}" khong co trong kho van ban.')

    if bad_value:
        report.error('E007', f'{len(bad_value)} cau co gia tri khong phai object.', bad_value)
    if no_answer:
        report.error('E008', f'{len(no_answer)} cau thieu key "answer".', no_answer)
    if bad_answer_type:
        report.error('E009', f'{len(bad_answer_type)} cau co "answer" sai kieu (phai la list).',
                     bad_answer_type)
    if bad_id_type:
        report.error('E010',
                     f'{len(bad_id_type)} document_id sai kieu du lieu (phai la CHUOI).',
                     bad_id_type + ['LUU Y: truong "id" trong context_*.json la so nguyen, '
                                    'nhung dap an bat buoc la chuoi. Hay ep kieu str() '
                                    'truoc khi ghi ra JSON.'])
    if empty_id:
        report.error('E011', f'{len(empty_id)} cau chua document_id rong.', empty_id)
    if duplicated:
        report.error('E012',
                     f'{len(duplicated)} cau co document_id bi lap lai.',
                     duplicated + ['ID lap khong tang Recall nhung lam giam Precision.'])
    if unknown_id:
        report.error('E014',
                     f'{len(unknown_id)} document_id khong ton tai trong kho van ban.',
                     unknown_id)

    if empty_answer:
        report.warn('W001', f'{len(empty_answer)} cau tra ve danh sach rong.', empty_answer)
    if over_limit:
        report.warn('W002',
                    f'{len(over_limit)} cau vuot qua {MAX_DOCS_PER_QUESTION} document_id.',
                    over_limit + ['Cac cau nay van duoc tinh vao trung binh voi diem 0, '
                                  'keo tut diem tong.'])

    return {'n_questions': len(sub), 'length_dist': length_dist,
            'n_over_limit': len(over_limit), 'n_empty': len(empty_answer)}


# ---------------------------------------------------------------- entrypoint

def main():
    parser = argparse.ArgumentParser(
        description='Validator local cho bai nop Task 1 - LegalIR (DSC 2026).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Vi du:\n'
               '  python validate_submission.py submission.zip --ref public-official.json\n'
               '  python validate_submission.py submission.zip --ref public-official.json '
               '--corpus selected-contexts/\n')
    parser.add_argument('submission', help='Duong dan toi submission.zip (hoac submission.json)')
    parser.add_argument('--ref', '-r', default=None,
                        help='File de bai (public-official.json / private-official.json) '
                             'de doi chieu danh sach question_id')
    parser.add_argument('--corpus', '-c', default=None,
                        help='Thu muc selected-contexts de kiem tra document_id co that')
    args = parser.parse_args()

    report = Report()

    print('=' * 72)
    print('DSC 2026 - Task 1: LegalIR - Kiem tra bai nop')
    print('=' * 72)
    print(f'Bai nop : {args.submission}')
    print(f'De bai  : {args.ref or "(khong doi chieu - nen truyen --ref)"}')
    print(f'Corpus  : {args.corpus or "(khong kiem tra document_id)"}')
    print('-' * 72)

    ref_ids = load_reference(args.ref, report)
    corpus_ids = load_corpus_ids(args.corpus)

    if ref_ids is None:
        report.warn('W005', 'Khong truyen --ref nen KHONG kiem tra duoc thieu/thua question_id.',
                    ['Day la loi khien bai nop bi tu choi thuong gap nhat. '
                     'Nen chay lai voi --ref public-official.json'])

    raw = read_submission(args.submission, report)
    stats = {}
    if raw is not None:
        sub = load_json_bytes(raw, report, 'submission.json')
        if sub is not None:
            stats = validate(sub, ref_ids, corpus_ids, report)

    if stats:
        dist = stats['length_dist']
        total_ids = sum(k * v for k, v in dist.items())
        report.note(f'So cau tra loi        : {stats["n_questions"]}'
                    + (f' / {len(ref_ids)} cau trong de bai' if ref_ids else ''))
        report.note(f'Tong so document_id   : {total_ids}')
        if dist:
            report.note('Phan bo so ID/cau     : '
                        + ', '.join(f'{k} ID: {v} cau' for k, v in sorted(dist.items())))
        if corpus_ids is not None:
            report.note(f'Kho van ban           : {len(corpus_ids)} van ban')

    print(report.render())
    return 0 if report.ok else 1


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass
    sys.exit(main())
