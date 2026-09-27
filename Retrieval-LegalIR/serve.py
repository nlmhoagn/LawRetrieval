#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSC 2026 - Task 1: LegalIR
Web UI local de tim kiem va doc van ban. Khong can cai them thu vien nao.

    python serve.py --index index --contexts "../LegalIR - Public Test/selected-contexts"

Roi mo http://127.0.0.1:8000

Trang trai: o tim kiem + danh sach ket qua (doc_id, path, diem, trich doan).
Trang phai: bam vao ket qua se render toan van ban theo cau truc Chuong/Dieu/Khoan,
tu nhay den dung Dieu da khop va to vang cac tu trong cau hoi.
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bm25 import BM25Index, tokenize

MAX_RENDER_CHARS = 400_000   # van ban dai hon se chi render vung quanh Dieu khop

RE_CHUONG = re.compile(r'^Chương\s+[IVXLCDM\d]')
RE_DIEU = re.compile(r'^Điều\s+(\d+[a-zđ]?)')
RE_MUC = re.compile(r'^Mục\s+\d')
RE_KHOAN = re.compile(r'^\d+\.\s')
RE_DIEM = re.compile(r'^[a-zđ]\)\s')

# dong bat dau mot don vi cau truc moi -> khong duoc noi vao dong truoc
RE_STRUCT = re.compile(r'^(Chương\s+[IVXLCDM\d]|Điều\s+\d|Mục\s+\d|Phần\s+[IVXLCDM\d]'
                       r'|\d+\.\s|[a-zđ]\)\s|[-+*]\s)')
RE_SEP = re.compile(r'^[-_=–—\s.]+$')       # dong ke ngan cach "--------"
RE_ENDS = re.compile(r'[.;:!?]["\')\]]?$')  # dong da ket thuc y

STATE = {}


def clean(text):
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def highlight(escaped, terms):
    """To vang tu khoa. Chay TREN chuoi da escape nen khong pha HTML."""
    if not terms:
        return escaped
    pat = '|'.join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.sub(f'(?<!\\w)({pat})(?!\\w)', r'<mark>\1</mark>', escaped, flags=re.I)


def reflow(text):
    """Noi lai cac dong bi ngat giua cau.

    Van ban duoc cao tu HTML nen xuong dong theo be ngang cua bang, khong theo cau:
    "Điều 1. Phạm" / "vi điều chỉnh" la hai dong rieng. Noi lai khi dong truoc
    chua ket thuc y va dong sau khong mo mot don vi cau truc moi.
    """
    out = []
    for raw in text.split('\n'):
        s = raw.strip()
        if not s or RE_SEP.match(s):
            continue
        if (out and not RE_STRUCT.match(s) and not RE_ENDS.search(out[-1])
                and not RE_SEP.match(out[-1])):
            out[-1] += ' ' + s
        else:
            out.append(s)
    return out


def anchor_of(label):
    m = re.match(r'Điều\s+(\d+[a-zđ]?)', label)
    return f'dieu-{m.group(1)}' if m else ''


def render_doc(text, terms, focus=''):
    """Bien van ban tho thanh HTML co cau truc, giong preview markdown."""
    text = clean(text)
    truncated = False

    if len(text) > MAX_RENDER_CHARS:
        pos = 0
        if focus:
            m = re.search(r'(?m)^' + re.escape(focus) + r'\b', text)
            if m:
                pos = max(0, m.start() - MAX_RENDER_CHARS // 4)
        text = text[pos:pos + MAX_RENDER_CHARS]
        truncated = True

    out = []
    seen = set()
    for s in reflow(text):
        esc = highlight(html.escape(s), terms)
        if RE_CHUONG.match(s):
            out.append(f'<h2>{esc}</h2>')
        elif RE_DIEU.match(s):
            a = anchor_of(s)
            if a in seen:
                a = ''
            seen.add(a)
            out.append(f'<h3{f" id={a!r}" if a else ""}>{esc}</h3>')
        elif RE_MUC.match(s):
            out.append(f'<h4>{esc}</h4>')
        elif RE_KHOAN.match(s):
            out.append(f'<p class="khoan">{esc}</p>')
        elif RE_DIEM.match(s):
            out.append(f'<p class="diem">{esc}</p>')
        else:
            out.append(f'<p>{esc}</p>')
    return '\n'.join(out), truncated


def load_raw_doc(doc_id):
    path = os.path.join(STATE['contexts'], f'context_{doc_id}.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def do_search(q, k, mode, agg):
    idx = STATE['idx']
    scores, terms = idx.score(q)
    results = []

    if mode == 'chunk':
        n = min(k, int((scores > 0).sum()))
        if n:
            import numpy as np
            top = np.argpartition(-scores, n - 1)[:n]
            top = top[np.argsort(-scores[top])]
            for ci in top:
                c = idx.read_chunk(int(ci))
                head, _, body = c['text'].partition('\n')
                body = ' '.join(reflow(body))
                results.append({'doc_id': c['doc_id'], 'path': c['path'],
                                'score': round(float(scores[ci]), 2),
                                'title': head.split(' > ')[0],
                                'n_words': c['n_words'], 'tier': c['tier'],
                                'snippet': body[:260], 'n_chunk': 1})
    else:
        for did, s, ci, n in idx.rank_docs(scores, topk=k, agg=agg):
            c = idx.read_chunk(ci)
            head, _, body = c['text'].partition('\n')
            body = ' '.join(reflow(body))
            results.append({'doc_id': did, 'path': c['path'], 'score': round(s, 2),
                            'title': head.split(' > ')[0], 'n_words': c['n_words'],
                            'tier': c['tier'], 'snippet': body[:260], 'n_chunk': n})
    return {'terms': terms, 'results': results}


def do_doc(doc_id, q, focus):
    doc = load_raw_doc(doc_id)
    if doc is None:
        return {'error': f'Khong tim thay van ban {doc_id}'}
    # chi to vang cac tu THUC SU nam trong tu dien index: tu qua pho bien nhu
    # "dieu", "quy", "cua" da bi loai luc build, to vang chung chi lam roi mat.
    vocab = STATE['idx'].vocab
    terms = {t for t in tokenize(q) if t in vocab} if q else set()
    body, truncated = render_doc(doc.get('passage', ''), terms, focus)
    if not body:
        body = '<p class="empty">(Van ban nay co passage rong - day la 1 trong 20 ' \
               'van ban rong cua corpus)</p>'
    return {'doc_id': doc_id, 'name': doc.get('name', ''), 'link': doc.get('link', ''),
            'html': body, 'truncated': truncated, 'anchor': anchor_of(focus)}


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LegalIR - Tim kiem van ban phap luat</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--line:#e6e4df;--fg:#1f1e1c;--dim:#6b6862;
      --acc:#8a5a2b;--mark:#ffe89a;--code:#f3f1ec}
@media(prefers-color-scheme:dark){:root{--bg:#191817;--panel:#211f1e;--line:#35322f;
      --fg:#e9e6e1;--dim:#9a958d;--acc:#d99a5b;--mark:#7a5f14;--code:#2a2724}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel);
       display:flex;gap:10px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0 12px 0 0;font-weight:650;letter-spacing:-.01em}
input[type=search]{flex:1;min-width:260px;padding:9px 13px;border:1px solid var(--line);
       border-radius:8px;background:var(--bg);color:var(--fg);font-size:15px}
input[type=search]:focus{outline:2px solid var(--acc);outline-offset:-1px}
select,button{padding:8px 11px;border:1px solid var(--line);border-radius:8px;
       background:var(--bg);color:var(--fg);font-size:13px;cursor:pointer}
button{background:var(--acc);color:#fff;border-color:transparent;font-weight:600}
main{flex:1;display:flex;min-height:0}
#list{width:44%;min-width:320px;overflow-y:auto;border-right:1px solid var(--line);padding:8px}
#view{flex:1;overflow-y:auto;padding:26px 34px;background:var(--panel)}
.hit{padding:11px 13px;border:1px solid transparent;border-radius:9px;cursor:pointer;
     margin-bottom:4px}
.hit:hover{background:var(--code)}
.hit.sel{background:var(--code);border-color:var(--acc)}
.hit .top{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.hit .path{font-weight:650;color:var(--acc);font-size:14px}
.hit .score{font-variant-numeric:tabular-nums;color:var(--dim);font-size:12px;flex-shrink:0}
.hit .meta{color:var(--dim);font-size:12px;margin:3px 0 5px}
.hit .snip{color:var(--dim);font-size:13px;line-height:1.5;
     display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.status{padding:10px 14px;color:var(--dim);font-size:13px}
#view h2{font-size:17px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
#view h3{font-size:15.5px;margin:22px 0 8px;color:var(--acc)}
#view h4{font-size:14px;margin:16px 0 6px;color:var(--dim)}
#view p{margin:7px 0}
#view p.khoan{margin-left:12px}
#view p.diem{margin-left:30px;color:var(--dim)}
#view h3:target{background:var(--code);padding:6px 10px;margin-left:-10px;border-radius:6px}
mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px}
.docmeta{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:8px}
.docmeta .id{font-size:12px;color:var(--dim);font-family:ui-monospace,monospace}
.docmeta a{color:var(--acc);font-size:12px;word-break:break-all}
.warn{background:var(--code);padding:9px 13px;border-radius:8px;font-size:13px;
      color:var(--dim);margin:10px 0}
.empty{color:var(--dim);font-style:italic}
</style></head><body>
<header>
  <h1>LegalIR</h1>
  <input type="search" id="q" placeholder="Nhập câu hỏi pháp luật..." autofocus>
  <select id="mode"><option value="doc">Gộp theo văn bản</option>
                    <option value="chunk">Từng chunk</option></select>
  <select id="agg"><option value="top3">top3</option><option value="max">max</option>
                   <option value="top3_log">top3_log</option></select>
  <select id="k"><option>5</option><option selected>10</option><option>20</option>
                 <option>50</option></select>
  <button id="go">Tìm</button>
</header>
<main>
  <div id="list"><div class="status">Nhập câu hỏi rồi nhấn Enter.</div></div>
  <div id="view"><div class="status">Bấm vào một kết quả để đọc toàn văn.</div></div>
</main>
<script>
const $=s=>document.querySelector(s);
let lastQuery='';

async function search(){
  const q=$('#q').value.trim(); if(!q) return;
  lastQuery=q;
  $('#list').innerHTML='<div class="status">Đang tìm...</div>';
  const p=new URLSearchParams({q,k:$('#k').value,mode:$('#mode').value,agg:$('#agg').value});
  const r=await (await fetch('/api/search?'+p)).json();
  if(!r.results.length){$('#list').innerHTML='<div class="status">Không có kết quả.</div>';return;}
  $('#list').innerHTML='<div class="status">'+r.results.length+' kết quả · từ khớp: '
    +r.terms.join(', ')+'</div>'+r.results.map((h,i)=>`
    <div class="hit" data-i="${i}" data-doc="${h.doc_id}" data-focus="${esc(h.path)}">
      <div class="top"><span class="path">${esc(h.path)}</span>
        <span class="score">${h.score}</span></div>
      <div class="meta">doc ${h.doc_id} · tầng ${h.tier} · ${h.n_words} từ${
        h.n_chunk>1?' · '+h.n_chunk+' chunk khớp':''}</div>
      <div class="snip">${esc(h.snippet)}</div>
    </div>`).join('');
  [...document.querySelectorAll('.hit')].forEach(el=>
    el.onclick=()=>{document.querySelectorAll('.hit').forEach(x=>x.classList.remove('sel'));
                    el.classList.add('sel');
                    openDoc(el.dataset.doc, el.dataset.focus);});
  document.querySelector('.hit').click();
}

async function openDoc(id,focus){
  $('#view').innerHTML='<div class="status">Đang tải văn bản...</div>';
  const p=new URLSearchParams({q:lastQuery,focus:(focus||'').split(' > ')[0]});
  const d=await (await fetch('/api/doc/'+id+'?'+p)).json();
  if(d.error){$('#view').innerHTML='<div class="status">'+esc(d.error)+'</div>';return;}
  $('#view').innerHTML=`<div class="docmeta">
      <div class="id">doc ${d.doc_id}</div>
      <div>${esc((d.name||'(không có trường name)').replace(/-/g,' '))}</div>
      ${d.link?`<a href="${esc(d.link)}" target="_blank" rel="noopener">${esc(d.link)}</a>`:''}
    </div>${d.truncated?'<div class="warn">Văn bản quá dài — chỉ hiển thị phần quanh điều khớp.</div>':''}
    ${d.html}`;
  $('#view').scrollTop=0;
  if(d.anchor){const t=document.getElementById(d.anchor);
    if(t){t.scrollIntoView({block:'start'});$('#view').scrollTop-=70;}}
}

function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;',
  '>':'&gt;','"':'&quot;'}[c]));}
$('#go').onclick=search;
$('#q').onkeydown=e=>{if(e.key==='Enter')search();};
['mode','agg','k'].forEach(id=>$('#'+id).onchange=()=>{if(lastQuery)search();});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', f'{ctype}; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        one = lambda k, d='': qs.get(k, [d])[0]

        if parsed.path == '/':
            return self._send(PAGE, 'text/html')

        if parsed.path == '/api/search':
            q = one('q')
            if not q:
                return self._send(json.dumps({'terms': [], 'results': []}))
            k = max(1, min(100, int(one('k', '10'))))
            res = do_search(q, k, one('mode', 'doc'), one('agg', 'top3'))
            return self._send(json.dumps(res, ensure_ascii=False))

        if parsed.path.startswith('/api/doc/'):
            doc_id = parsed.path[len('/api/doc/'):]
            if not doc_id.isdigit():
                return self._send(json.dumps({'error': 'doc_id khong hop le'}))
            res = do_doc(doc_id, one('q'), one('focus'))
            return self._send(json.dumps(res, ensure_ascii=False))

        self.send_error(404)


def main():
    p = argparse.ArgumentParser(description='Web UI local cho BM25 LegalIR.')
    p.add_argument('--index', '-i', default='index')
    p.add_argument('--contexts', '-c', required=True, help='Thu muc selected-contexts')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--no-open', action='store_true', help='Khong tu mo trinh duyet')
    args = p.parse_args()

    ctx = args.contexts
    if not any(f.startswith('context_') for f in os.listdir(ctx)):
        for n in os.listdir(ctx):
            sub = os.path.join(ctx, n)
            if os.path.isdir(sub) and any(x.startswith('context_') for x in os.listdir(sub)):
                ctx = sub
                break
    STATE['contexts'] = ctx

    print(f'Dang nap index {args.index} ...')
    STATE['idx'] = BM25Index(args.index)
    print(f'  {STATE["idx"].meta["n_chunks"]:,} chunk, '
          f'tu dien {STATE["idx"].meta["vocab_size"]:,}')

    url = f'http://127.0.0.1:{args.port}'
    print(f'San sang: {url}   (Ctrl+C de dung)')
    if not args.no_open:
        webbrowser.open(url)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
