#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the numbers of two versions of the manuscript, e.g. across manuscript revisions.

Usage:  python check_numbers.py REFERENCE.docx CHECKED.docx [REPORT.md]

Every number in the text and in the tables (the reference list excluded) is collected from both files: integers, decimals,
thousands with commas, signs (− and -), percentages and p-values. The report lists the numbers that are missing in the
checked file and the numbers that appear only there, each with the sentence it stands in, so that a changed, lost or
invented number can be found in a minute. Identical files give an empty report."""
import sys, re, zipfile, html
from collections import Counter

NUM = re.compile(r'(?<![\w.])[−\-+]?\d[\d,]*(?:\.\d+)?%?')

def paragraphs(path):
    x = zipfile.ZipFile(path).read('word/document.xml').decode('utf8')
    out = []
    for m in re.finditer(r'<w:p[ >].*?</w:p>', x, re.S):
        t = html.unescape(''.join(re.findall(r'<w:t(?:\s[^>]*)?>([^<]*)</w:t>', m.group(0)))).strip()
        if t == 'References':
            break
        if t:
            out.append(t)
    return out

def tokens(paras):
    toks = Counter(); where = {}
    for p in paras:
        for m in NUM.finditer(p):
            t = m.group(0).replace('−', '-').rstrip(',')
            if not re.search(r'\d', t):
                continue
            toks[t] += 1
            a = max(0, m.start() - 70); b = min(len(p), m.end() + 70)
            where.setdefault(t, []).append(('…' if a else '') + p[a:b] + ('…' if b < len(p) else ''))
    return toks, where

def main():
    ref, chk = sys.argv[1], sys.argv[2]; rep = sys.argv[3] if len(sys.argv) > 3 else None
    tr, wr = tokens(paragraphs(ref)); tc, wc = tokens(paragraphs(chk))
    missing = tr - tc; extra = tc - tr
    lines = ['# Number check', '', 'Reference: `%s`  ' % ref, 'Checked: `%s`' % chk, '',
             'Numbers in the reference: %d (%d distinct). Numbers in the checked file: %d.' % (sum(tr.values()), len(tr), sum(tc.values())), '']
    if not missing and not extra:
        lines.append('No differences: every number of the reference appears in the checked file as often as before.')
    for title, diff, where in (('Missing or changed in the checked file', missing, wr), ('Only in the checked file', extra, wc)):
        if not diff:
            continue
        lines += ['## %s (%d)' % (title, sum(diff.values())), '']
        for t in sorted(diff, key=lambda s: (len(s), s)):
            lines.append('- **%s** ×%d' % (t, diff[t]))
            for ctx in where.get(t, [])[:3]:
                lines.append('  - %s' % ctx)
        lines.append('')
    text = '\n'.join(lines)
    if rep:
        open(rep, 'w', encoding='utf8').write(text); print('report written to', rep, '| missing %d, extra %d' % (sum(missing.values()), sum(extra.values())))
    else:
        print(text)

if __name__ == '__main__':
    main()
