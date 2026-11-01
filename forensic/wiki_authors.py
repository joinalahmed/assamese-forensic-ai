"""Build an author-labelled Assamese corpus from the Assamese Wikipedia full revision history.

Streams the pages-meta-history XML (bz2), and for every revision by a registered, non-bot editor extracts the
*lines added* relative to the previous revision of the same page, strips wiki markup, keeps lines that are mostly
Assamese script, and groups an editor's contributions into sessions (gap > SESSION_GAP hours starts a new one).

Output: JSONL, one record per (editor, session): {user, session, start, end, n_revisions, namespaces, text}.
Only text actually typed by the editor in that session is kept, so same-author trials across sessions are
genuine "different occasion" comparisons; talk/user-talk namespaces are the informal register.
"""
import sys, bz2, re, json, html
from collections import defaultdict
from datetime import datetime
import xml.etree.ElementTree as ET

SRC = sys.argv[1]; OUT = sys.argv[2]
SESSION_GAP_H, MIN_LINE_CHARS, MIN_SESSION_WORDS = 6, 20, 30
NS = "{http://www.mediawiki.org/xml/export-0.11/}"
BN = re.compile(r"[ঀ-৿]")
BOT = re.compile(r"bot\b|bot$|^bot|\bAWB\b", re.I)


def strip_markup(t):
    t = html.unescape(t)
    for _ in range(3): t = re.sub(r"\{\{[^{}]*\}\}", " ", t)          # templates (nested, 3 levels)
    t = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S); t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)             # [[a|b]] -> b
    t = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", t); t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"'{2,5}", "", t); t = re.sub(r"^[=]+\s*(.*?)\s*[=]+\s*$", r"\1", t, flags=re.M)
    t = re.sub(r"^[*#:;]+\s*", "", t, flags=re.M); t = re.sub(r"~{3,5}", " ", t)
    return t


def keep_line(l):
    l = l.strip()
    if len(l) < MIN_LINE_CHARS or l[0] in "{|!" or l.startswith("|}"): return None
    letters = sum(c.isalpha() for c in l)
    return l if letters and len(BN.findall(l)) / letters >= 0.6 else None


def added_lines(prev, cur):
    pl = set(prev.split("\n")); return [l for l in cur.split("\n") if l not in pl]


def main():
    sessions = defaultdict(list)   # user -> list of (ts, ns, text)
    n_pages = n_rev = 0; prev_text = ""; page_ns = None
    with bz2.open(SRC, "rb") as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.replace(NS, "")
            if tag == "ns": page_ns = el.text
            elif tag == "revision":
                n_rev += 1
                txt = el.find(NS + "text"); cur = txt.text or "" if txt is not None else ""
                c = el.find(NS + "contributor"); u = c.find(NS + "username") if c is not None else None
                ts = el.find(NS + "timestamp").text
                if u is not None and u.text and not BOT.search(u.text) and cur:
                    lines = [k for l in strip_markup("\n".join(added_lines(prev_text, cur))).split("\n") if (k := keep_line(l))]
                    if lines: sessions[u.text].append((ts, page_ns, "\n".join(lines)))
                prev_text = cur; el.clear()
            elif tag == "page":
                n_pages += 1; prev_text = ""; el.clear()
                if n_pages % 5000 == 0: print(f"{n_pages} pages, {n_rev} revisions, {len(sessions)} editors", file=sys.stderr, flush=True)
    n_out = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for user, items in sessions.items():
            items.sort(); cur = []; last = None; k = 0
            def flush():
                nonlocal k, n_out
                if not cur: return
                text = "\n".join(t for _, _, t in cur)
                if len(text.split()) >= MIN_SESSION_WORDS:
                    out.write(json.dumps(dict(user=user, session=k, start=cur[0][0], end=cur[-1][0], n_revisions=len(cur),
                                              namespaces=sorted({n for _, n, _ in cur}), text=text), ensure_ascii=False) + "\n"); n_out += 1
                k += 1
            for ts, ns, t in items:
                d = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                if last is not None and (d - last).total_seconds() > SESSION_GAP_H * 3600: flush(); cur = []
                cur.append((ts, ns, t)); last = d
            flush()
    print(f"done: {n_pages} pages, {n_rev} revisions, {len(sessions)} editors with text, {n_out} sessions written", file=sys.stderr)


if __name__ == "__main__":
    main()
