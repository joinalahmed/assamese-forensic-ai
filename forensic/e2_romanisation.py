"""E2: Romanisation variability and a lookup normaliser (AssameseBackTranslit).

sentence-aligned.txt alternates Roman line / native line. Paper reference (Table 6):
>=100 tokens: 728 types, mean 22.4 spellings, median 18, 98.6% >=2, 96.8% >=5;
lookup normaliser 80.6% token accuracy (85.6% in-vocab, 13.5% unseen, 19.1% whole sentences).
"""
import sys, json, os, re
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/backtranslit/sentence-aligned.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0
NATIVE = re.compile(r"[\u0980-\u09FF]")


def load(path):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    pairs, skipped = [], 0
    for r, n in zip(lines[0::2], lines[1::2]):
        r, n = r.split(), n.split()
        if len(r) == len(n) and r: pairs.append((r, n))
        else: skipped += 1
    return pairs, skipped


def variability(pairs):
    spell = defaultdict(Counter)   # native -> Counter(roman)
    reads = defaultdict(Counter)   # roman -> Counter(native)
    for r, n in pairs:
        for a, b in zip(r, n):
            if not NATIVE.search(b): continue  # Latin-script words that stay Roman are not romanisations
            spell[b][a.lower()] += 1; reads[a.lower()][b] += 1
    freq = {w: sum(c.values()) for w, c in spell.items()}
    out = {}
    for lo in (5, 20, 100):
        ks = np.array([len(spell[w]) for w in spell if freq[w] >= lo])
        out[f">={lo}"] = dict(types=int(len(ks)), mean=float(ks.mean()), median=float(np.median(ks)),
                              ge2=float((ks >= 2).mean()), ge5=float((ks >= 5).mean()))
    amb = [len(c) for a, c in reads.items() if sum(c.values()) >= 5]
    out["roman_ambiguous_share"] = float(np.mean(np.array(amb) > 1))
    out["roman_mean_readings_when_ambiguous"] = float(np.mean([x for x in amb if x > 1]))
    ntok = sum(len(r) for r, _ in pairs)
    out["n_tokens"] = ntok
    out["latin_stays_share"] = float(sum(a == b for r, n in pairs for a, b in zip(r, n)) / ntok)
    top = sorted(spell, key=lambda w: -freq[w])[:5]
    out["top_words"] = {w: dict(tokens=freq[w], variants=len(spell[w]), top=spell[w].most_common(4)) for w in top}
    most = max((w for w in spell if freq[w] >= 100), key=lambda w: len(spell[w]))
    out["most_variable"] = dict(word=most, variants=len(spell[most]), top=spell[most].most_common(4))
    return out, spell, reads


def lookup_normaliser(pairs):
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs)); cut = int(0.8 * len(pairs))
    tr, te = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    reads = defaultdict(Counter)
    for r, n in tr:
        for a, b in zip(r, n): reads[a][b] += 1
    table = {a: c.most_common(1)[0][0] for a, c in reads.items()}
    vocab = list(table)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit(vocab)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(vec.transform(vocab))
    unseen = sorted({a for r, _ in te for a in r if a not in table})
    fb = {}
    if unseen:
        _, ix = nn.kneighbors(vec.transform(unseen))
        fb = {u: table[vocab[i[0]]] for u, i in zip(unseen, ix)}
    tot = ok = iv = iv_ok = uv = uv_ok = sent_ok = 0
    for r, n in te:
        allok = True
        for a, b in zip(r, n):
            hit = (table.get(a) or fb[a]) == b
            tot += 1; ok += hit
            if a in table: iv += 1; iv_ok += hit
            else: uv += 1; uv_ok += hit
            allok &= hit
        sent_ok += allok
    return dict(n_train=len(tr), n_test=len(te), token_acc=ok / tot, unseen_share=uv / tot,
                in_vocab_acc=iv_ok / iv, unseen_acc=uv_ok / uv, sentence_acc=sent_ok / len(te))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pairs, skipped = load(PATH)
    print(f"{len(pairs)} aligned sentences ({skipped} skipped for token mismatch)", flush=True)
    var, spell, reads = variability(pairs)
    for k in (">=5", ">=20", ">=100"): print(k, var[k])
    print({k: v for k, v in var.items() if k not in (">=5", ">=20", ">=100", "top_words")})
    norm = lookup_normaliser(pairs); print(norm)
    json.dump(dict(variability=var, normaliser=norm), open(f"{OUT}/e2.json", "w"), indent=1, ensure_ascii=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    freq = {w: sum(c.values()) for w, c in spell.items()}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for lo in (5, 20, 100):
        ks = np.array([len(spell[w]) for w in spell if freq[w] >= lo]); xs = np.arange(1, 101)
        ax[0].plot(xs, [(ks >= x).mean() * 100 for x in xs], label=f"≥{lo} tokens")
    ax[0].set_xscale("log"); ax[0].set_xlabel("k = distinct Roman spellings"); ax[0].set_ylabel("native words with ≥ k spellings (%)"); ax[0].legend()
    ws = [w for w in spell if freq[w] >= 20]
    ax[1].scatter([freq[w] for w in ws], [len(spell[w]) for w in ws], s=4, alpha=.4)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("token frequency"); ax[1].set_ylabel("distinct Roman spellings")
    ax[2].bar(["all", "in-vocab", "unseen", "sentences"], [norm[k] * 100 for k in ("token_acc", "in_vocab_acc", "unseen_acc", "sentence_acc")])
    ax[2].set_ylabel("lookup normaliser accuracy (%)")
    fig.tight_layout(); fig.savefig(f"{OUT}/e2.png", dpi=150)
