"""E1: authorship on short texts (Arambhani Assamese poetry).

Closed-set attribution (Table 4, Fig 4a/b) and open-set verification (Table 5, Fig 4c).
Paper reference: 840 poems / 178 authors / median 75 words; 20 authors with >=10 poems = 351 poems.
"""
import re, json, zlib, sys
from collections import defaultdict
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from metrics import LogisticCalibrator, report, tippett

SEED = 0
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/poems/arambhani.txt"
MIN_WORDS = 16  # inferred: gives the paper's 351 poems / 20 authors with >=10 poems


def author_of(h):
    """Headers come as 'কবিতা :: title --- author', 'title -- author' or 'title - কবি author'."""
    h = re.sub(r"^কবিতা\s*::\s*", "", h.strip())
    m = re.split(r"\s*-{2,}\s*", h)
    if len(m) == 1: m = re.split(r"\s+-\s+", h)
    if len(m) == 1: return None
    a = re.sub(r"\s*-\s*Assamese.*$", "", re.sub(r"^কবি\s+", "", m[-1].strip()))
    return " ".join(a.split()) or None


def load_poems(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    idx = [i for i, l in enumerate(lines) if i > 0 and re.fullmatch(r"=+", l.strip())]
    poems = []
    for k, i in enumerate(idx):
        end = idx[k + 1] - 1 if k + 1 < len(idx) else len(lines)
        body = [l.rstrip() for l in lines[i + 1:end]]
        a = author_of(lines[i - 1])
        text = "\n".join(body).strip()
        ws = " ".join(text.split())
        if a and len(ws.split()) >= MIN_WORDS: poems.append(dict(author=a, text=text, ws=ws))
    return poems


def feats_fit(train_txt, kind):
    if kind == "char14+word":
        c = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True)
        w = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)
        c.fit(train_txt); w.fit(train_txt)
        return lambda t: hstack([c.transform(t), w.transform(t)]).tocsr()
    if kind == "char14":
        v = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True).fit(train_txt)
    elif kind == "char25":
        v = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True).fit(train_txt)
    elif kind == "word1":
        v = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True).fit(train_txt)
    elif kind == "char13counts":
        v = CountVectorizer(analyzer="char", ngram_range=(1, 3)).fit(train_txt)
    return v.transform


def zlen(s): return len(zlib.compress(s.encode("utf-8"), 9))


def zlib_predict(train_txt, train_y, test_txt):
    by = defaultdict(list)
    for t, y in zip(train_txt, train_y): by[y].append(t)
    prof = {y: "\n".join(v) for y, v in by.items()}
    base = {y: zlen(p) for y, p in prof.items()}
    preds = []
    for q in test_txt:
        # extra compressed bytes for the questioned text given each author's profile
        d = {y: zlen(p + "\n" + q) - base[y] for y, p in prof.items()}
        preds.append(min(d, key=d.get))
    return np.array(preds)


def closed_set(poems, res):
    cnt = defaultdict(int)
    for p in poems: cnt[p["author"]] += 1
    keep = {a for a, n in cnt.items() if n >= 10}
    P = [p for p in poems if p["author"] in keep]
    y = np.array([p["author"] for p in P]); X = [p["text"] for p in P]
    res["closed_set_n_authors"], res["closed_set_n_poems"] = len(keep), len(P)
    res["majority_rate"] = float(max(np.unique(y, return_counts=True)[1]) / len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    configs = [
        ("char 1–4 + word unigrams / LinearSVC", "char14+word", lambda: LinearSVC()),
        ("char 1–4 / LinearSVC", "char14", lambda: LinearSVC()),
        ("char 2–5 / LinearSVC", "char25", lambda: LinearSVC()),
        ("char 1–3 counts / multinomial NB", "char13counts", lambda: MultinomialNB(alpha=0.01, fit_prior=False)),
        ("word unigrams / LinearSVC", "word1", lambda: LinearSVC()),
        ("char 1–4 / logistic regression", "char14", lambda: LogisticRegression(max_iter=2000, C=20)),
        ("zlib compression distance", None, None),
    ]
    table = {}
    for name, kind, mk in configs:
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            trt, tet = [X[i] for i in tr], [X[i] for i in te]
            if kind is None:
                pred = zlib_predict(trt, y[tr], tet)
            else:
                f = feats_fit(trt, kind); pred = mk().fit(f(trt), y[tr]).predict(f(tet))
            accs.append((pred == y[te]).mean()); f1s.append(f1_score(y[te], pred, average="macro"))
        table[name] = dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), macro_f1=float(np.mean(f1s)))
        print(f"  {name:42s} {np.mean(accs):.3f} ± {np.std(accs):.3f}  F1 {np.mean(f1s):.3f}", flush=True)
    res["table4"] = table

    # query-length curve: train on full poems, truncate only questioned poem; whitespace-normalised
    ws = [p["ws"] for p in P]
    curve = {}
    for n in [15, 30, 50, 75, 100, None]:
        accs = []
        for tr, te in skf.split(X, y):
            trt = [ws[i] for i in tr]
            f = feats_fit(trt, "char14"); clf = LinearSVC().fit(f(trt), y[tr])
            q = [" ".join(ws[i].split()[:n]) if n else ws[i] for i in te]
            accs.append((clf.predict(f(q)) == y[te]).mean())
        curve["full" if n is None else str(n)] = float(np.mean(accs))
        print(f"  truncate {n}: {np.mean(accs):.3f}", flush=True)
    res["query_length_curve"] = curve


def verification(poems, res, make_plot=True):
    """Author-disjoint halves; TF-IDF + calibrator fitted on one half, evaluated on the other."""
    rng = np.random.RandomState(SEED)
    by = defaultdict(list)
    for p in poems: by[p["author"]].append(p["ws"])
    authors = sorted(a for a, v in by.items() if len(v) >= 2)
    rng.shuffle(authors)
    halves = [authors[::2], authors[1::2]]

    def trials(auth, mode, minpoems, max_ds=None):
        auth = [a for a in auth if len(by[a]) >= minpoems]
        # items: (questioned text, known (text | list of texts), label, author_q, author_k)
        ss, ds = [], []
        if mode == "pair":
            for a in auth:
                v = by[a]
                for i in range(len(v)):
                    for j in range(i + 1, len(v)): ss.append((v[i], [v[j]]))
            allp = [(a, t) for a in auth for t in by[a]]
            for _ in range(max_ds or 6000):
                (a1, t1), (a2, t2) = [allp[k] for k in rng.choice(len(allp), 2, replace=False)]
                if a1 != a2: ds.append((t1, [t2]))
        else:  # profile: questioned poem vs. all other poems of the candidate author
            for a in auth:
                for i, q in enumerate(by[a]):
                    ss.append((q, by[a][:i] + by[a][i + 1:]))
                    for b in auth:
                        if b != a: ds.append((q, by[b]))
            if max_ds and len(ds) > max_ds:
                ds = [ds[k] for k in rng.choice(len(ds), max_ds, replace=False)]
        return ss, ds

    def score(vec, trials_):
        out = []
        for q, known in trials_:
            qv = vec.transform([q]); kv = normalize(vec.transform(known))
            prof = normalize(np.asarray(kv.mean(axis=0)))
            out.append(float((normalize(qv) @ prof.T)[0, 0]))
        return np.array(out)

    protocols = [("Pairwise, one poem", "pair", 2), ("Profile, ≥3 poems", "profile", 3), ("Profile, ≥10 poems", "profile", 10)]
    table5, plot = {}, None
    for name, mode, minp in protocols:
        folds = []
        for dev, ev in ((0, 1), (1, 0)):
            tr_ss, tr_ds = trials(halves[dev], mode, minp, 12000 if mode == "pair" else None)
            ev_ss, ev_ds = trials(halves[ev], mode, minp, 12000 if mode == "pair" else None)
            if not ev_ss or not tr_ss: continue
            vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True).fit(
                [t for a in halves[dev] for t in by[a]])
            cal = LogisticCalibrator().fit(score(vec, tr_ss), score(vec, tr_ds))
            s_ss, s_ds = score(vec, ev_ss), score(vec, ev_ds)
            folds.append((report(s_ss, s_ds, cal), cal(s_ss), cal(s_ds)))
        r = {k: float(np.mean([f[0][k] for f in folds])) for k in ("eer", "cllr", "cllr_min", "misleading_ss", "misleading_ds")}
        r["n_ss"] = int(sum(f[0]["n_ss"] for f in folds)); r["n_ds"] = int(sum(f[0]["n_ds"] for f in folds))
        table5[name] = r
        print(f"  {name:22s} {r['n_ss']}/{r['n_ds']}  EER {r['eer']:.3f}  Cllr {r['cllr']:.3f}  Cllr_min {r['cllr_min']:.3f}", flush=True)
        if name == "Profile, ≥3 poems":
            plot = (np.concatenate([f[1] for f in folds]), np.concatenate([f[2] for f in folds]), r)
    res["table5"] = table5
    if make_plot and plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 3)); tippett(plot[0], plot[1], ax)
        ax.set_title(f"EER {plot[2]['eer']:.1%}  Cllr {plot[2]['cllr']:.2f}")
        fig.tight_layout(); fig.savefig(f"{OUT}/e1_tippett.png", dpi=150)


if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    poems = load_poems(PATH)
    lens = [len(p["ws"].split()) for p in poems]
    res = dict(n_poems=len(poems), n_authors=len({p["author"] for p in poems}), median_words=float(np.median(lens)))
    print("poems", res, flush=True)
    print("closed-set:", flush=True); closed_set(poems, res)
    print("verification:", flush=True); verification(poems, res)
    json.dump(res, open(f"{OUT}/e1.json", "w"), indent=1, ensure_ascii=False)
