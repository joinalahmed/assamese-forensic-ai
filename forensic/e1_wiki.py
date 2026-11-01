"""E1 on Assamese Wikipedia editors: open-set authorship verification with many authors and real sessions.

Items are editor *sessions* (text the editor added within one sitting, see wiki_authors.py). Same-author trials
compare a questioned session with a profile built from the editor's *other* sessions, so every same-author trial
is a cross-occasion comparison; different-author trials pair the questioned session with other editors' profiles.
Authors are split into disjoint halves: TF-IDF, impostor pool, cohort and calibration come from one half,
evaluation from the other, folds swapped. Systems: fused cosine, fused cosine + S-norm, author SVM vs impostors.
Prolific editors are capped at MAX_SESS random sessions so no editor dominates the trials. Conditions: (a) questioned session truncated to the first 75 words (poem-length, comparable to the paper's E1);
(b) full sessions. Register: all namespaces, and talk/user namespaces only (the informal slice).
"""
import sys, os, json
from collections import defaultdict
import numpy as np
from scipy.sparse import vstack
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from metrics import LogisticCalibrator, report, tippett
from e1_verification_plus import Fused

SRC = sys.argv[1] if len(sys.argv) > 1 else "../data/new/aswiki/editor_sessions.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED, MIN_SESS, MIN_WORDS, MAX_AUTHORS, MAX_SESS, Q_WORDS = 0, 3, 40, 150, 12, 75
TALK_NS = {"1", "3", "4", "5", "2"}


def load(talk_only):
    by = defaultdict(list)
    for l in open(SRC, encoding="utf-8"):
        r = json.loads(l)
        if talk_only and not set(r["namespaces"]) & TALK_NS: continue
        if len(r["text"].split()) >= MIN_WORDS: by[r["user"]].append(r["text"])
    return {a: v for a, v in by.items() if len(v) >= MIN_SESS}


def run(by, truncate, rng):
    authors = sorted(by); rng.shuffle(authors); authors = authors[:MAX_AUTHORS]; halves = [authors[::2], authors[1::2]]
    by = {a: [by[a][i] for i in rng.choice(len(by[a]), min(len(by[a]), MAX_SESS), replace=False)] for a in authors}  # cap prolific editors
    q = (lambda t: " ".join(t.split()[:Q_WORDS])) if truncate else (lambda t: t)
    out = {}
    for sysname in ("cosine fused", "cosine fused + S-norm", "author SVM vs impostors"):
        folds = []
        for dev, ev in ((0, 1), (1, 0)):
            fx = Fused().fit([t for a in halves[dev] for t in by[a]])
            X = {a: fx.transform(by[a]) for a in authors}; Xq = {a: fx.transform([q(t) for t in by[a]]) for a in authors}

            def scores(auth, pool):
                P = vstack([X[a] for a in pool]).tocsr(); cohT = np.ascontiguousarray(normalize(np.vstack([np.asarray(X[a].mean(0)) for a in pool])).T); y0 = np.zeros(P.shape[0])
                prof = {a: normalize(np.asarray(X[a].mean(0))) for a in auth}
                svm = {a: LinearSVC(C=1.0, class_weight="balanced").fit(vstack([X[a], P]), np.r_[np.ones(X[a].shape[0]), y0]) for a in auth} if "SVM" in sysname else {}
                # z-norm statistics depend only on the candidate profile/model, so cache them per author
                zc = {a: (lambda z: (z.mean(), z.std() + 1e-9))(np.asarray(P @ prof[a].T).ravel()) for a in auth} if "S-norm" in sysname else {}
                zs = {a: (lambda z: (z.mean(), z.std() + 1e-9))(svm[a].decision_function(P)) for a in auth} if "SVM" in sysname else {}

                def sc(qv, a, K=None, clf=None, tstat=None):
                    if "SVM" in sysname:
                        m = clf or svm[a]; s = float(m.decision_function(qv)[0]); mu, sd = zs[a] if clf is None else (lambda z: (z.mean(), z.std() + 1e-9))(m.decision_function(P)); return (s - mu) / sd
                    pr = normalize(np.asarray(K.mean(0))) if K is not None else prof[a]; s = float(qv @ pr.T)
                    if "S-norm" not in sysname: return s
                    mu, sd = zc[a] if K is None else (lambda z: (z.mean(), z.std() + 1e-9))(np.asarray(P @ pr.T).ravel())
                    tm, ts = tstat; return 0.5 * ((s - mu) / sd + (s - tm) / ts)
                ss, ds = [], []
                for a in auth:
                    n = X[a].shape[0]
                    for i in range(n):
                        qv = Xq[a][i]; K = X[a][[j for j in range(n) if j != i]]
                        clf = LinearSVC(C=1.0, class_weight="balanced").fit(vstack([K, P]), np.r_[np.ones(K.shape[0]), y0]) if "SVM" in sysname else None
                        t = np.asarray(qv @ cohT).ravel(); tstat = (t.mean(), t.std() + 1e-9)  # t-norm stats depend only on the questioned item
                        ss.append(sc(qv, a, K, clf, tstat)); ds += [sc(qv, b, tstat=tstat) for b in auth if b != a]
                return np.array(ss), np.array(ds)

            d1, d2 = halves[dev][::2], halves[dev][1::2]; sa, sb = scores(d1, d2), scores(d2, d1)
            cal = LogisticCalibrator().fit(np.r_[sa[0], sb[0]], np.r_[sa[1], sb[1]])
            e_ss, e_ds = scores(halves[ev], halves[dev]); folds.append((report(e_ss, e_ds, cal), cal(e_ss), cal(e_ds)))
        r = {k: float(np.mean([f[0][k] for f in folds])) for k in ("eer", "cllr", "cllr_min")}; r["n_ss"] = int(sum(f[0]["n_ss"] for f in folds)); r["n_ds"] = int(sum(f[0]["n_ds"] for f in folds))
        r["_llr"] = (np.concatenate([f[1] for f in folds]), np.concatenate([f[2] for f in folds])); out[sysname] = r
        print(f"    {sysname:26s} {r['n_ss']}/{r['n_ds']}  EER {r['eer']:.3f}  Cllr {r['cllr']:.3f}  Cllr_min {r['cllr_min']:.3f}", flush=True)
    return out, len(authors)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); res = {}; plots = {}
    for talk in (False, True):
        by = load(talk); words = [len(t.split()) for v in by.values() for t in v]
        tag = "talk/user namespaces" if talk else "all namespaces"
        print(f"{tag}: {len(by)} editors with ≥{MIN_SESS} sessions of ≥{MIN_WORDS} words; {len(words)} sessions, median {np.median(words):.0f} words", flush=True)
        res[tag] = dict(n_editors=len(by), n_sessions=len(words), median_words=float(np.median(words)))
        for trunc in (True, False):
            cond = f"questioned {'truncated to 75 words' if trunc else 'full session'}"; print(f"  {cond}", flush=True)
            out, na = run(by, trunc, np.random.RandomState(SEED)); res[tag][cond] = {k: {m: v for m, v in r.items() if m != "_llr"} for k, r in out.items()}; res[tag][cond]["n_authors_used"] = na
            if trunc: plots[tag] = out["cosine fused + S-norm"]["_llr"], out["cosine fused + S-norm"]
    json.dump(res, open(f"{OUT}/e1_wiki.json", "w"), indent=1, ensure_ascii=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, (tag, (ll, r)) in zip(ax, plots.items()): tippett(*ll, a); a.set_title(f"{tag}, 75-word queries\nEER {r['eer']:.1%}  Cllr {r['cllr']:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/e1_wiki_tippett.png", dpi=150)
