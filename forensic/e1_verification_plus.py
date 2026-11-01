"""E1+: stronger open-set authorship verification (profile protocol, authors with >=3 poems).

Same author-disjoint two-fold protocol as e1_authorship.verification, three upgrades the paper names
as the way to close the gap (Sec. 5.2): fused feature streams, impostor-normalised scoring (S-norm with
a development-author cohort) and author-specific models (one-vs-impostors LinearSVC per candidate).
"""
import sys, os, json
from collections import defaultdict
import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from metrics import LogisticCalibrator, report, tippett
from e1_authorship import load_poems

PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/poems/arambhani.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED, MINP = 0, 3


class Fused:
    """char 1-4 + word unigram + punctuation/layout TF-IDF, each block L2-normalised then stacked."""

    def __init__(self, blocks=("char", "word", "punct")):
        self.blocks = blocks

    def fit(self, texts):
        self.v = {}
        if "char" in self.blocks: self.v["char"] = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True).fit(texts)
        if "word" in self.blocks: self.v["word"] = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True).fit(texts)
        if "punct" in self.blocks:
            # keep only punctuation, digits, line breaks and spacing; everything else becomes a placeholder
            self.v["punct"] = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, preprocessor=self.punct).fit(texts)
        return self

    @staticmethod
    def punct(t):
        return "".join(c if (not c.isalnum() or c.isdigit()) else "x" for c in t)

    def transform(self, texts):
        return hstack([normalize(v.transform(texts)) for v in self.v.values()]).tocsr()


def main():
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    poems = load_poems(PATH)
    by = defaultdict(list)
    for p in poems: by[p["author"]].append(p["text"])  # original layout kept (line breaks carry signal)
    authors = sorted(a for a, v in by.items() if len(v) >= 2); rng.shuffle(authors)
    halves = [authors[::2], authors[1::2]]

    systems = {
        "cosine, char only (paper)": dict(blocks=("char",), norm=None, model="cos"),
        "cosine, fused": dict(blocks=("char", "word", "punct"), norm=None, model="cos"),
        "cosine, fused, S-norm": dict(blocks=("char", "word", "punct"), norm="s", model="cos"),
        "author SVM vs impostors, fused": dict(blocks=("char", "word", "punct"), norm=None, model="svm"),
        "author SVM, fused, S-norm": dict(blocks=("char", "word", "punct"), norm="s", model="svm"),
        "fusion: cosine S-norm + SVM S-norm": dict(blocks=("char", "word", "punct"), norm="s", model="both"),
    }
    results, plots = {}, {}
    for name, cfg in systems.items():
        folds = []
        for dev, ev in ((0, 1), (1, 0)):
            dev_a = [a for a in halves[dev] if len(by[a]) >= MINP]; ev_a = [a for a in halves[ev] if len(by[a]) >= MINP]
            dev_txt = [t for a in halves[dev] for t in by[a]]
            fx = Fused(cfg["blocks"]).fit(dev_txt)
            X = {a: fx.transform(by[a]) for a in dev_a + ev_a}

            def scores(auth_set, pool_authors):
                """Profile trials among auth_set -> {system: (ss, ds)} raw scores. SVMs: one per candidate author
                (all known poems vs development impostors) for different-author trials; leave-one-poem-out models
                for same-author trials, so the questioned poem is never in its own candidate's training set."""
                # impostor / cohort pool must be disjoint from the authors being scored, on the dev side as on the eval side
                Xdev_all = vstack([X[a] for a in pool_authors]); coh_prof = normalize(np.vstack([np.asarray(X[a].mean(0)) for a in pool_authors]))
                ydev = np.zeros(Xdev_all.shape[0]); use_svm = cfg["model"] in ("svm", "both")

                def fit_svm(K): return LinearSVC(C=1.0, class_weight="balanced").fit(vstack([K, Xdev_all]), np.r_[np.ones(K.shape[0]), ydev])

                def norm_cos(s, prof, q):
                    if cfg["norm"] != "s": return s
                    z = (Xdev_all @ prof.T).toarray().ravel() if hasattr(Xdev_all @ prof.T, "toarray") else np.asarray(Xdev_all @ prof.T).ravel(); t = np.asarray(q @ coh_prof.T).ravel()
                    return 0.5 * ((s - z.mean()) / (z.std() + 1e-9) + (s - t.mean()) / (t.std() + 1e-9))

                def norm_svm(s, clf):
                    if cfg["norm"] != "s": return s
                    zs = clf.decision_function(Xdev_all); return (s - zs.mean()) / (zs.std() + 1e-9)

                full_prof = {a: normalize(np.asarray(X[a].mean(0))) for a in auth_set}
                full_svm = {a: fit_svm(X[a]) for a in auth_set} if use_svm else {}
                out = {k: ([], []) for k in ("cos", "svm")}
                for a in auth_set:
                    n = X[a].shape[0]
                    for i in range(n):
                        q = X[a][i]; K = X[a][[j for j in range(n) if j != i]]
                        prof = normalize(np.asarray(K.mean(0))); out["cos"][0].append(norm_cos(float(q @ prof.T), prof, q))
                        if use_svm: clf = fit_svm(K); out["svm"][0].append(norm_svm(float(clf.decision_function(q)[0]), clf))
                        for b in auth_set:
                            if b == a: continue
                            out["cos"][1].append(norm_cos(float(q @ full_prof[b].T), full_prof[b], q))
                            if use_svm: out["svm"][1].append(norm_svm(float(full_svm[b].decision_function(q)[0]), full_svm[b]))
                return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items() if v[0]}

            # dev-side trials: split the development authors in two, each half scored against the other as impostor pool
            d1, d2 = dev_a[::2], dev_a[1::2]; sa, sb = scores(d1, d2), scores(d2, d1)
            d = {k: (np.r_[sa[k][0], sb[k][0]], np.r_[sa[k][1], sb[k][1]]) for k in sa}
            e = scores(ev_a, dev_a)
            if cfg["model"] == "both":
                # equal-weight fusion of standardised scores (standardisation fitted on dev trials)
                def fuse(S, ref):
                    return sum((S[k][0] - np.r_[ref[k][0], ref[k][1]].mean()) / np.r_[ref[k][0], ref[k][1]].std() for k in ("cos", "svm")), \
                           sum((S[k][1] - np.r_[ref[k][0], ref[k][1]].mean()) / np.r_[ref[k][0], ref[k][1]].std() for k in ("cos", "svm"))
                d_ss, d_ds = fuse(d, d); e_ss, e_ds = fuse(e, d)
            else:
                k = "svm" if cfg["model"] == "svm" else "cos"; d_ss, d_ds = d[k]; e_ss, e_ds = e[k]
            cal = LogisticCalibrator().fit(d_ss, d_ds)
            folds.append((report(e_ss, e_ds, cal), cal(e_ss), cal(e_ds)))
        r = {k: float(np.mean([f[0][k] for f in folds])) for k in ("eer", "cllr", "cllr_min", "misleading_ss", "misleading_ds")}
        r["n_ss"] = int(sum(f[0]["n_ss"] for f in folds)); r["n_ds"] = int(sum(f[0]["n_ds"] for f in folds))
        results[name] = r; plots[name] = (np.concatenate([f[1] for f in folds]), np.concatenate([f[2] for f in folds]))
        print(f"  {name:38s} {r['n_ss']}/{r['n_ds']}  EER {r['eer']:.3f}  Cllr {r['cllr']:.3f}  Cllr_min {r['cllr_min']:.3f}", flush=True)
    json.dump(results, open(f"{OUT}/e1_plus.json", "w"), indent=1, ensure_ascii=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    best = min(results, key=lambda k: results[k]["eer"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    tippett(*plots["cosine, char only (paper)"], ax[0]); ax[0].set_title(f"baseline  EER {results['cosine, char only (paper)']['eer']:.1%}")
    tippett(*plots[best], ax[1]); ax[1].set_title(f"{best}\nEER {results[best]['eer']:.1%}  Cllr {results[best]['cllr']:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/e1_plus_tippett.png", dpi=150)


if __name__ == "__main__":
    main()
