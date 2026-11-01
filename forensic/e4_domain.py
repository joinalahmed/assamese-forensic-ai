"""E4 test: is the genre gap closable with target-genre data, and how much, and does it need labels?

Round 3 showed no feature view is genre-robust in both directions. Here the question is operational: an
investigator has text from the platform the evidence came from. Three regimes, each from a detector trained on
the *source* genre (machine + source-genre humans) and tested on the *target* genre's humans + held-out machine:
  A  labelled target-genre humans, k ∈ {0, 25, 50, 100, 250, 500, 1000, 2500}:
       A1 retrain with them added (weighted up to match the source humans in total weight);
       A2 recalibrate only — keep the detector, refit the 2-parameter logistic calibration on k target humans + k machine.
  B  unlabelled target-genre text (realistic: no one labels the seized messages), n ∈ {500, 2000, 5000}:
       B1 self-training: pseudo-label the target text with the source detector, keep confident predictions, retrain;
       B2 importance weighting: a genre classifier (source-human vs target-unlabelled) reweights source training
          sentences towards those that look like the target genre.
Metric of interest is "flagged" (share of target-genre human sentences called machine) with accuracy and AUC.
Test sentences are disjoint from any target-genre sentences used for adaptation.
"""
import os, sys, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import e4_machine_text as B
from e4_machine_text_plus import normalise_script

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0
KS = [0, 25, 50, 100, 250, 500, 1000, 2500]; NS = [500, 2000, 5000]


def vec(texts): return TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=3, max_features=200000).fit(texts)


def fit(X, y, w=None): return LogisticRegression(max_iter=3000, C=1.0).fit(X, y, sample_weight=w)


def metrics(s, nm):
    y = np.r_[np.ones(nm), np.zeros(len(s) - nm)]; return dict(acc=float(((s > 0) == y).mean()), auc=float(roc_auc_score(y, s)), flagged=float((s[nm:] > 0).mean()))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    B.D = "../data/text"; m, w, c = B.machine(), B.wikipedia(), B.cc100()
    n = min(len(m), len(w), len(c)); m = list(rng.permutation(m)[:n])
    H = {"wiki": [normalise_script(t) for t in B.length_match(w, m, rng)], "cc": [normalise_script(t) for t in B.length_match(c, m, rng)]}
    m = [normalise_script(t) for t in m]
    m_tr, m_rest = train_test_split(m, test_size=0.5, random_state=SEED); m_adapt, m_te = train_test_split(m_rest, test_size=0.6, random_state=SEED)
    res = {}
    for src, dst in (("wiki", "cc"), ("cc", "wiki")):
        h_src = H[src]; h_adapt, h_te = train_test_split(H[dst], test_size=0.5, random_state=SEED)  # adaptation pool disjoint from test
        test = m_te + h_te; nm = len(m_te)
        v = vec(m_tr + h_src); Xs = normalize(v.transform(m_tr + h_src)); ys = np.r_[np.ones(len(m_tr)), np.zeros(len(h_src))]
        base = fit(Xs, ys); s0 = base.decision_function(normalize(v.transform(test)))
        R = {"source only": metrics(s0, nm)}; print(f"\n{src} → {dst}  source only: {R['source only']}", flush=True)
        for k in KS[1:]:
            ht = h_adapt[:k]; mt = m_adapt[:k]
            # A1 retrain: target humans weighted so that k of them carry the same total weight as the source humans
            Xa = normalize(v.transform(m_tr + h_src + ht)); ya = np.r_[ys, np.zeros(k)]; wa = np.r_[np.ones(len(ys)), np.full(k, len(h_src) / k)]
            s1 = fit(Xa, ya, wa).decision_function(normalize(v.transform(test))); R[f"A1 retrain, k={k}"] = metrics(s1, nm)
            # A2 recalibrate: 2-parameter logistic on k target humans + k machine sentences, applied to source detector scores
            sd = base.decision_function(normalize(v.transform(mt + ht))); cal = LogisticRegression(C=1e6).fit(sd.reshape(-1, 1), np.r_[np.ones(k), np.zeros(k)])
            s2 = cal.decision_function(s0.reshape(-1, 1)); R[f"A2 recalibrate, k={k}"] = metrics(s2, nm)
            print(f"  k={k:5d}  retrain: flagged {R[f'A1 retrain, k={k}']['flagged']:.3f} acc {R[f'A1 retrain, k={k}']['acc']:.3f} | recalibrate: flagged {R[f'A2 recalibrate, k={k}']['flagged']:.3f} acc {R[f'A2 recalibrate, k={k}']['acc']:.3f}", flush=True)
        for nn in NS:
            # unlabelled target text = a mix the investigator would actually hold: target-genre humans plus some machine text, unlabelled
            U = h_adapt[:nn] + m_adapt[: nn // 4]; Xu = normalize(v.transform(U)); su = base.decision_function(Xu)
            # B1 self-training, two rounds, keep the 60% most confident pseudo-labels each round
            clf = base
            for _ in range(2):
                su = clf.decision_function(Xu); conf = np.abs(su) >= np.quantile(np.abs(su), 0.4)
                from scipy.sparse import vstack
                clf = fit(vstack([Xs, Xu[conf]]), np.r_[ys, (su[conf] > 0).astype(float)])
            R[f"B1 self-training, n={nn}"] = metrics(clf.decision_function(normalize(v.transform(test))), nm)
            # B2 importance weighting: P(target | x) / P(source | x) from a genre classifier over human-looking text
            g = LogisticRegression(max_iter=2000, C=1.0).fit(vstack([normalize(v.transform(h_src)), Xu]), np.r_[np.zeros(len(h_src)), np.ones(len(U))])
            p = g.predict_proba(Xs)[:, 1]; iw = np.clip(p / (1 - p + 1e-9), 0.05, 20); iw[: len(m_tr)] = 1.0; iw[len(m_tr):] *= len(h_src) / iw[len(m_tr):].sum()
            R[f"B2 importance weighting, n={nn}"] = metrics(fit(Xs, ys, iw).decision_function(normalize(v.transform(test))), nm)
            print(f"  n={nn:5d}  self-training: {R[f'B1 self-training, n={nn}']} | importance weighting: {R[f'B2 importance weighting, n={nn}']}", flush=True)
        res[f"train {src} → test {dst}"] = R
    json.dump(res, open(f"{OUT}/e4_domain.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a, (name, R) in zip(ax, res.items()):
        a.plot([0] + KS[1:], [R["source only"]["flagged"]] + [R[f"A1 retrain, k={k}"]["flagged"] for k in KS[1:]], "o-", label="A1 retrain with k labelled target humans")
        a.plot([0] + KS[1:], [R["source only"]["flagged"]] + [R[f"A2 recalibrate, k={k}"]["flagged"] for k in KS[1:]], "s-", label="A2 recalibrate only")
        for nn in NS: a.axhline(R[f"B1 self-training, n={nn}"]["flagged"], ls=":", color="grey")
        a.set_xscale("symlog", linthresh=25); a.set_xlabel("labelled target-genre human sentences k"); a.set_ylabel("target humans flagged as machine"); a.set_title(name); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/e4_domain.png", dpi=150)
