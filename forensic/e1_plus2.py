"""E1 round 3: (a) closed-set attribution with the fused feature streams and calibrated probabilities;
(b) the forensically hardest protocol, *pairwise* verification (one questioned poem vs one known poem),
with fused features and S-norm, which e1_verification_plus.py only applied to the profile protocol.
Same splits/seed as e1_authorship.py.
"""
import sys, os, json
from collections import defaultdict
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from metrics import LogisticCalibrator, report
from e1_authorship import load_poems, feats_fit
from e1_verification_plus import Fused

PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/poems/arambhani.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0


def closed_set(poems):
    cnt = defaultdict(int)
    for p in poems: cnt[p["author"]] += 1
    P = [p for p in poems if cnt[p["author"]] >= 10]; y = np.array([p["author"] for p in P]); X = [p["text"] for p in P]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED); out = {}
    configs = {
        "char1-4+word / LinearSVC (baseline best)": (lambda tr: feats_fit(tr, "char14+word"), lambda: LinearSVC()),
        "fused char+word+punct / LinearSVC": (lambda tr: Fused().fit(tr).transform, lambda: LinearSVC()),
        "fused / LinearSVC C=0.3": (lambda tr: Fused().fit(tr).transform, lambda: LinearSVC(C=0.3)),
        "fused / LinearSVC C=3": (lambda tr: Fused().fit(tr).transform, lambda: LinearSVC(C=3)),
        "fused / LogReg C=50": (lambda tr: Fused().fit(tr).transform, lambda: LogisticRegression(C=50, max_iter=3000)),
        "fused, original layout + ws-normalised copy / LinearSVC": (None, lambda: LinearSVC()),
    }
    for name, (ff, mk) in configs.items():
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            trt, tet = [X[i] for i in tr], [X[i] for i in te]
            if ff is None:  # augmentation: train on both the original layout and a whitespace-normalised copy
                aug = trt + [" ".join(t.split()) for t in trt]; ya = np.r_[y[tr], y[tr]]; f = Fused().fit(aug).transform
                pred = mk().fit(f(aug), ya).predict(f(tet))
            else:
                f = ff(trt); pred = mk().fit(f(trt), y[tr]).predict(f(tet))
            accs.append((pred == y[te]).mean()); f1s.append(f1_score(y[te], pred, average="macro"))
        out[name] = dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), macro_f1=float(np.mean(f1s)))
        print(f"  {name:58s} {np.mean(accs):.3f} ± {np.std(accs):.3f}  F1 {np.mean(f1s):.3f}", flush=True)
    return out


def pairwise(poems):
    rng = np.random.RandomState(SEED); by = defaultdict(list)
    for p in poems: by[p["author"]].append(p["text"])
    authors = sorted(a for a, v in by.items() if len(v) >= 2); rng.shuffle(authors); halves = [authors[::2], authors[1::2]]
    out = {}
    for name, blocks, sn in (("pairwise, char (paper)", ("char",), False), ("pairwise, fused", ("char", "word", "punct"), False), ("pairwise, fused, S-norm", ("char", "word", "punct"), True)):
        folds = []
        for dev, ev in ((0, 1), (1, 0)):
            fx = Fused(blocks).fit([t for a in halves[dev] for t in by[a]]); X = {a: normalize(fx.transform(by[a])) for a in authors}

            def scores(auth, pool):
                from scipy.sparse import vstack as svstack
                allp = [(a, i) for a in auth for i in range(X[a].shape[0])]; Xpool = svstack([X[a] for a in pool]).tocsr()
                def sc(a, i, b, j):
                    s = (X[a][i] @ X[b][j].T)[0, 0]
                    if not sn: return s
                    z = (X[a][i] @ Xpool.T).toarray().ravel(); t = (X[b][j] @ Xpool.T).toarray().ravel()
                    return 0.5 * ((s - z.mean()) / (z.std() + 1e-9) + (s - t.mean()) / (t.std() + 1e-9))
                ss = [sc(a, i, a, j) for a in auth for i in range(X[a].shape[0]) for j in range(i + 1, X[a].shape[0])]
                ds = []
                while len(ds) < 12000:
                    (a, i), (b, j) = [allp[k] for k in rng.choice(len(allp), 2, replace=False)]
                    if a != b: ds.append(sc(a, i, b, j))
                return np.array(ss), np.array(ds)

            d1, d2 = halves[dev][::2], halves[dev][1::2]; sa, sb = scores(d1, d2), scores(d2, d1)
            d_ss, d_ds = np.r_[sa[0], sb[0]], np.r_[sa[1], sb[1]]; e_ss, e_ds = scores(halves[ev], halves[dev])
            cal = LogisticCalibrator().fit(d_ss, d_ds); folds.append(report(e_ss, e_ds, cal))
        r = {k: float(np.mean([f[k] for f in folds])) for k in ("eer", "cllr", "cllr_min")}; r["n_ss"] = int(sum(f["n_ss"] for f in folds)); r["n_ds"] = int(sum(f["n_ds"] for f in folds))
        out[name] = r; print(f"  {name:30s} {r['n_ss']}/{r['n_ds']}  EER {r['eer']:.3f}  Cllr {r['cllr']:.3f}  Cllr_min {r['cllr_min']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); poems = load_poems(PATH)
    cs = json.load(open(f"{OUT}/e1_plus2.json"))["closed_set"] if os.environ.get("SKIP_CLOSED") and os.path.exists(f"{OUT}/e1_plus2.json") else closed_set(poems)
    print("pairwise verification:", flush=True); pw = pairwise(poems)
    json.dump(dict(closed_set=cs, pairwise=pw), open(f"{OUT}/e1_plus2.json", "w"), indent=1, ensure_ascii=False)
