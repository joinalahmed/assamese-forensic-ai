"""E4: machine-generated Assamese text vs human text (in-genre, cross-genre, calibrated).

Machine: ChatGPT-generated sentences (indian-nlp/assamese-dataset output_100k_split_1.jsonl).
Human: Assamese Wikipedia sentences and CC-100, length-matched to the machine sentences.
Paper reference (Table 9): in-genre 97.0% / 97.7%; cross-genre flags 33.6% / 55.7% of human text;
calibrated mixed-genre EER 3.3%, Cllr 0.13.
"""
import sys, json, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from metrics import LogisticCalibrator, report, tippett

D = sys.argv[1] if len(sys.argv) > 1 else "../data/text"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0
MIN_C, MAX_C = 20, 400


def machine():
    s = [json.loads(l)["sentence"].strip() for l in open(f"{D}/chatgpt.jsonl", encoding="utf-8") if l.strip()]
    return list(dict.fromkeys(x for x in s if x))


def wikipedia():
    out = []
    for l in open(f"{D}/wiki100k_sentences.txt", encoding="utf-8"):
        parts = l.rstrip("\n").split("\t", 1)
        out.append(parts[-1].strip())
    return list(dict.fromkeys(x for x in out if MIN_C <= len(x) <= MAX_C))


def cc100():
    out = []
    for l in open(f"{D}/cc100_head.txt", encoding="utf-8", errors="ignore"):
        for s in re.split(r"(?<=।)\s+", l.strip()):
            if MIN_C <= len(s) <= MAX_C: out.append(s)
    return list(dict.fromkeys(out))


def length_match(pool, targets, rng, ratio=1):
    """For each target length pick `ratio` unused pool sentences with the closest character length."""
    pool = sorted(pool, key=len); lens = np.array([len(p) for p in pool]); used = np.zeros(len(pool), bool); out = []
    for t in rng.permutation(len(targets)):
        L = len(targets[t]); i = np.searchsorted(lens, L)
        for _ in range(ratio):
            lo, hi = i - 1, i
            while True:
                cand = [j for j in (lo, hi) if 0 <= j < len(pool) and not used[j]]
                if cand:
                    j = min(cand, key=lambda j: abs(lens[j] - L)); used[j] = True; out.append(pool[j]); break
                if lo < 0 and hi >= len(pool): break
                lo, hi = lo - 1, hi + 1
    return out


def vec():
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=3, max_features=200000)


def clf():
    return LogisticRegression(max_iter=2000, C=10)


def cv_in_genre(m, h):
    X = m + h; y = np.r_[np.ones(len(m)), np.zeros(len(h))]; accs, aucs = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        v = vec().fit([X[i] for i in tr]); c = clf().fit(v.transform([X[i] for i in tr]), y[tr])
        Xt = v.transform([X[i] for i in te]); accs.append(c.score(Xt, y[te])); aucs.append(roc_auc_score(y[te], c.decision_function(Xt)))
    return float(np.mean(accs)), float(np.mean(aucs))


def cross_genre(m, h_tr, h_te):
    mtr, mte = train_test_split(m, test_size=0.3, random_state=SEED)
    X = mtr + h_tr; y = np.r_[np.ones(len(mtr)), np.zeros(len(h_tr))]
    v = vec().fit(X); c = clf().fit(v.transform(X), y)
    Xt = mte + h_te; yt = np.r_[np.ones(len(mte)), np.zeros(len(h_te))]; s = c.decision_function(v.transform(Xt))
    return dict(acc=float(((s > 0) == yt).mean()), auc=float(roc_auc_score(yt, s)), flagged=float((s[len(mte):] > 0).mean()))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    m, w, c = machine(), wikipedia(), cc100()
    print(f"machine {len(m)}  wikipedia {len(w)}  cc100 {len(c)}", flush=True)
    # equal numbers of human sentences per genre, length-matched to the machine sentences
    n = min(len(m), len(w), len(c)); m = list(rng.permutation(m)[:n])
    hw, hc = length_match(w, m, rng), length_match(c, m, rng)
    print(f"using {len(m)} machine, {len(hw)} wiki, {len(hc)} cc100 sentences (length matched)", flush=True)
    res = dict(n_machine=len(m), n_wiki=len(hw), n_cc=len(hc))
    res["in_genre_wiki"] = dict(zip(("acc", "auc"), cv_in_genre(m, hw)))
    res["in_genre_cc"] = dict(zip(("acc", "auc"), cv_in_genre(m, hc)))
    res["train_wiki_test_cc"] = cross_genre(m, hw, hc)
    res["train_cc_test_wiki"] = cross_genre(m, hc, hw)
    for k in ("in_genre_wiki", "in_genre_cc", "train_wiki_test_cc", "train_cc_test_wiki"): print(k, res[k], flush=True)

    # calibrated, mixed-genre humans: train / calibration / test, disjoint sentences
    H = hw + hc
    mtr, mrest = train_test_split(m, test_size=0.5, random_state=SEED); mdev, mte = train_test_split(mrest, test_size=0.5, random_state=SEED)
    htr, hrest = train_test_split(H, test_size=0.5, random_state=SEED); hdev, hte = train_test_split(hrest, test_size=0.5, random_state=SEED)
    v = vec().fit(mtr + htr); c1 = clf().fit(v.transform(mtr + htr), np.r_[np.ones(len(mtr)), np.zeros(len(htr))])
    sc = lambda t: c1.decision_function(v.transform(t))
    cal = LogisticCalibrator().fit(sc(mdev), sc(hdev))
    s_m, s_h = sc(mte), sc(hte); res["calibrated"] = report(s_m, s_h, cal); print("calibrated", res["calibrated"], flush=True)
    json.dump(res, open(f"{OUT}/e4.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    labs = ["in genre\nWiki", "in genre\nCC-100", "wiki→CC", "CC→wiki"]
    ax[0].bar(labs, [res["in_genre_wiki"]["acc"] * 100, res["in_genre_cc"]["acc"] * 100, res["train_wiki_test_cc"]["acc"] * 100, res["train_cc_test_wiki"]["acc"] * 100])
    ax[0].set_ylabel("accuracy (%)")
    ax[1].bar(["wiki→CC", "CC→wiki"], [res["train_wiki_test_cc"]["flagged"] * 100, res["train_cc_test_wiki"]["flagged"] * 100]); ax[1].set_ylabel("human flagged as machine (%)")
    tippett(cal(s_m), cal(s_h), ax[2], "machine-written", "human-written")
    fig.tight_layout(); fig.savefig(f"{OUT}/e4.png", dpi=150)
    # most indicative character n-grams (in-genre detector, human = Wikipedia)
    v2 = vec().fit(m + hw); c2 = clf().fit(v2.transform(m + hw), np.r_[np.ones(len(m)), np.zeros(len(hw))])
    names = np.array(v2.get_feature_names_out()); o = np.argsort(c2.coef_[0])
    res["top_machine"] = [(names[i], float(c2.coef_[0][i])) for i in o[::-1][:8]]
    res["top_human"] = [(names[i], float(c2.coef_[0][i])) for i in o[:8]]
    json.dump(res, open(f"{OUT}/e4.json", "w"), indent=1, ensure_ascii=False); print(res["top_machine"], res["top_human"])
