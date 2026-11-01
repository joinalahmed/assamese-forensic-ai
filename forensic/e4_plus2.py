"""E4 round 3: content-independent features for genre-robust machine-text detection.

The round-2 result was that lexical detectors learn genre. Here the hypothesis is that *content-free* signals
transfer better: (a) the char-LM perplexity pair alone (3 dims); (b) a "style-only" view of the sentence in
which every Assamese letter is replaced by a class symbol (consonant / vowel sign / virama / punctuation /
digit / Latin / space), so the model sees only punctuation, length and script-structure habits; (c) a
strongly regularised char model (C=0.05); (d) fusions of these with the script-normalised char model.
Also an **ensemble by rank-averaging** of models trained on each view. All evaluated on the unseen genre.
"""
import sys, os, json, unicodedata
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, StandardScaler
from scipy.stats import rankdata
import e4_machine_text as B
from e4_machine_text_plus import normalise_script, lm_features

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0


def style_view(t):
    out = []
    for ch in t:
        o = ord(ch)
        if ch.isspace(): out.append(" ")
        elif ch == "্": out.append("H")                       # virama (conjunct former)
        elif 0x09BE <= o <= 0x09CC or ch == "ঁ" or ch == "ং" or ch == "ঃ": out.append("V")  # vowel signs / nasal marks
        elif 0x0985 <= o <= 0x0994: out.append("I")            # independent vowels
        elif 0x0995 <= o <= 0x09DF: out.append("C")            # consonants
        elif 0x09E6 <= o <= 0x09EF or ch.isdigit(): out.append("9")
        elif ch.isascii() and ch.isalpha(): out.append("L")
        else: out.append(ch)                                   # punctuation incl. । ।। , - ' " ( ) : ; ? !
    return "".join(out)


def views(texts, name):
    if name == "char": return [normalise_script(t) for t in texts]
    if name == "style": return [style_view(t) for t in texts]
    raise ValueError(name)


def fit_score(train_m, train_h, test_txt, view, C):
    tr = views(train_m + train_h, view); te = views(test_txt, view); y = np.r_[np.ones(len(train_m)), np.zeros(len(train_h))]
    ng = (1, 4) if view == "char" else (2, 6)
    v = TfidfVectorizer(analyzer="char", ngram_range=ng, sublinear_tf=True, min_df=3, max_features=200000).fit(tr)
    clf = LogisticRegression(max_iter=3000, C=C).fit(normalize(v.transform(tr)), y); return clf.decision_function(normalize(v.transform(te)))


def lm_score(train_m, train_h, test_txt):
    m_lm, m_rest = train_test_split(train_m, test_size=0.5, random_state=SEED); h_lm, h_rest = train_test_split(train_h, test_size=0.5, random_state=SEED)
    f = lambda T: lm_features([normalise_script(t) for t in m_lm], [normalise_script(t) for t in h_lm], [normalise_script(t) for t in T])
    Xtr = f(m_rest + h_rest); sc = StandardScaler().fit(Xtr); y = np.r_[np.ones(len(m_rest)), np.zeros(len(h_rest))]
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), y); return clf.decision_function(sc.transform(f(test_txt)))


def metrics(s, nm):
    y = np.r_[np.ones(nm), np.zeros(len(s) - nm)]
    return dict(acc=float(((s > 0) == y).mean()), auc=float(roc_auc_score(y, s)), flagged=float((s[nm:] > 0).mean()))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    B.D = "../data/text"; m, w, c = B.machine(), B.wikipedia(), B.cc100()
    n = min(len(m), len(w), len(c)); m = list(rng.permutation(m)[:n]); H = {"wiki": B.length_match(w, m, rng), "cc": B.length_match(c, m, rng)}
    mtr, mte = train_test_split(m, test_size=0.3, random_state=SEED)
    res = {}
    for src, dst in (("wiki", "cc"), ("cc", "wiki")):
        test = mte + H[dst]; S = {}
        S["char (script-norm) C=1"] = fit_score(mtr, H[src], test, "char", 1.0)
        S["char (script-norm) C=0.05"] = fit_score(mtr, H[src], test, "char", 0.05)
        S["style-only view C=1"] = fit_score(mtr, H[src], test, "style", 1.0)
        S["char-LM perplexity only"] = lm_score(mtr, H[src], test)
        z = lambda s: (s - s.mean()) / s.std()
        S["fusion z: char + style"] = z(S["char (script-norm) C=1"]) + z(S["style-only view C=1"])
        S["fusion z: char + style + LM"] = S["fusion z: char + style"] + z(S["char-LM perplexity only"])
        S["rank-ensemble: char + style + LM"] = sum(rankdata(S[k]) / len(test) - 0.5 for k in ("char (script-norm) C=1", "style-only view C=1", "char-LM perplexity only"))
        for k, s in S.items():
            # fusions have no natural zero: threshold at the training-prior-free midpoint of the two class medians on *source* dev data is
            # unavailable here, so we report accuracy at threshold 0 of the standardised score (balanced test set) and AUC/flagged.
            r = metrics(s, len(mte)); res[f"train {src} → test {dst} | {k}"] = r
            print(f"  train {src} → test {dst} | {k:36s} acc {r['acc']:.3f}  AUC {r['auc']:.3f}  flagged {r['flagged']:.3f}", flush=True)
    json.dump(res, open(f"{OUT}/e4_plus2.json", "w"), indent=1, ensure_ascii=False)
