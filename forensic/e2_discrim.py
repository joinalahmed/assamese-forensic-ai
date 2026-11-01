"""E2 round 8: discriminative lattice reranker + Aksharantar lexicon (CPU only).

Round 7 left E2 at 84.9% token accuracy with a lattice whose oracle recall is 96.5%: the decoder *selects* badly,
not the candidate generator. Here every lattice edge (token, candidate, previous word) gets a feature vector —
direct-model log P(n|r), char-transformer beam log-prob and rank, copy flag, Aksharantar flag and count, big-LM and
in-domain bigram log-probs, unigram log-probs, length ratio, Latin-script flag, and the round-7 Viterbi
forward–backward posterior / 1-best flag — and a logistic-regression or gradient-boosted classifier trained on the
dev lattices (label = candidate equals the reference) replaces the hand-set emission + λ·LM score. Test sentences
are decoded with Viterbi over the classifier's log-odds. Aksharantar (179k news-register Roman↔native word pairs,
CC0/CC-BY) adds readings to the lattice for every Roman form it knows.

Protocol (unchanged from round 7): 80/20 sentence split, seed 0; dev = first 2,000 training sentences; everything
tuned on dev is fitted with the word-level decoder, in-domain LM and char transformer refitted *without* dev
(`NODEV=1 NFC=1 ... e2_neural.py`); Unicode NFC on the native side everywhere. The test split is identical to every
other E2 script. External test: Bhasha-Abhijnaanam Assamese sentences (news register, different romanisation habits).

Run (from forensic/):
  EPOCHS=8 D=256 LAYERS=2 BEAM=5 NFC=1 NODEV=1 TAG=_big_nodev_nfc ../.venv/bin/python e2_neural.py   # ~40 min CPU
  ../.venv/bin/python e2_discrim.py
"""
import os, sys, json, math, time, pickle, unicodedata, re
from collections import Counter, defaultdict
import numpy as np
os.environ.setdefault("D", "256"); os.environ.setdefault("LAYERS", "2"); os.environ.setdefault("BEAM", "5")
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
import e2_neural as N
import e2_normaliser_plus as V
from e2_romanisation import load, PATH
from e2_rerank import KN, build_big_lm, NATIVE, K_SEEN

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"; SEED, LAM, MU = 0, 0.2, 0.7   # round-7 best bigram config
CKPT = os.environ.get("CKPT", f"{OUT}/e2_translit_big_nodev_nfc.pt"); K_AKSH = 5
AKSH = "../data/new/aksharantar/asm_train.json"; BHASHA = "../data/new/bhasha/bhasha-abhijnaanam.json"
NEG = -12.0  # "feature absent" filler on the log scale
SMOKE = int(os.environ.get("SMOKE", 0))  # >0: tiny run (that many dev/test sentences) to check the pipeline
LM_CACHE = "../data/new/lm_cache"
CF = ["direct_lp", "direct_seen", "direct_logcount", "tok_seen", "tok_logcount", "neural_lp_rel", "neural_top_lp", "neural_rank", "neural_in", "copy", "aksh", "aksh_logcount",
      "latin", "tok_latin_like", "tok_has_digit", "cand_apos", "len_ratio", "log_ncand", "ind_uni", "big_uni", "big_oov"]  # context-free (token, candidate) features
EDGE = ["ind_bi", "big_bi", "lm_interp"]                                                                 # depend on the previous word
FEATS = CF + EDGE + ["posterior", "on_path", "tok_len"]; NCF = len(CF)
nfc = lambda t: unicodedata.normalize("NFC", t)


def load_aksharantar(path):
    """Roman form (lower-cased) -> Counter(native word). Aksharantar asm_train.json is one JSON object per line."""
    tab = defaultdict(Counter)
    if not os.path.exists(path): print(f"!! {path} missing — running without Aksharantar", flush=True); return tab
    with open(path, encoding="utf-8") as f:
        txt = f.read().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    for r in rows: tab[r["english word"].strip().lower()][nfc(r["native word"].strip())] += 1
    print(f"Aksharantar: {len(rows)} pairs, {len(tab)} Roman forms", flush=True); return tab


def build_lm(vocab):
    """Round-7 big LM (IndicCorp v2 + Wikipedia + Sangraha, 113M tokens) when IndicCorp is on disk; otherwise a fallback
    Kneser-Ney bigram from the two corpora get_data.sh fetches from GitHub (Wikipedia 100k sentences + 12 MB CC-100 prefix)."""
    if os.path.exists("../data/new/indiccorp/data/as.txt"): return build_big_lm(vocab) + ("indiccorp+wiki+sangraha (round 7)",)
    os.makedirs(LM_CACHE, exist_ok=True); f = f"{LM_CACHE}/fallback_bigram.pkl"
    if os.path.exists(f): lm, voc = pickle.load(open(f, "rb")); vocab.update(voc); return lm, vocab, "wiki100k+cc100_head (fallback)"
    lm = KN(2); n = 0; t0 = time.time()
    def feed(line):
        nonlocal n
        toks = [nfc(t) for t in line.split() if NATIVE.search(t)]
        if len(toks) >= 2: lm.add([vocab.setdefault(t, len(vocab)) for t in toks]); n += len(toks)
    for line in open("../data/text/wiki100k_sentences.txt", encoding="utf-8", errors="ignore"): feed(line.split("\t", 1)[-1])
    if os.path.exists("../data/text/cc100_head.txt"):
        for line in open("../data/text/cc100_head.txt", encoding="utf-8", errors="ignore"):
            for s in line.split("।"): feed(s)
    lm.finish(); print(f"  fallback LM: {n / 1e6:.1f}M tokens, vocab {len(vocab)} ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump((lm, dict(vocab)), open(f, "wb")); return lm, vocab, "wiki100k+cc100_head (fallback)"


def load_bhasha(path):
    if not os.path.exists(path): print(f"!! {path} missing — skipping the external test", flush=True); return []
    rows = json.load(open(path, encoding="utf-8")); rows = rows if isinstance(rows, list) else rows.get("data", list(rows.values())[0])
    out, skipped = [], 0
    for r in rows:
        if r.get("language") != "Assamese": continue
        rom, nat = r["romanized sentence"].split(), [nfc(x) for x in r["native sentence"].split()]
        if rom and len(rom) == len(nat): out.append((rom, nat))
        else: skipped += 1
    print(f"Bhasha-Abhijnaanam Assamese: {len(out)} token-aligned sentences ({skipped} skipped for length mismatch)", flush=True); return out


class System:
    """Word-level candidate generator + LMs fitted on `train_pairs`; neural beams and Aksharantar are shared."""

    def __init__(self, train_pairs, neural, aksh, big, vocab, use_aksh=True):
        self.dec = V.Decoder().fit(train_pairs); self.neural, self.aksh, self.big, self.vocab, self.use_aksh = neural, aksh, big, vocab, use_aksh
        self.ind = KN(2)
        for _, n in train_pairs: self.ind.add([vocab.setdefault(t, len(vocab)) for t in n])
        self.ind.finish(); self.cache = {}

    def wid(self, w): return -1 if w == V.BOS else self.vocab.get(w, -3)

    def cands(self, tok):
        """{candidate: (baseline emission, context-free feature dict)} for one Roman token (NFC candidates)."""
        if tok in self.cache: return self.cache[tok]
        t = tok.lower(); out = {}; seen = t in self.dec.reads
        def feat(c): return out.setdefault(c, [NEG, dict(direct_lp=NEG, direct_seen=0.0, direct_logcount=0.0, neural_lp_rel=NEG, neural_rank=6.0, neural_in=0.0, aksh=0.0, aksh_logcount=0.0)])
        tot = 0
        if seen:
            c = self.dec.reads[t]; tot = sum(c.values())
            for b, k in c.most_common(K_SEEN): f = feat(b); f[0] = math.log(k / tot); f[1].update(direct_lp=f[0], direct_seen=1.0, direct_logcount=math.log(k))
            f = feat(tok); f[0] = max(f[0], math.log(0.5 / (tot + 1)))
        hyps = self.neural[t]; top = hyps[0][1]
        for rank, (p, lp) in enumerate(hyps):
            p = nfc(p)
            if not p: continue
            f = feat(p); f[0] = max(f[0], (lp - top) * 0.5 + top * 0.3 + math.log(0.6 if not seen else 0.15))
            if f[1]["neural_in"] == 0: f[1].update(neural_lp_rel=lp - top, neural_rank=float(rank), neural_in=1.0)
        if not seen: f = feat(tok); f[0] = max(f[0], math.log(0.3) - 0.5)
        if self.use_aksh and t in self.aksh:
            c = self.aksh[t]
            for b, k in c.most_common(K_AKSH):
                f = feat(b); f[0] = max(f[0], math.log(0.05 if seen else 0.2)); f[1].update(aksh=1.0, aksh_logcount=math.log1p(k))
        n = len(out); tl = len(tok)
        for c, (em, f) in out.items():
            f.update(tok_seen=float(seen), tok_logcount=math.log1p(tot), neural_top_lp=top, copy=float(c == tok), latin=float(not NATIVE.search(c)), tok_latin_like=float(bool(re.fullmatch(r"[a-z]+", t))),
                     tok_has_digit=float(any(ch.isdigit() for ch in t)), cand_apos=float("'" in c or "\u2019" in c), len_ratio=len(c) / max(tl, 1), log_ncand=math.log(n), tok_len=float(tl), ind_uni=math.log(self.ind.prob((), self.wid(c))), big_uni=math.log(self.big.prob((), self.wid(c))),
                     big_oov=float(self.wid(c) == -3))
        self.cache[tok] = out; return out

    def lm_mats(self, prev, cur):
        """Bigram log-probs (in-domain, big, round-7 interpolation) for every (p in prev, c in cur): three |P|x|C| arrays."""
        P, C = [self.wid(p) for p in prev], [self.wid(c) for c in cur]
        ind = np.array([[self.ind.prob((p,), c) for c in C] for p in P]); big = np.array([[self.big.prob((p,), c) for c in C] for p in P])
        return np.log(ind), np.log(big), np.log(MU * ind + (1 - MU) * big)

    def lattice(self, toks):
        """Per position: candidate list, emission vector, context-free feature matrix, and the LM matrices to the previous position.
        Also runs the round-7 Viterbi (baseline path) and forward–backward (posterior feature) on the same lattice."""
        C = [self.cands(t) for t in toks]; cl = [list(cd) for cd in C]; em = [np.array([cd[c][0] for c in cl_]) for cd, cl_ in zip(C, cl)]
        X = [np.array([[cd[c][1][k] for k in CF] for c in cl_]) for cd, cl_ in zip(C, cl)]  # context-free part
        lms = [self.lm_mats([V.BOS] if t == 0 else cl[t - 1], cl[t]) for t in range(len(toks))]
        # baseline Viterbi + forward-backward with score = em + LAM * interpolated bigram
        T = [lm[2] * LAM + em[t][None, :] for t, lm in enumerate(lms)]
        delta, bp, alpha = [T[0][0]], [None], [T[0][0]]
        for t in range(1, len(toks)):
            s = delta[-1][:, None] + T[t]; bp.append(s.argmax(0)); delta.append(s.max(0)); alpha.append(np.logaddexp.reduce(alpha[-1][:, None] + T[t], axis=0))
        beta = [None] * len(toks); beta[-1] = np.zeros(len(cl[-1]))
        for t in range(len(toks) - 2, -1, -1): beta[t] = np.logaddexp.reduce(T[t + 1] + beta[t + 1][None, :], axis=1)
        Z = np.logaddexp.reduce(alpha[-1]); post = [a + b - Z for a, b in zip(alpha, beta)]
        path = [int(delta[-1].argmax())]
        for t in range(len(toks) - 1, 0, -1): path.append(int(bp[t][path[-1]]))
        path = path[::-1]; base = [cl[t][i] for t, i in enumerate(path)]
        for t in range(len(toks)):
            onp = np.zeros(len(cl[t])); onp[path[t]] = 1
            X[t] = np.column_stack([X[t], np.zeros((len(cl[t]), 3)), post[t], onp, np.full(len(cl[t]), float(len(toks[t])))])  # LM slots filled per edge
        return cl, X, lms, base

    def rows(self, cl, X, lms, t, prev_idx):
        """Feature rows for every candidate at position t given previous-candidate indices prev_idx (array) — shape (|prev|, |C_t|, F)."""
        R = np.repeat(X[t][None, :, :], len(prev_idx), axis=0).copy()
        for j in range(3): R[:, :, NCF + j] = lms[t][j][prev_idx]
        return R


def train_rows(sysm, data):
    Xs, ys = [], []
    for r, n in data:
        cl, X, lms, _ = sysm.lattice(r)
        for t in range(len(r)):
            prev = 0 if t == 0 else (cl[t - 1].index(n[t - 1]) if n[t - 1] in cl[t - 1] else None)
            if prev is None: continue  # reference context not in the lattice: no clean edge to learn from
            R = sysm.rows(cl, X, lms, t, np.array([prev]))[0]; Xs.append(R); ys.append(np.array([c == n[t] for c in cl[t]], dtype=float))
    return np.vstack(Xs), np.concatenate(ys)


def fit(kind, X, y):
    if kind == "logreg": m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    else: m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    return m.fit(X, y)


def logit(m, X):
    if isinstance(m, HistGradientBoostingClassifier): p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
    return m.decision_function(X)


def decode(sysm, model, toks):
    """Viterbi over lattice edges scored by the reranker's log-odds; returns (hypothesis, baseline hypothesis, lattice)."""
    cl, X, lms, base = sysm.lattice(toks); L = len(toks)
    delta = [logit(model, sysm.rows(cl, X, lms, 0, np.array([0]))[0])]; bp = [None]
    for t in range(1, L):
        R = sysm.rows(cl, X, lms, t, np.arange(len(cl[t - 1]))); S = logit(model, R.reshape(-1, R.shape[-1])).reshape(R.shape[0], R.shape[1])
        s = delta[-1][:, None] + S; bp.append(s.argmax(0)); delta.append(s.max(0))
    path = [int(delta[-1].argmax())]
    for t in range(L - 1, 0, -1): path.append(int(bp[t][path[-1]]))
    path = path[::-1]; return [cl[t][i] for t, i in enumerate(path)], base, cl


def score(data, hyps, seen):
    tot = ok = iv = iv_ok = uv = uv_ok = sent = 0
    for (r, n), h in zip(data, hyps):
        for a, b, p in zip(r, n, h):
            hit = p == b; tot += 1; ok += hit
            if a.lower() in seen: iv += 1; iv_ok += hit
            else: uv += 1; uv_ok += hit
        sent += h == n
    return dict(token_acc=ok / tot, in_vocab_acc=iv_ok / max(iv, 1), unseen_acc=uv_ok / max(uv, 1), unseen_share=uv / tot, sentence_acc=sent / len(data), n_tokens=tot, n_sentences=len(data))


def oracle(sysm, data):
    hit = tot = 0
    for r, n in data:
        for a, b in zip(r, n): tot += 1; hit += b in sysm.cands(a)
    return hit / tot


def error_classes(ref, hyp):
    """Round-7 inconsistency classes (Latin kept vs transliterated, য়/য, টো/তো) plus two that the round-8 errors made
    visible: the apostrophe marking /o/ (হল vs হ'ল) and the short/long vowel signs (ি/ী, ু/ূ)."""
    strip = lambda s: s.replace("য়", "য").replace("ড়", "ড").replace("ঢ়", "ঢ")
    if bool(NATIVE.search(ref)) != bool(NATIVE.search(hyp)): return "latin_kept_vs_transliterated"
    if strip(ref) == strip(hyp): return "ya_nukta (য়/য, ড়/ড, ঢ়/ঢ)"
    if len(ref) == len(hyp) and all(a == b or {a, b} == {"ট", "ত"} for a, b in zip(ref, hyp)): return "to (টো/তো)"
    apos = lambda s: s.replace("'", "").replace("\u2019", "")
    if apos(ref) == apos(hyp): return "apostrophe (হল/হ'ল)"
    vow = lambda s: s.replace("ী", "ি").replace("ূ", "ু")
    if vow(ref) == vow(hyp): return "vowel_length (ি/ী, ু/ূ)"
    return "other"


def main():
    os.makedirs(OUT, exist_ok=True); t00 = time.time()
    pairs, _ = load(PATH); rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs)); cut = int(0.8 * len(pairs)); tr, te = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    tr = [(r, [nfc(x) for x in n]) for r, n in tr]; te = [(r, [nfc(x) for x in n]) for r, n in te]
    dev, tr_lm = tr[:2000], tr[2000:]
    aksh = load_aksharantar(AKSH); bhasha = load_bhasha(BHASHA)
    if SMOKE: dev, te, bhasha = dev[:SMOKE], te[:SMOKE], bhasha[:SMOKE]; print(f"SMOKE run: {SMOKE} dev / {SMOKE} test sentences", flush=True)
    vocab = {}; print("building large LM", flush=True); big, vocab, lm_name = build_lm(vocab); print("  LM:", lm_name, flush=True)
    # char transformer (trained without dev, NFC targets): beam-5 alternatives for every token we will ever see
    ck = torch.load(CKPT, map_location="cpu"); sv, tv = N.Vocab([]), N.Vocab([]); sv.itos, tv.itos = ck["sv"], ck["tv"]
    sv.stoi, tv.stoi = {c: i for i, c in enumerate(sv.itos)}, {c: i for i, c in enumerate(tv.itos)}; model = N.Translit(len(sv.itos), len(tv.itos)); model.load_state_dict(ck["model"]); model.eval()
    toks = sorted({a.lower() for r, _ in dev + te + bhasha for a in r}); cachef = f"{OUT}/e2_discrim_neural.pkl"
    neural = pickle.load(open(cachef, "rb")) if os.path.exists(cachef) else {}
    todo = [t for t in toks if t not in neural]
    if todo:
        t0 = time.time(); N.predict(model, sv, tv, todo); neural.update(zip(todo, N.predict.alts)); pickle.dump(neural, open(cachef, "wb")); print(f"neural beam-{N.BEAM} for {len(todo)} tokens ({time.time() - t0:.0f}s)", flush=True)

    res = dict(config=dict(ckpt=CKPT, lam=LAM, mu=MU, k_seen=K_SEEN, k_aksh=K_AKSH, features=FEATS, dev_sentences=len(dev), big_lm=lm_name,
                           aksharantar_available=bool(aksh), bhasha_available=bool(bhasha), smoke=SMOKE))
    dev_sys = System(tr_lm, neural, aksh, big, vocab); test_sys = System(tr, neural, aksh, big, vocab)
    dev_sys0 = System(tr_lm, neural, aksh, big, vocab, use_aksh=False); test_sys0 = System(tr, neural, aksh, big, vocab, use_aksh=False)
    res["oracle_recall"] = dict(dev_without_aksharantar=oracle(dev_sys0, dev), dev_with_aksharantar=oracle(dev_sys, dev), test_without_aksharantar=oracle(test_sys0, te), test_with_aksharantar=oracle(test_sys, te))
    if bhasha: res["oracle_recall"].update(bhasha_without_aksharantar=oracle(test_sys0, bhasha), bhasha_with_aksharantar=oracle(test_sys, bhasha))
    print("oracle recall:", res["oracle_recall"], flush=True)
    res["aksharantar_coverage_of_test_tokens"] = sum(a.lower() in aksh for r, _ in te for a in r) / sum(len(r) for r, _ in te)

    # --- model selection on dev: 2-fold by sentence, decode the held-out fold with Viterbi over the reranker ---
    folds = [dev[:len(dev) // 2], dev[len(dev) // 2:]]; sel = {}
    t0 = time.time(); Xf = [train_rows(dev_sys, f) for f in folds]; print(f"dev lattices: {sum(len(y) for _, y in Xf)} edge rows, {Xf[0][1].mean():.3f} positive ({time.time() - t0:.0f}s)", flush=True)
    base_dev = [[dev_sys.lattice(r)[3] for r, _ in f] for f in folds]
    sel["baseline_round7_lattice (dev, decoder refit without dev)"] = score(dev, base_dev[0] + base_dev[1], dev_sys.dec.reads)
    for kind in ("logreg", "gbm"):
        hyps = []
        for k in (0, 1):
            m = fit(kind, *Xf[1 - k]); t0 = time.time(); hyps += [decode(dev_sys, m, r)[0] for r, _ in folds[k]]
        sel[kind] = score(dev, hyps, dev_sys.dec.reads); print(f"dev 2-fold {kind}: {sel[kind]} ({time.time() - t0:.0f}s/fold)", flush=True)
    print("dev baseline:", sel["baseline_round7_lattice (dev, decoder refit without dev)"], flush=True)
    res["dev_selection"] = sel; best = max(("logreg", "gbm"), key=lambda k: sel[k]["token_acc"]); res["selected"] = best
    if aksh:  # ablation on dev: no Aksharantar candidates/features
        Xf0 = [train_rows(dev_sys0, f) for f in folds]; hyps = []
        for k in (0, 1): m = fit(best, *Xf0[1 - k]); hyps += [decode(dev_sys0, m, r)[0] for r, _ in folds[k]]
        res["dev_selection"][f"{best} without Aksharantar"] = score(dev, hyps, dev_sys0.dec.reads); print("dev ablation no-aksh:", res["dev_selection"][f"{best} without Aksharantar"], flush=True)

    # --- final: reranker on all 2,000 dev sentences; test decoded with the decoder/LM fitted on all of tr ---
    Xall, yall = np.vstack([x for x, _ in Xf]), np.concatenate([y for _, y in Xf]); final = fit(best, Xall, yall)
    if best == "logreg":
        w = final[-1].coef_[0]; res["logreg_weights_standardised"] = {f: float(v) for f, v in zip(FEATS, w)}
    else:
        from sklearn.inspection import permutation_importance
        sub = np.random.RandomState(SEED).choice(len(yall), min(20000, len(yall)), replace=False)
        imp = permutation_importance(final, Xall[sub], yall[sub], n_repeats=3, random_state=SEED, scoring="neg_log_loss"); res["gbm_permutation_importance"] = {f: float(v) for f, v in zip(FEATS, imp.importances_mean)}
    t0 = time.time(); out = [decode(test_sys, final, r) for r, _ in te]; print(f"test decoded ({time.time() - t0:.0f}s)", flush=True)
    hyps, base = [o[0] for o in out], [o[1] for o in out]
    res["test"] = {"round7_published (rerank2 NFC, beta=0.05)": dict(token_acc=0.849, sentence_acc=0.281),
                   "round7_lattice_reproduced (Viterbi + em + 0.2*interp bigram, NFC, nodev transformer)": score(te, base, test_sys.dec.reads),
                   f"discriminative_{best}": score(te, hyps, test_sys.dec.reads)}
    if aksh:
        final0 = fit(best, np.vstack([x for x, _ in Xf0]), np.concatenate([y for _, y in Xf0]))
        out0 = [decode(test_sys0, final0, r)[0] for r, _ in te]; res["test"][f"discriminative_{best}_without_aksharantar"] = score(te, out0, test_sys0.dec.reads)
    for k, v in res["test"].items(): print(f"TEST {k}: {v}", flush=True)

    # --- error analysis on test ---
    conf = Counter(); cls = Counter(); attested = 0; nerr = 0
    for (r, n), h in zip(te, hyps):
        for a, b, p in zip(r, n, h):
            if p == b: continue
            nerr += 1; conf[(a.lower(), b, p)] += 1; cls[error_classes(b, p)] += 1; attested += p in test_sys.dec.reads.get(a.lower(), {})
    res["errors"] = dict(n_errors=nerr, top20=[dict(roman=a, reference=b, hypothesis=p, count=c, cls=error_classes(b, p)) for (a, b, p), c in conf.most_common(20)],
                         class_share={k: v / nerr for k, v in cls.most_common()}, hypothesis_attested_as_training_reading_share=attested / nerr)
    print("error classes:", res["errors"]["class_share"], "attested:", res["errors"]["hypothesis_attested_as_training_reading_share"], flush=True)
    for e in res["errors"]["top20"]: print("  ", e)

    # --- external test: Bhasha-Abhijnaanam Assamese ---
    if bhasha:
        outb = [decode(test_sys, final, r) for r, _ in bhasha]; hb, bb = [o[0] for o in outb], [o[1] for o in outb]
        lookup = {a: c.most_common(1)[0][0] for a, c in test_sys.dec.reads.items()}
        res["bhasha"] = dict(lookup_copy_fallback=score(bhasha, [[lookup.get(a.lower(), a) for a in r] for r, _ in bhasha], test_sys.dec.reads),
                             round7_lattice=score(bhasha, bb, test_sys.dec.reads), discriminative=score(bhasha, hb, test_sys.dec.reads),
                             aksharantar_only=score(bhasha, [[aksh[a.lower()].most_common(1)[0][0] if a.lower() in aksh else a for a in r] for r, _ in bhasha], test_sys.dec.reads))
        for k, v in res["bhasha"].items(): print(f"BHASHA {k}: {v}", flush=True)
    res["runtime_s"] = time.time() - t00
    json.dump(res, open(f"{OUT}/e2_discrim.json", "w"), indent=1, ensure_ascii=False); print(f"done in {res['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
