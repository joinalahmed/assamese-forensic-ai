"""E3 round 8: trained backends on top of the pretrained encoders, speaker-bootstrap uncertainty, and the
artefact that `score_pair.py` ships.

Same data, VAD, segmentation, speaker-disjoint folds (6→4: calls 1–3 train/calibrate, calls 4–5 evaluate;
4→6: swapped), adaptive S-norm and class-balanced logistic calibration as e3_speaker_plus.py / e3_encoders.py.
Primary labelling is *merged* (round 4 showed call 4 ch00 and call 5 ch00 are one agent, so `c400 == c500`
and the evaluation fold of 6→4 has 3 speakers, the training fold of 4→6 has 3); the as-labelled numbers are
reported alongside.

Systems: the three Mandarin-trained encoders that won round 3 (ERes2Net-base, ERes2Net-v2, CAM++ zh-en) and
their dev-selected fusion (L2-normalised embeddings concatenated), each with three backends
  * cosine (the round-3 baseline, no training),
  * LDA (shrinkage, ≤ n_speakers−1 dims) → length-norm → cosine,
  * two-covariance PLDA (x = μ + y + ε, y ~ N(0,B), ε ~ N(0,W); EM; closed-form log-LR scoring),
each with and without S-norm, 10 s and 3 s segments. Backends and calibration are fitted on the training fold's
segments only; the evaluation speakers are never seen.

Uncertainty: (a) 500 speaker bootstraps *within the evaluation fold* (resample evaluation speakers with
replacement, calibrator fixed) → 95% CI for EER and Cllr; (b) 500 speaker bootstraps of the *calibration*
speakers (resample training-fold speakers, refit only the calibrator, evaluate on the full evaluation fold) →
how much Cllr moves because the calibration set has 3–6 speakers.

Robustness of the selected system: 3/5/10/20 s segments, telephone band 300–3400 Hz, 10 dB white noise,
each both matched (backend + calibration refitted in the condition) and mismatched (clean-10 s system applied
to the degraded evaluation audio).

Writes results/e3_backend.json, results/e3_backend_model.npz (the shipped backend for score_pair.py) and figures.
Usage: cd forensic && ../.venv/bin/python -W ignore e3_backend.py [data_dir] [out_dir] [--cache DIR] [--boot N]
"""
import os, sys, json, hashlib, time, platform
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import e3_speaker_plus as P
from e3_encoder_probe import telephone, noisy
from metrics import LogisticCalibrator, report, tippett, eer, cllr, cllr_min

args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = args[0] if len(args) > 0 else "../data/calls"
OUT = args[1] if len(args) > 1 else "results"
CACHE = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
N_BOOT = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 500
SEED, SR = 0, 16000
ENCODERS = {"ERes2Net-base": "../models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
            "ERes2Net-v2": "../models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
            "CAM++ zh-en": "../models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"}
KEYS = [(c, ch) for c in range(1, 6) for ch in ("00", "01")]


# ----------------------------------------------------------------------------------------------- backends
class Cosine:
    """Round-3 baseline: L2-normalised embeddings, inner product. Nothing is trained."""
    name = "cosine"

    def fit(self, X, y): return self

    def transform(self, X): return normalize(X)

    def score(self, A, B): return self.transform(A) @ self.transform(B).T

    def params(self): return {}

    @classmethod
    def from_params(cls, p): return cls()


class LDACosine:
    """Centre → shrinkage LDA to ≤ n_speakers−1 dims → length-norm → cosine. With 3–6 training speakers the
    subspace has 2–5 dimensions, which is the whole story of this backend on this corpus."""
    name = "LDA + length-norm + cosine"

    def fit(self, X, y):
        self.mu = X.mean(0); n = len(set(y)); self.k = min(n - 1, X.shape[1])
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=self.k).fit(X - self.mu, y)
        return self

    def transform(self, X): return normalize((X - self.mu) @ self.W)

    def score(self, A, B): return self.transform(A) @ self.transform(B).T

    @property
    def W(self): return self.lda.scalings_[:, : self.k] if hasattr(self, "lda") else self._W

    def params(self): return {"lda_dims": int(self.k), "mu": self.mu, "W": self.W}

    @classmethod
    def from_params(cls, p):
        o = cls(); o.mu, o._W, o.k = p["mu"], p["W"], int(p["lda_dims"]); return o


class PLDA:
    """Two-covariance PLDA (Brümmer & de Villiers / Sizov et al.): x = μ + y + ε, y ~ N(0,B), ε ~ N(0,W).
    Pre-processing: centre → PCA whitening to `k` dims (k ≤ n_segments/4, default cap 32) → length-norm → re-centre.
    Trained by EM on speaker-labelled segments. Score of a pair = log N([x1;x2] | same) − log N(x1) − log N(x2) =
    ½x1ᵀQx1 + ½x2ᵀQx2 + x1ᵀPx2 + c with T = B+W, M = T − B T⁻¹ B, Q = T⁻¹ − M⁻¹, P = T⁻¹ B M⁻¹,
    c = ½(log|T| − log|M|). sklearn has no PLDA, hence this implementation (checked against scipy's logpdf in
    `_selftest`)."""
    name = "PLDA (two-covariance)"

    def __init__(self, k=32, iters=20, ridge=1e-3): self.k, self.iters, self.ridge = k, iters, ridge

    def _pre(self, X):
        Z = (X - self.mu) @ self.proj; return normalize(Z) - self.mu2

    def fit(self, X, y):
        y = np.asarray(y); n, d = X.shape; self.k = int(min(self.k, d, max(2, n // 4)))
        self.mu = X.mean(0); Xc = X - self.mu
        U, s, _ = np.linalg.svd(Xc, full_matrices=False)
        self.proj = (_[: self.k].T / (s[: self.k] / np.sqrt(n - 1) + 1e-6))  # whitening projection (d, k)
        Z = normalize(Xc @ self.proj); self.mu2 = Z.mean(0); Z = Z - self.mu2
        spk = sorted(set(y)); groups = [Z[y == s] for s in spk]; k = self.k
        means = np.array([g.mean(0) for g in groups]); cnt = np.array([len(g) for g in groups])
        B = np.cov(means.T, bias=True) + self.ridge * np.eye(k)
        W = sum(((g - m).T @ (g - m)) for g, m in zip(groups, means)) / n + self.ridge * np.eye(k)
        for _ in range(self.iters):  # EM
            Wi = np.linalg.inv(W); Bi = np.linalg.inv(B); Eyy = np.zeros((k, k)); R = np.zeros((k, k))
            for g, m, c in zip(groups, means, cnt):
                Pst = Bi + c * Wi; Cov = np.linalg.inv(Pst); mean_y = Cov @ Wi @ (c * m)
                Eyy += Cov + np.outer(mean_y, mean_y); r = g - mean_y; R += r.T @ r + c * Cov
            B = Eyy / len(groups) + self.ridge * np.eye(k); W = R / n + self.ridge * np.eye(k)
        self.B, self.W = B, W; T = B + W; Ti = np.linalg.inv(T); M = T - B @ Ti @ B; Mi = np.linalg.inv(M)
        self.Q = Ti - Mi; self.Pm = Ti @ B @ Mi; self.c = 0.5 * (np.linalg.slogdet(T)[1] - np.linalg.slogdet(M)[1])
        return self

    def transform(self, X): return self._pre(X)

    def score(self, A, B):
        Za, Zb = self._pre(A), self._pre(B)
        qa = 0.5 * np.einsum("ij,jk,ik->i", Za, self.Q, Za); qb = 0.5 * np.einsum("ij,jk,ik->i", Zb, self.Q, Zb)
        return qa[:, None] + qb[None, :] + Za @ self.Pm @ Zb.T + self.c

    def params(self): return {"plda_dims": int(self.k), "mu": self.mu, "proj": self.proj, "mu2": self.mu2, "Q": self.Q, "Pm": self.Pm, "c": float(self.c)}

    @classmethod
    def from_params(cls, p):
        o = cls(k=int(p["plda_dims"])); o.mu, o.proj, o.mu2, o.Q, o.Pm, o.c = p["mu"], p["proj"], p["mu2"], p["Q"], p["Pm"], float(p["c"]); return o


BACKENDS = {"cosine": Cosine, "LDA": LDACosine, "PLDA": PLDA}


def _selftest():
    """PLDA scoring formula vs. brute-force Gaussian log-densities."""
    from scipy.stats import multivariate_normal as mvn
    rng = np.random.RandomState(1); k = 4; m = PLDA(k=k); m.k = k
    m.mu = np.zeros(8); m.proj = np.eye(8)[:, :k]; m.mu2 = np.zeros(k)
    A = rng.randn(k, k); B = A @ A.T / k + 0.1 * np.eye(k); A = rng.randn(k, k); W = A @ A.T / k + 0.1 * np.eye(k)
    T = B + W; Ti = np.linalg.inv(T); M = T - B @ Ti @ B; Mi = np.linalg.inv(M)
    m.Q = Ti - Mi; m.Pm = Ti @ B @ Mi; m.c = 0.5 * (np.linalg.slogdet(T)[1] - np.linalg.slogdet(M)[1])
    x = rng.randn(3, 8); x[:, k:] = 0; x = x / np.linalg.norm(x, axis=1, keepdims=True)  # already unit-norm, k-dim
    S = m.score(x, x); J = np.block([[T, B], [B, T]])
    for i in range(3):
        for j in range(3):
            ref = mvn.logpdf(np.r_[x[i, :k], x[j, :k]], np.zeros(2 * k), J) - mvn.logpdf(x[i, :k], np.zeros(k), T) - mvn.logpdf(x[j, :k], np.zeros(k), T)
            assert abs(S[i, j] - ref) < 1e-8, (S[i, j], ref)
    ss, ds = rng.randn(400) + 1.5, rng.randn(3000); cal = LogisticCalibrator().fit(ss, ds)
    s = np.r_[ss, ds]; y = np.r_[np.ones(400, bool), np.zeros(3000, bool)]; w = np.where(y, 0.5 / 400, 0.5 / 3000) * len(y)
    a_, b_ = fit_logistic(s, y, w); assert abs(a_ - cal.m.coef_[0, 0]) < 1e-3 and abs(b_ - cal.m.intercept_[0]) < 1e-3, (a_, b_, cal.m.coef_, cal.m.intercept_)
    return True


# ----------------------------------------------------------------------------------------------- weighted metrics
def eer_w(s, same, w):
    """EER with per-trial weights (speaker-bootstrap multiplicities)."""
    o = np.argsort(s, kind="stable"); s, same, w = s[o], same[o], w[o]
    wss = w * same; wds = w * ~same; Wss, Wds = wss.sum(), wds.sum()
    miss = np.r_[0, np.cumsum(wss)[:-1]] / Wss  # same-source weight strictly below threshold s[i]
    fa = (Wds - np.r_[0, np.cumsum(wds)[:-1]]) / Wds  # different-source weight at or above s[i]
    i = np.argmin(np.abs(miss - fa)); return float((miss[i] + fa[i]) / 2)


def cllr_w(llr, same, w):
    ln2 = np.log(2); a = np.logaddexp(0, -llr[same]) / ln2; b = np.logaddexp(0, llr[~same]) / ln2
    return float(0.5 * np.average(a, weights=w[same]) + 0.5 * np.average(b, weights=w[~same]))


def trials(S, spk):
    """All upper-triangle trials of a within-set score matrix: scores, speaker ids of both sides."""
    spk = np.asarray(spk); iu = np.triu_indices(len(spk), 1); return S[iu], spk[iu[0]], spk[iu[1]]


def boot_eval(Se, se, cal, rng, n_boot):
    """Resample *evaluation speakers* with replacement (calibrator fixed). A trial's weight is the product of the
    multiplicities of its two speakers; trials of unsampled speakers drop out. 95% percentile intervals."""
    s, a, b = trials(Se, se); same = a == b; llr = cal(s); spk = sorted(set(se)); idx = {k: i for i, k in enumerate(spk)}
    ia, ib = np.array([idx[x] for x in a]), np.array([idx[x] for x in b]); E, C = [], []
    for _ in range(n_boot):
        cnt = np.bincount(rng.randint(len(spk), size=len(spk)), minlength=len(spk)); w = (cnt[ia] * cnt[ib]).astype(float)
        if (w * same).sum() == 0 or (w * ~same).sum() == 0: continue  # degenerate draw (e.g. 1 speaker): skipped
        E.append(eer_w(s, same, w)); C.append(cllr_w(llr, same, w))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(n_boot_valid=len(E), n_eval_speakers=len(spk), eer_ci95=q(E), cllr_ci95=q(C), eer_boot_median=float(np.median(E)), cllr_boot_median=float(np.median(C)))


def fit_logistic(x, y, w, iters=50, ridge=1e-6):
    """Weighted 1-D logistic regression by Newton's method (same solution as LogisticCalibrator / sklearn C=1e6,
    ~20× faster for the 500 calibration bootstraps). Returns slope, offset in natural-log units."""
    y = y.astype(float); mu, sd = np.average(x, weights=w), np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)) + 1e-12
    z = (x - mu) / sd; X = np.c_[z, np.ones_like(z)]; th = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ th))); g = X.T @ (w * (p - y)) + ridge * th; H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g); th -= step
        if np.abs(step).max() < 1e-10: break
    return float(th[0] / sd), float(th[1] - th[0] * mu / sd)


def boot_cal(Sd, sd, Se, se, rng, n_boot, max_trials=12000):
    """Resample *calibration speakers* with replacement, refit only the (class-balanced, weighted) logistic
    calibrator, evaluate Cllr on the full evaluation fold. Backend stays fixed."""
    s, a, b = trials(Sd, sd); same = a == b
    if len(s) > max_trials: pick = rng.choice(len(s), max_trials, replace=False); s, a, b, same = s[pick], a[pick], b[pick], same[pick]
    spk = sorted(set(sd)); idx = {k: i for i, k in enumerate(spk)}; ia, ib = np.array([idx[x] for x in a]), np.array([idx[x] for x in b])
    es, ea, eb = trials(Se, se); esame = ea == eb; C, A = [], []
    for _ in range(n_boot):
        cnt = np.bincount(rng.randint(len(spk), size=len(spk)), minlength=len(spk)); w = (cnt[ia] * cnt[ib]).astype(float)
        if (w * same).sum() == 0 or (w * ~same).sum() == 0: continue
        w = w * np.where(same, 0.5 / (w * same).sum(), 0.5 / (w * ~same).sum()) * len(w)
        a_, b_ = fit_logistic(s, same, w); llr = a_ * es + b_; C.append(cllr(llr[esame], llr[~esame])); A.append(a_)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(n_boot_valid=len(C), n_cal_speakers=len(spk), cllr_ci95=q(C), cllr_boot_median=float(np.median(C)), slope_ci95=q(A))


# ----------------------------------------------------------------------------------------------- data / folds
def load_segments(sec, fx=None, cache=None):
    """Segments, speaker labels (as given), dev mask (calls 1–3). Embeddings for each encoder, cached on disk."""
    segs, spk, dev = [], [], []
    for k in KEYS:
        s = P.speech_segments(f"{D}/call{k[0]}_ch{k[1]}.wav", sec); s = [fx(x) for x in s] if fx else s
        segs += s; spk += [f"c{k[0]}{k[1]}"] * len(s); dev += [k[0] <= 3] * len(s)
    emb = {}
    for name, path in ENCODERS.items():
        cf = f"{CACHE}/{cache}_{name.replace(' ', '_').replace('+', 'p')}.npy" if CACHE and cache else None
        if cf and os.path.exists(cf): emb[name] = np.load(cf)
        else:
            emb[name] = P.embed_all(path, segs)
            if cf: os.makedirs(CACHE, exist_ok=True); np.save(cf, emb[name])
    return segs, np.array(spk), np.array(dev), emb


def merge(spk): return np.where(np.isin(spk, ["c400", "c500"]), "c400+c500", spk)


def dev_order(emb, spk, dev):
    """Encoder ranking by EER among development speakers (calls 1–3) only, as in round 3."""
    def dev_eer(E):
        X = normalize(E[dev]); s, a, b = trials(X @ X.T, spk[dev]); return eer(s[a == b], s[a != b])
    return sorted(emb, key=lambda n: dev_eer(emb[n])), {n: dev_eer(emb[n]) for n in emb}


def heldout_dev_scores(Xd, sd, backend_cls, use_snorm, rng):
    """Calibration scores from speakers the backend has *not* seen: split the training speakers into two
    speaker-disjoint halves, fit the backend on one half, score the other half within itself (S-norm cohort =
    the fitting half), and vice versa. Needs ≥ 2 speakers per half, i.e. ≥ 4 training speakers; returns None otherwise."""
    spk = sorted(set(sd))
    if len(spk) < 4: return None
    perm = rng.permutation(spk); halves = [set(perm[: len(spk) // 2]), set(perm[len(spk) // 2:])]; ss, ds = [], []
    for h in halves:
        m = np.isin(sd, list(h)); bk = backend_cls().fit(Xd[~m], sd[~m]); S = bk.score(Xd[m], Xd[m])
        if use_snorm: S = P.snorm(S, bk.score(Xd[m], Xd[~m]))
        a, b = P.trials_from_matrix(S, list(sd[m]), rng); ss.append(a); ds.append(b)
    return np.concatenate(ss), np.concatenate(ds)


def run_backend(E, spk, dev, rng, backend_cls, use_snorm, n_boot=0, want_llr=False, heldout_cal=False):
    """Mirror of e3_speaker_plus.run with a trainable backend. Returns per-fold metrics (+ bootstraps, + LLRs).
    heldout_cal=False: calibration (and S-norm cohort statistics on the training side) use in-sample scores of the
    training speakers, exactly as the existing scripts do. heldout_cal=True: see heldout_dev_scores."""
    out = {}
    for fold, (dm, em) in {"6→4": (dev, ~dev), "4→6": (~dev, dev)}.items():
        Xd, Xe, sd, se = E[dm], E[em], spk[dm], spk[em]
        bk = backend_cls().fit(Xd, sd); Sd, Se = bk.score(Xd, Xd), bk.score(Xe, Xe)
        if use_snorm:
            Cd = Sd.copy(); Cd[sd[:, None] == sd[None, :]] = np.nan; Sd = P.snorm(Sd, Cd); Se = P.snorm(Se, bk.score(Xe, Xd))
        if heldout_cal:
            ho = heldout_dev_scores(Xd, sd, backend_cls, use_snorm, rng)
            if ho is None: out[fold] = dict(skipped=f"held-out calibration impossible with {len(set(sd))} training speakers", n_train_speakers=len(set(sd)), n_eval_speakers=len(set(se))); continue
            d_ss, d_ds = ho
        else: d_ss, d_ds = P.trials_from_matrix(Sd, list(sd), rng)
        e_ss, e_ds = P.trials_from_matrix(Se, list(se), rng)
        cal = LogisticCalibrator().fit(d_ss, d_ds); r = report(e_ss, e_ds, cal); r["n_train_speakers"] = len(set(sd)); r["n_eval_speakers"] = len(set(se))
        r["cal_slope"], r["cal_offset"] = float(cal.m.coef_[0, 0]), float(cal.m.intercept_[0])
        if n_boot:
            r["boot_eval"] = boot_eval(Se, se, cal, rng, n_boot)
            if not heldout_cal: r["boot_cal"] = boot_cal(Sd, sd, Se, se, rng, n_boot)
        if want_llr: r["llr"] = (cal(e_ss), cal(e_ds)); r["_fit"] = (bk, cal, Xd, sd)
        out[fold] = r
    return out


def strip(r): return {f: {k: v for k, v in rr.items() if k not in ("llr", "_fit")} for f, rr in r.items()}


def fmt(r): return "  ".join(f"{f}: {rr['skipped']}" if "skipped" in rr else f"{f}: EER {rr['eer']:.3f} [{rr['boot_eval']['eer_ci95'][0]:.3f},{rr['boot_eval']['eer_ci95'][1]:.3f}] Cllr {rr['cllr']:.2f} [{rr['boot_eval']['cllr_ci95'][0]:.2f},{rr['boot_eval']['cllr_ci95'][1]:.2f}] min {rr['cllr_min']:.2f}" if "boot_eval" in rr else f"{f}: EER {rr['eer']:.3f} Cllr {rr['cllr']:.2f} min {rr['cllr_min']:.2f}" for f, rr in r.items())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    assert _selftest(); os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED); t0 = time.time()
    res = {"config": dict(seed=SEED, n_boot=N_BOOT, folds="6→4: train/calibrate on calls 1–3, evaluate calls 4–5; 4→6: swapped",
                          primary_labelling="merged (c400 == c500, round 4)", encoders=ENCODERS, snorm_topk=P.TOPK, different_trials_cap=12000,
                          python=platform.python_version(), numpy=np.__version__, sklearn=__import__("sklearn").__version__, sherpa_onnx=__import__("sherpa_onnx").__version__)}
    systems_res, plots, cache = {}, {}, {}
    for sec in (10, 3):
        segs, spk0, dev, emb = load_segments(sec, cache=f"clean{sec}"); cache[sec] = (spk0, dev, emb)
        order, deer = dev_order(emb, spk0, dev); res[f"{sec} s dev-speaker EER (selection)"] = deer; res[f"{sec} s dev-selection order"] = order
        systems = dict(emb); systems["fusion (3 encoders, dev-selected)"] = normalize(np.hstack([emb[n] for n in order[:3]]))
        print(f"{sec} s: {len(segs)} segments; dev order {order}", flush=True)
        for lab, spk in (("merged", merge(spk0)), ("as-labelled", spk0)):
            for name, E in systems.items():
                for bname, bcls in BACKENDS.items():
                    for sn in (False, True):
                        tag = f"{sec} s | {lab} | {name} | {bname}{' + S-norm' if sn else ''}"
                        for ho in ((False, True) if bname != "cosine" else (False,)):
                            t = tag + (" | held-out cal" if ho else "")
                            r = run_backend(E, spk, dev, rng, bcls, sn, n_boot=N_BOOT, want_llr=True, heldout_cal=ho)
                            systems_res[t] = strip(r); plots[t] = r
                            print(f"  {t:94s} {fmt(r)}", flush=True)
    res["systems"] = systems_res

    # ---- select the system to ship / probe: lowest mean-of-folds Cllr among the 24 merged 10 s configurations.
    # This is a post-hoc choice on evaluation-fold numbers (there is no third speaker set to select on), so the
    # selected system's own figures are optimistically biased; the CIs below are the honest part.
    cands = [t for t in systems_res if t.startswith("10 s | merged") and all("skipped" not in systems_res[t][f] for f in ("6→4", "4→6"))]
    best = min(cands, key=lambda t: (np.mean([systems_res[t][f]["cllr"] for f in ("6→4", "4→6")]), np.mean([systems_res[t][f]["eer"] for f in ("6→4", "4→6")])))
    res["selected_system"] = dict(tag=best, rule=f"min mean-of-folds Cllr over the {len(cands)} merged 10 s configurations with both folds available (post hoc)", **systems_res[best])
    print(f"\nselected: {best}", flush=True)
    parts = [x.strip() for x in best.split("|")]; sel_name, sel_backend = parts[2], parts[3]; sel_ho = len(parts) > 4
    sel_snorm = sel_backend.endswith("+ S-norm"); sel_backend = sel_backend.replace(" + S-norm", "")

    # ---- robustness of the selected system (merged labelling): duration, telephone band, 10 dB white noise.
    spk0, dev, emb10 = cache[10]; order10 = res["10 s dev-selection order"]
    def system_E(emb): return normalize(np.hstack([emb[n] for n in order10[:3]])) if sel_name.startswith("fusion") else emb[sel_name]
    rob = {}; nrng = np.random.RandomState(SEED)
    conds = {"clean 3 s": (3, None), "clean 5 s": (5, None), "clean 10 s": (10, None), "clean 20 s": (20, None),
             "telephone band 300–3400 Hz, 10 s": (10, telephone), "white noise 10 dB, 10 s": (10, noisy(10, nrng))}
    for cond, (sec, fx) in conds.items():
        if sec in cache and fx is None: spk_c, dev_c, emb_c = cache[sec]
        else: _, spk_c, dev_c, emb_c = load_segments(sec, fx, cache=cond.split(",")[0].replace(" ", "_").replace("–", "-"))
        E = system_E(emb_c); spk_m = merge(spk_c); r = run_backend(E, spk_m, dev_c, rng, BACKENDS[sel_backend], sel_snorm, n_boot=N_BOOT, heldout_cal=sel_ho)
        rob[cond] = {"matched": dict(n_segments=int(len(spk_c)), **strip(r))}
        if fx is not None:  # mismatched: clean-10 s backend + calibration applied to degraded evaluation audio
            Ec = system_E(emb10); mm = {}
            for fold, (dm, em) in {"6→4": (dev, ~dev), "4→6": (~dev, dev)}.items():
                bk = BACKENDS[sel_backend]().fit(Ec[dm], spk_m[dm]); Sd = bk.score(Ec[dm], Ec[dm]); Se = bk.score(E[em], E[em])
                if sel_snorm:
                    Cd = Sd.copy(); Cd[spk_m[dm][:, None] == spk_m[dm][None, :]] = np.nan; Sd = P.snorm(Sd, Cd); Se = P.snorm(Se, bk.score(E[em], Ec[dm]))
                ho = heldout_dev_scores(Ec[dm], spk_m[dm], BACKENDS[sel_backend], sel_snorm, rng) if sel_ho else P.trials_from_matrix(Sd, list(spk_m[dm]), rng)
                if ho is None: mm[fold] = dict(skipped=f"held-out calibration impossible with {len(set(spk_m[dm]))} training speakers"); continue
                d_ss, d_ds = ho; e_ss, e_ds = P.trials_from_matrix(Se, list(spk_m[em]), rng)
                cal = LogisticCalibrator().fit(d_ss, d_ds); mm[fold] = report(e_ss, e_ds, cal); mm[fold]["boot_eval"] = boot_eval(Se, spk_m[em], cal, rng, N_BOOT)
            rob[cond]["mismatched (clean 10 s system)"] = mm
        print(f"  robustness {cond:36s} matched " + fmt(r) + ("" if fx is None else "  | mismatched " + fmt(mm)), flush=True)
    res["robustness"] = rob

    # ---- ship the 6→4 system (trained + calibrated on calls 1–3, validated on calls 4–5) for score_pair.py
    bk, cal, Xd, sd = plots[best]["6→4"]["_fit"]; prm = bk.params()
    lss, lds = plots[best]["6→4"]["llr"]; l10 = lambda v: [float(np.min(v) / np.log(10)), float(np.max(v) / np.log(10))]
    arrays = dict(cohort=Xd, cohort_spk=sd, cal_a=np.array([float(cal.m.coef_[0, 0])]), cal_b=np.array([float(cal.m.intercept_[0])]))
    arrays.update({k: np.asarray(v) for k, v in prm.items()})
    meta = dict(encoders=order10[:3] if sel_name.startswith("fusion") else [sel_name], encoder_files={n: os.path.basename(ENCODERS[n]) for n in ENCODERS},
                encoder_sha256={n: sha256(ENCODERS[n]) for n in ENCODERS}, backend=sel_backend, snorm=sel_snorm, snorm_topk=P.TOPK, segment_seconds=10,
                fusion=sel_name.startswith("fusion"), calibration="held-out (speaker-disjoint halves of the training fold)" if sel_ho else "in-sample training-fold trials", training="calls 1–3 (6 speakers), segments of 10 s speech after energy VAD", labelling="merged (c400 == c500)",
                validation={"fold": "6→4 (calls 4–5, 3 speakers after merging)", **systems_res[best]["6→4"], "validated_log10lr_range": {"same_speaker": l10(lss), "different_speaker": l10(lds)},
                            "other_fold_4→6": systems_res[best]["4→6"]},
                robustness=rob, selection_rule=res["selected_system"]["rule"], versions=res["config"], created=time.strftime("%Y-%m-%d"))
    np.savez(f"{OUT}/e3_backend_model.npz", meta=json.dumps(meta, ensure_ascii=False), **arrays)
    res["shipped_model"] = dict(file="results/e3_backend_model.npz", sha256=sha256(f"{OUT}/e3_backend_model.npz"), **{k: v for k, v in meta.items() if k not in ("robustness", "validation")})
    res["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/e3_backend.json", "w"), indent=1, ensure_ascii=False)

    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = {"cosine": "#4C72B0", "LDA": "#DD8452", "PLDA": "#55A868"}
    # 1. Tippett of the selected system, both folds
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, f in zip(ax, ("6→4", "4→6")):
        rr = plots[best][f]; tippett(*rr["llr"], a); a.set_title(f"{best.replace('10 s | merged | ', '')}\nfold {f}: EER {rr['eer']:.1%} {rr['boot_eval']['eer_ci95'][0]:.1%}–{rr['boot_eval']['eer_ci95'][1]:.1%}; Cllr {rr['cllr']:.2f} ({rr['boot_eval']['cllr_ci95'][0]:.2f}–{rr['boot_eval']['cllr_ci95'][1]:.2f})", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/e3_backend_tippett.png", dpi=150); plt.close(fig)
    # 2. EER and Cllr with speaker-bootstrap CIs, 10 s merged, both folds, all systems
    VARIANTS = [(b, sn, ho) for b in BACKENDS for ho in ((False, True) if b != "cosine" else (False,)) for sn in (False, True)]
    names = list(ENCODERS) + ["fusion (3 encoders, dev-selected)"]; fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex="col")
    for col, f in enumerate(("6→4", "4→6")):
        for row, (metric, lab) in enumerate((("eer", "EER"), ("cllr", "Cllr"))):
            a = axes[row, col]; x = 0
            for name in names:
                for bname, sn, ho in VARIANTS:
                    rr = systems_res[f"10 s | merged | {name} | {bname}{' + S-norm' if sn else ''}{' | held-out cal' if ho else ''}"][f]
                    if "skipped" not in rr:
                        lo, hi = rr["boot_eval"][f"{metric}_ci95"]
                        a.errorbar(x, rr[metric], yerr=[[rr[metric] - lo], [hi - rr[metric]]], fmt="^" if ho else ("o" if sn else "s"), color=C[bname], ms=4, capsize=2, mfc=C[bname] if sn else "white")
                    x += 1
                x += 1
            a.set_ylabel(lab); a.grid(axis="y", alpha=.3)
            if row == 0: a.set_title(f"fold {f}: {systems_res[f'10 s | merged | ERes2Net-base | cosine'][f]['n_eval_speakers']} evaluation speakers (95% speaker-bootstrap CI)", fontsize=9)
            if row == 1: a.set_xticks([i * (len(VARIANTS) + 1) + len(VARIANTS) / 2 - 0.5 for i in range(len(names))]); a.set_xticklabels([n.replace(" (3 encoders, dev-selected)", "") for n in names], fontsize=8)
    from matplotlib.lines import Line2D
    axes[0, 0].legend([Line2D([], [], color=C[b], marker="s", ls="") for b in BACKENDS] + [Line2D([], [], color="k", marker="s", mfc="white", ls=""), Line2D([], [], color="k", marker="o", ls=""), Line2D([], [], color="k", marker="^", ls="")],
                      list(BACKENDS) + ["no S-norm", "S-norm", "held-out calibration"], fontsize=7, ncol=6, loc="upper left")
    fig.suptitle("E3 round 8: backends on the three best encoders, 10 s segments, merged labelling", fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/e3_backend_ci.png", dpi=150); plt.close(fig)
    # 3. duration / channel robustness of the selected system
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, (metric, lab) in zip(ax, (("eer", "EER"), ("cllr", "Cllr"))):
        for f, mk in (("6→4", "o"), ("4→6", "s")):
            secs = [3, 5, 10, 20]; v = [rob[f"clean {s} s"]["matched"][f][metric] for s in secs]; ci = np.array([rob[f"clean {s} s"]["matched"][f]["boot_eval"][f"{metric}_ci95"] for s in secs])
            a.errorbar(secs, v, yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], marker=mk, capsize=2, label=f"fold {f}")
            for cond, dx, c in (("telephone band 300–3400 Hz, 10 s", -0.6, "#C44E52"), ("white noise 10 dB, 10 s", 0.6, "#8172B2")):
                rr = rob[cond]["mismatched (clean 10 s system)"][f]
                if "skipped" in rr: continue
                a.plot(10 + dx, rr[metric], marker=mk, color=c, ls="", label=f"{cond.split(',')[0]} (mismatched)" if f == "6→4" else None)
        a.set_xscale("log"); a.set_xticks([3, 5, 10, 20]); a.set_xticklabels(["3 s", "5 s", "10 s", "20 s"]); a.set_xticks([], minor=True); a.set_xlabel("speech per segment"); a.set_ylabel(lab); a.grid(alpha=.3)
    ax[0].legend(fontsize=7); fig.suptitle(f"robustness of the selected system: {best.replace('10 s | merged | ', '')}", fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/e3_backend_robustness.png", dpi=150)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
