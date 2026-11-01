"""E3: speaker comparison on Assamese call-centre telephone speech (GMM-UBM supervectors).

5 calls x 2 channels (= 10 speakers), 16 kHz. Energy VAD -> 20 MFCC + deltas -> 64-comp diagonal
UBM fitted on development speakers only -> MAP mean adaptation (r=16) supervector per 3 s / 10 s
segment -> cosine score -> logistic calibration on development trials.
Paper reference (Table 8): 10 s EER 16.3% / 7.6%, 3 s EER 25.2% / 12.2%, all Cllr > 1 (min 0.23-0.75).
"""
import sys, json, os, itertools
import numpy as np, soundfile as sf
from scipy.fft import dct
from scipy.signal import stft
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from metrics import LogisticCalibrator, report, tippett, eer

D = sys.argv[1] if len(sys.argv) > 1 else "../data/calls"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED, SR, HOP, WIN, NMEL, NCEP, NMIX, RELEVANCE = 0, 16000, 160, 400, 40, 20, 64, 16


def mel_fb(n_fft=512):
    f = lambda hz: 2595 * np.log10(1 + hz / 700); fi = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = fi(np.linspace(f(0), f(SR / 2), NMEL + 2)); bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((NMEL, n_fft // 2 + 1))
    for i in range(NMEL):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        fb[i, l:c] = (np.arange(l, c) - l) / max(c - l, 1); fb[i, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = mel_fb()


def deltas(x, w=2):
    pad = np.pad(x, ((w, w), (0, 0)), mode="edge"); n = np.arange(1, w + 1)
    return sum(k * (pad[w + k: len(x) + w + k] - pad[w - k: len(x) + w - k]) for k in n) / (2 * (n ** 2).sum())


def features(path):
    x, sr = sf.read(path, dtype="float32"); x = x if x.ndim == 1 else x.mean(1); assert sr == SR, sr
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])  # pre-emphasis
    _, _, Z = stft(x, nperseg=WIN, noverlap=WIN - HOP, nfft=512, window="hamming", boundary=None, padded=False)
    P = np.abs(Z.T) ** 2  # (frames, bins)
    loge = np.log(P.sum(1) + 1e-10)
    lo, hi = np.percentile(loge, 5), np.percentile(loge, 95)
    vad = loge > lo + 0.4 * (hi - lo)  # threshold between noise floor and speech peak
    mfcc = dct(np.log(P @ FB.T + 1e-10), type=2, axis=1, norm="ortho")[:, 1:NCEP + 1]
    F = np.hstack([mfcc, deltas(mfcc)])
    F = F[vad]; F = (F - F.mean(0)) / (F.std(0) + 1e-8)  # global mean/variance normalisation
    return F


def segments(F, sec):
    n = sec * 100
    return [F[i:i + n] for i in range(0, len(F) - n + 1, n)]


def map_supervector(ubm, F):
    g = ubm.predict_proba(F); n = g.sum(0); ex = (g.T @ F) / np.maximum(n[:, None], 1e-10)
    a = (n / (n + RELEVANCE))[:, None]; mu = a * ex + (1 - a) * ubm.means_
    return ((mu - ubm.means_) * np.sqrt(ubm.weights_)[:, None] / np.sqrt(ubm.covariances_)).ravel()


def trials(meta, rng, max_ds=12000):
    """meta: list of speaker ids per segment. Same-speaker = distinct segments of one speaker."""
    ss = [(i, j) for i, j in itertools.combinations(range(len(meta)), 2) if meta[i] == meta[j]]
    ds = [(i, j) for i, j in itertools.combinations(range(len(meta)), 2) if meta[i] != meta[j]]
    if len(ds) > max_ds: ds = [ds[k] for k in rng.choice(len(ds), max_ds, replace=False)]
    return ss, ds


def cos_scores(X, pairs):
    X = normalize(X); return np.array([float(X[i] @ X[j]) for i, j in pairs])


def run_fold(feats, dev, ev, sec, speaker_of, rng):
    seg = {k: segments(feats[k], sec) for k in feats}
    dev_frames = np.vstack([feats[k] for k in dev if k in feats])
    sub = dev_frames[rng.choice(len(dev_frames), min(len(dev_frames), 60000), replace=False)]
    ubm = GaussianMixture(NMIX, covariance_type="diag", max_iter=50, random_state=SEED, reg_covar=1e-3).fit(sub)

    def build(keys):
        X, spk, raw = [], [], []
        for k in keys:
            for s in seg[k]:
                X.append(map_supervector(ubm, s)); spk.append(speaker_of[k]); raw.append(np.r_[s.mean(0), s.std(0)])
        return np.array(X), spk, np.array(raw)

    Xd, sd, rd = build(dev); Xe, se, re_ = build(ev)
    ss_d, ds_d = trials(sd, rng); ss_e, ds_e = trials(se, rng)
    cal = LogisticCalibrator().fit(cos_scores(Xd, ss_d), cos_scores(Xd, ds_d))
    s_ss, s_ds = cos_scores(Xe, ss_e), cos_scores(Xe, ds_e)
    out = report(s_ss, s_ds, cal)
    out["eer_raw"] = eer(cos_scores(re_, ss_e), cos_scores(re_, ds_e))
    out["n_dev_speakers"], out["n_eval_speakers"] = len(set(sd)), len(set(se))
    return out, (cal(s_ss), cal(s_ds)), (Xe, se), (s_ss, s_ds, ss_e, ds_e)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    feats = {}
    for c in range(1, 6):
        for ch in ("00", "01"):
            feats[(c, ch)] = features(f"{D}/call{c}_ch{ch}.wav")
            print(f"call{c} ch{ch}: {len(feats[(c, ch)]) / 100:.0f} s speech", flush=True)
    print(f"total speech {sum(len(f) for f in feats.values()) / 6000:.1f} min", flush=True)
    ident = {k: f"c{k[0]}{k[1]}" for k in feats}
    folds = {"6→4": ([k for k in feats if k[0] <= 3], [k for k in feats if k[0] >= 4]),
             "4→6": ([k for k in feats if k[0] >= 4], [k for k in feats if k[0] <= 3])}
    res = {}
    for sec in (10, 3):
        for name, (dev, ev) in folds.items():
            out, ll, (Xe, se), _ = run_fold(feats, dev, ev, sec, ident, rng); res[f"{sec}s {name}"] = out
            print(f"{sec}s {name}: EER {out['eer']:.3f} raw {out['eer_raw']:.3f} Cllr {out['cllr']:.3f} min {out['cllr_min']:.3f}  ({out['n_ss']}/{out['n_ds']})", flush=True)
            if sec == 10: res.setdefault("_ll10", []).append(ll)
    # labelling check: centroid cosine between channels (10 s, UBM from all speakers)
    allf = np.vstack(list(feats.values())); ubm = GaussianMixture(NMIX, covariance_type="diag", max_iter=50, random_state=SEED, reg_covar=1e-3).fit(allf[rng.choice(len(allf), 60000, replace=False)])
    keys = list(feats); C = normalize(np.array([np.mean([map_supervector(ubm, s) for s in segments(feats[k], 10)], 0) for k in keys]))
    M = C @ C.T; np.fill_diagonal(M, np.nan)
    pairs = sorted(((M[i, j], keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys)) if keys[i][0] != keys[j][0]), reverse=True)
    res["most_similar_cross_call_channels"] = [(float(a), str(b), str(c)) for a, b, c in pairs[:3]]
    print("most similar cross-call channel pairs:", res["most_similar_cross_call_channels"], flush=True)
    # re-score with call-4/5 channel 00 treated as one speaker (suspected repeated agent)
    merged = dict(ident); merged[(4, "00")] = merged[(5, "00")] = "agent"
    for sec in (10,):
        out, *_ = run_fold(feats, folds["6→4"][0], folds["6→4"][1], sec, merged, rng); res["merged_10s_6→4"] = out
        print("merged 10 s 6→4:", {k: round(v, 3) for k, v in out.items() if k in ("eer", "cllr_min")}, flush=True)
    ll = res.pop("_ll10")
    json.dump(res, open(f"{OUT}/e3.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    tippett(np.concatenate([a for a, _ in ll]), np.concatenate([b for _, b in ll]), ax[0])
    im = ax[1].imshow(M, cmap="viridis"); ax[1].set_xticks(range(10)); ax[1].set_yticks(range(10))
    lab = [f"c{k[0]}{'AB'[int(k[1])]}" for k in keys]; ax[1].set_xticklabels(lab, rotation=90); ax[1].set_yticklabels(lab); fig.colorbar(im, ax=ax[1])
    fig.tight_layout(); fig.savefig(f"{OUT}/e3.png", dpi=150)
