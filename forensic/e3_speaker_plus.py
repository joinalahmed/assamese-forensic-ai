"""E3+: speaker comparison with pretrained neural embeddings and impostor score normalisation.

Same data, VAD, segmentation, speaker-disjoint folds and calibration protocol as e3_speaker.py. Upgrades:
  * pretrained speaker encoders via sherpa-onnx (CAM++ trained on VoxCeleb; ERes2Net trained on 3D-Speaker),
    applied cross-lingually to Assamese telephone speech, no fine-tuning;
  * adaptive S-norm (symmetric z/t normalisation against a cohort of development-speaker segments),
    which is what makes a calibrator fitted on a handful of speakers transfer;
  * the classical GMM-UBM supervector backend with the same S-norm, for a like-for-like comparison.
"""
import sys, os, json, itertools
import numpy as np, soundfile as sf
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import e3_speaker as B
from metrics import LogisticCalibrator, report, tippett

D = sys.argv[1] if len(sys.argv) > 1 else "../data/calls"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
MODELS = {"CAM++ (VoxCeleb)": "../models/wespeaker_en_voxceleb_CAM++.onnx",
          "ERes2Net (3D-Speaker)": "../models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"}
SEED, TOPK = 0, 100


def speech_segments(path, sec):
    """Raw-waveform segments of `sec` seconds of *speech* (VAD frames concatenated), same VAD as e3_speaker."""
    x, sr = sf.read(path, dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
    from scipy.signal import stft
    _, _, Z = stft(np.append(x[0], x[1:] - 0.97 * x[:-1]), nperseg=B.WIN, noverlap=B.WIN - B.HOP, nfft=512, window="hamming", boundary=None, padded=False)
    loge = np.log((np.abs(Z.T) ** 2).sum(1) + 1e-10); lo, hi = np.percentile(loge, 5), np.percentile(loge, 95)
    vad = loge > lo + 0.4 * (hi - lo)
    frames = np.where(vad)[0]
    speech = np.concatenate([x[f * B.HOP: f * B.HOP + B.HOP] for f in frames])
    n = sec * sr
    return [speech[i:i + n] for i in range(0, len(speech) - n + 1, n)]


def embed_all(model_path, segs):
    import sherpa_onnx
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=4))
    out = []
    for s in segs:
        st = ex.create_stream(); st.accept_waveform(16000, s); st.input_finished(); out.append(np.array(ex.compute(st)))
    return normalize(np.array(out))


def snorm(S, C, topk=TOPK):
    """Adaptive S-norm. S[i,j]: raw cosine between items i and j. C[i,:]: cosine of item i with cohort segments
    (NaN where the cohort segment belongs to item i's own speaker, so the cohort is always other-speaker speech)."""
    m, sd = np.zeros(len(C)), np.zeros(len(C))
    for i, row in enumerate(C):
        v = np.sort(row[~np.isnan(row)])[-topk:]; m[i], sd[i] = v.mean(), v.std() + 1e-9
    return 0.5 * ((S - m[:, None]) / sd[:, None] + (S - m[None, :]) / sd[None, :])


def trials_from_matrix(S, spk, rng, max_ds=12000):
    n = len(spk); iu = np.triu_indices(n, 1); same = np.array([spk[i] == spk[j] for i, j in zip(*iu)])
    ss = S[iu][same]; ds = S[iu][~same]
    if len(ds) > max_ds: ds = ds[rng.choice(len(ds), max_ds, replace=False)]
    return ss, ds


def run(E, spk, dev_mask, rng, use_snorm):
    """E: embeddings (n, d) L2-normalised; returns metrics with calibration fitted on dev speakers."""
    out = {}
    for name, (dm, em) in {"6→4": (dev_mask, ~dev_mask), "4→6": (~dev_mask, dev_mask)}.items():
        Xd, Xe = E[dm], E[em]
        Sd, Se = Xd @ Xd.T, Xe @ Xe.T
        if use_snorm:
            # cohort = development segments. On the dev side a segment's own speaker is masked out of its cohort,
            # so dev and eval scores are normalised against other-speaker speech alike (else calibration breaks).
            sd_ = np.array(spk)[dm]; Cd = Xd @ Xd.T; Cd[sd_[:, None] == sd_[None, :]] = np.nan
            Sd = snorm(Sd, Cd); Se = snorm(Se, Xe @ Xd.T)
        d_ss, d_ds = trials_from_matrix(Sd, list(np.array(spk)[dm]), rng); e_ss, e_ds = trials_from_matrix(Se, list(np.array(spk)[em]), rng)
        cal = LogisticCalibrator().fit(d_ss, d_ds); r = report(e_ss, e_ds, cal); r["llr"] = (cal(e_ss), cal(e_ds)); out[name] = r
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    keys = [(c, ch) for c in range(1, 6) for ch in ("00", "01")]
    res, plots = {}, {}
    for sec in (10, 3):
        segs, spk, dev = [], [], []
        for k in keys:
            s = speech_segments(f"{D}/call{k[0]}_ch{k[1]}.wav", sec); segs += s; spk += [f"c{k[0]}{k[1]}"] * len(s); dev += [k[0] <= 3] * len(s)
        dev = np.array(dev); print(f"{sec} s: {len(segs)} segments", flush=True)
        # classical baseline embeddings (UBM on dev speakers only would differ per fold; use all-speaker UBM here, marked as such)
        feats = {k: B.features(f"{D}/call{k[0]}_ch{k[1]}.wav") for k in keys}
        allf = np.vstack(list(feats.values())); ubm = GaussianMixture(B.NMIX, covariance_type="diag", max_iter=50, random_state=SEED, reg_covar=1e-3).fit(allf[rng.choice(len(allf), 60000, replace=False)])
        sv = normalize(np.array([B.map_supervector(ubm, s) for k in keys for s in B.segments(feats[k], sec)]))
        systems = {"GMM-UBM supervector (all-speaker UBM)": sv}
        for name, path in MODELS.items(): systems[name] = embed_all(path, segs)
        for name, E in systems.items():
            assert E.shape[0] == len(spk), (E.shape, len(spk))
            for sn in (False, True):
                tag = f"{sec} s | {name}{' + S-norm' if sn else ''}"; r = run(E, spk, dev, rng, sn)
                res[tag] = {f: {k: v for k, v in rr.items() if k != "llr"} for f, rr in r.items()}
                if sec == 10: plots[tag] = (np.concatenate([rr["llr"][0] for rr in r.values()]), np.concatenate([rr["llr"][1] for rr in r.values()]))
                print(f"  {tag:60s} " + "  ".join(f"{f}: EER {rr['eer']:.3f} Cllr {rr['cllr']:.2f} min {rr['cllr_min']:.2f}" for f, rr in r.items()), flush=True)
        if sec == 10:
            # labelling check (paper Sec. 5.4): per-channel centroid cosine with the pretrained encoder
            E = systems["ERes2Net (3D-Speaker)"]; sp = np.array(spk)
            C = normalize(np.array([E[sp == f"c{k[0]}{k[1]}"].mean(0) for k in keys])); M = C @ C.T
            cross = sorted(((float(M[i, j]), f"c{keys[i][0]}{keys[i][1]}", f"c{keys[j][0]}{keys[j][1]}") for i in range(10) for j in range(i + 1, 10) if keys[i][0] != keys[j][0]), reverse=True)
            within = [float(normalize(E[sp == s][: (sp == s).sum() // 2].mean(0, keepdims=True)) @ normalize(E[sp == s][(sp == s).sum() // 2:].mean(0, keepdims=True)).T) for s in sorted(set(spk))]
            res["eres2net_centroids"] = dict(top_cross_call_pairs=cross[:4], same_channel_half_vs_half=dict(zip(sorted(set(spk)), within)))
            print("ERes2Net centroid cosines, top cross-call pairs:", cross[:4]); print("  same-channel half-vs-half:", {k: round(v, 2) for k, v in zip(sorted(set(spk)), within)}, flush=True)
    json.dump(res, open(f"{OUT}/e3_plus.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = ["10 s | GMM-UBM supervector (all-speaker UBM)", "10 s | GMM-UBM supervector (all-speaker UBM) + S-norm", "10 s | CAM++ (VoxCeleb) + S-norm", "10 s | ERes2Net (3D-Speaker) + S-norm"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for a, t in zip(ax, pick):
        tippett(*plots[t], a); m = res[t]; a.set_title(t.replace("10 s | ", "") + f"\nmean EER {np.mean([v['eer'] for v in m.values()]):.1%}  Cllr {np.mean([v['cllr'] for v in m.values()]):.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/e3_plus_tippett.png", dpi=150)
