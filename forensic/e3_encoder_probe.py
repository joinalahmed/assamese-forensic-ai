"""E3 test 2: why do the VoxCeleb-trained encoders fail on Assamese call-centre speech?

Candidate explanations, each with a controlled probe on the same 10 s segments (6→4 fold, EER on eval speakers,
and the fold swapped, mean reported):
  H-band   channel/bandwidth: 3D-Speaker and CN-Celeb include narrowband telephone speech, VoxCeleb is wideband
           YouTube. Probe: band-limit the audio to 300–3400 Hz (telephone band) and to 8 kHz-then-16 kHz resampling.
           If a VoxCeleb model is already "at the floor" it should degrade *less* than the others in relative terms,
           while a bandwidth-sensitive one should degrade more.
  H-noise  SNR: add white noise at 20 / 10 dB. Same logic.
  H-dur    duration: VoxCeleb models may need more context. Probe: 20 s and 30 s segments.
  H-lang   language: cannot be probed without other-language speakers; left as the residual explanation.
Outputs the relative EER change per encoder per condition.
"""
import os, sys, json, glob
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.preprocessing import normalize
import e3_speaker_plus as P
from metrics import eer

D = sys.argv[1] if len(sys.argv) > 1 else "../data/calls"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED, SR = 0, 16000


def telephone(x):
    sos = butter(6, [300, 3400], btype="band", fs=SR, output="sos"); return sosfiltfilt(sos, x).astype(np.float32)


def narrowband(x):
    return resample_poly(resample_poly(x, 1, 2), 2, 1)[: len(x)].astype(np.float32)


def noisy(snr_db, rng):
    def f(x):
        p = (x ** 2).mean(); n = rng.randn(len(x)).astype(np.float32); return (x + n * np.sqrt(p / (10 ** (snr_db / 10)) / (n ** 2).mean())).astype(np.float32)
    return f


def fold_eers(E, spk, dev):
    spk = np.array(spk); out = []
    for m in (~dev, dev):
        X = E[m]; s = spk[m]; S = X @ X.T; iu = np.triu_indices(len(s), 1); same = s[iu[0]] == s[iu[1]]; out.append(eer(S[iu][same], S[iu][~same]))
    return float(np.mean(out))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    keys = [(c, ch) for c in range(1, 6) for ch in ("00", "01")]
    models = {os.path.basename(p).replace(".onnx", ""): p for p in sorted(glob.glob("../models/*.onnx"))}
    res = {}
    conditions = {"clean 10 s": (10, None), "telephone band 300–3400 Hz": (10, telephone), "8 kHz narrowband": (10, narrowband),
                  "white noise 20 dB": (10, noisy(20, rng)), "white noise 10 dB": (10, noisy(10, rng)), "clean 20 s": (20, None), "clean 30 s": (30, None)}
    for cond, (sec, fx) in conditions.items():
        segs, spk, dev = [], [], []
        for k in keys:
            s = P.speech_segments(f"{D}/call{k[0]}_ch{k[1]}.wav", sec); segs += [fx(x) if fx else x for x in s]; spk += [f"c{k[0]}{k[1]}"] * len(s); dev += [k[0] <= 3] * len(s)
        dev = np.array(dev); res[cond] = {"n_segments": len(segs)}
        for n, p in models.items():
            res[cond][n] = fold_eers(P.embed_all(p, segs), spk, dev)
        print(f"{cond:28s} n={len(segs):3d}  " + "  ".join(f"{n.split('_')[0][:3]}-{n.split('_')[2][:8] if 'wespeaker' not in n else n.split('_')[1]+'-'+n.split('_')[3][:6]}: {v:.3f}" for n, v in res[cond].items() if n != "n_segments"), flush=True)
    base = res["clean 10 s"]
    res["relative_change_vs_clean_10s"] = {c: {n: (v - base[n]) / base[n] for n, v in r.items() if n != "n_segments"} for c, r in res.items() if c != "clean 10 s" and "relative" not in c}
    json.dump(res, open(f"{OUT}/e3_encoder_probe.json", "w"), indent=1)
