"""E3 test 1: does the paper's merged-channel claim hold with stronger systems?

Paper (Sec. 5.4): call 4 ch00 and call 5 ch00 are probably the same agent; re-scoring fold 6→4 with the two
channels merged reduced 10 s EER 16.7% → 9.4%. The GMM-UBM reproduction saw no change. Hypotheses tested:
  H1  the two channels really are one speaker: their cross-channel segment scores should sit inside the
      genuine same-speaker distribution and far above other cross-call different-speaker pairs, for every encoder;
  H2  with labels as given, those c400–c500 trials dominate the false alarms at the EER operating point, so a
      system has to be discriminative enough for them to *be* the dominant errors before merging helps
      (explains why the weak GMM-UBM did not move);
  H3  merging the labels therefore lowers EER for the neural encoders.
Runs on the 6→4 fold (dev calls 1–3 → eval calls 4–5), 10 s and 3 s, GMM-UBM and every ONNX encoder.
"""
import os, sys, json, glob
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import e3_speaker as B, e3_speaker_plus as P
from metrics import eer, LogisticCalibrator, report

D = sys.argv[1] if len(sys.argv) > 1 else "../data/calls"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0


def analyse(E, spk, dev, rng):
    spk = np.array(spk); Xd, Xe = E[dev], E[~dev]; sd, se = spk[dev], spk[~dev]
    S = Xe @ Xe.T; iu = np.triu_indices(len(se), 1); a, b = se[iu[0]], se[iu[1]]; s = S[iu]
    same = a == b; agent = ((a == "c400") & (b == "c500")) | ((a == "c500") & (b == "c400"))
    other_cross = ~same & ~agent & (np.char.ljust(a, 2) != np.char.ljust(b, 2))
    within_call = ~same & ~agent & (np.char.ljust(a, 2) == np.char.ljust(b, 2))
    # H1: where do the c400–c500 scores sit?
    h1 = dict(same_speaker=[float(s[same].mean()), float(s[same].std())], c400_c500=[float(s[agent].mean()), float(s[agent].std())],
              other_cross_call=[float(s[other_cross].mean()), float(s[other_cross].std())], within_call_pair=[float(s[within_call].mean()), float(s[within_call].std())],
              share_of_c400_c500_above_same_speaker_median=float((s[agent] > np.median(s[same])).mean()))
    # H2: false alarms at the EER threshold, labels as given
    ss, ds = s[same], s[~same]; thr = np.sort(np.unique(np.r_[ss, ds]))
    miss = np.array([(ss < t).mean() for t in thr]); fa = np.array([(ds >= t).mean() for t in thr]); t = thr[np.argmin(np.abs(miss - fa))]
    fa_idx = (~same) & (s >= t); h2 = dict(eer_as_labelled=float(eer(ss, ds)), false_alarms=int(fa_idx.sum()), false_alarms_from_c400_c500=int((fa_idx & agent).sum()),
                                           c400_c500_trials=int(agent.sum()), different_trials=int((~same).sum()))
    # H3: merged labels, with calibration from the dev fold (dev speakers unaffected by the merge)
    same_m = same | agent; Sd = Xd @ Xd.T; iud = np.triu_indices(len(sd), 1); sdm = sd[iud[0]] == sd[iud[1]]
    cal = LogisticCalibrator().fit(Sd[iud][sdm], Sd[iud][~sdm])
    h3 = dict(as_labelled=report(ss, ds, cal), merged=report(s[same_m], s[~same_m], cal))
    for k in ("as_labelled", "merged"): h3[k] = {m: round(v, 4) for m, v in h3[k].items()}
    return dict(H1=h1, H2=h2, H3=h3)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    keys = [(c, ch) for c in range(1, 6) for ch in ("00", "01")]
    models = {os.path.basename(p).replace(".onnx", ""): p for p in sorted(glob.glob("../models/*.onnx"))}
    feats = {k: B.features(f"{D}/call{k[0]}_ch{k[1]}.wav") for k in keys}
    res = {}
    for sec in (10, 3):
        segs, spk, dev = [], [], []
        for k in keys:
            s = P.speech_segments(f"{D}/call{k[0]}_ch{k[1]}.wav", sec); segs += s; spk += [f"c{k[0]}{k[1]}"] * len(s); dev += [k[0] <= 3] * len(s)
        dev = np.array(dev)
        devf = np.vstack([feats[k] for k in keys if k[0] <= 3]); ubm = GaussianMixture(B.NMIX, covariance_type="diag", max_iter=50, random_state=SEED, reg_covar=1e-3).fit(devf[rng.choice(len(devf), 60000, replace=False)])
        systems = {"GMM-UBM supervector (dev UBM)": normalize(np.array([B.map_supervector(ubm, s) for k in keys for s in B.segments(feats[k], sec)]))}
        for n, p in models.items(): systems[n] = P.embed_all(p, segs)
        for name, E in systems.items():
            if E.shape[0] != len(spk): print(f"skip {name}: {E.shape[0]} vs {len(spk)} segments"); continue
            r = analyse(E, spk, dev, rng); res[f"{sec} s | {name}"] = r
            print(f"{sec} s | {name:58s} c400–c500 mean {r['H1']['c400_c500'][0]:.2f} (same {r['H1']['same_speaker'][0]:.2f}, other cross {r['H1']['other_cross_call'][0]:.2f}) | "
                  f"FA from c400–c500 {r['H2']['false_alarms_from_c400_c500']}/{r['H2']['false_alarms']} | EER {r['H3']['as_labelled']['eer']:.3f} → merged {r['H3']['merged']['eer']:.3f}, "
                  f"Cllr {r['H3']['as_labelled']['cllr']:.2f} → {r['H3']['merged']['cllr']:.2f}", flush=True)
    json.dump(res, open(f"{OUT}/e3_merge_test.json", "w"), indent=1)
