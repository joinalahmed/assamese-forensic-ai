"""E3 round 3: which pretrained encoder transfers to Assamese telephone speech, and does fusing them help?

Reuses the segmentation, folds, S-norm and calibration of e3_speaker_plus.py. Systems: every ONNX encoder in
../models, each alone, plus embedding-level fusion (L2-normalised embeddings concatenated) of the best two/three
by development-fold EER (selected on the 6→4 fold's *development* speakers only, i.e. calls 1–3, never on the
evaluation speakers).
"""
import os, sys, json, glob
import numpy as np
from sklearn.preprocessing import normalize
import e3_speaker_plus as P
from metrics import tippett

D = sys.argv[1] if len(sys.argv) > 1 else "../data/calls"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0


def dev_eer(E, spk, dev):
    """EER among development speakers only (calls 1-3), used for model selection."""
    Xd = E[dev]; S = Xd @ Xd.T; ss, ds = P.trials_from_matrix(S, list(np.array(spk)[dev]), np.random.RandomState(SEED))
    from metrics import eer; return eer(ss, ds)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    keys = [(c, ch) for c in range(1, 6) for ch in ("00", "01")]
    models = {os.path.basename(p).replace(".onnx", ""): p for p in sorted(glob.glob("../models/*.onnx"))}
    res, plots = {}, {}
    for sec in (10, 3):
        segs, spk, dev = [], [], []
        for k in keys:
            s = P.speech_segments(f"{D}/call{k[0]}_ch{k[1]}.wav", sec); segs += s; spk += [f"c{k[0]}{k[1]}"] * len(s); dev += [k[0] <= 3] * len(s)
        dev = np.array(dev); emb = {}
        for name, path in models.items():
            emb[name] = P.embed_all(path, segs); print(f"{sec} s  {name}: dev-speaker EER {dev_eer(emb[name], spk, dev):.3f}", flush=True)
        order = sorted(emb, key=lambda n: dev_eer(emb[n], spk, dev))
        systems = {n: emb[n] for n in order}
        systems["fusion top-2 (dev-selected)"] = normalize(np.hstack([emb[n] for n in order[:2]]))
        systems["fusion top-3 (dev-selected)"] = normalize(np.hstack([emb[n] for n in order[:3]]))
        res[f"{sec} s dev-selection order"] = order
        for name, E in systems.items():
            for sn in (False, True):
                tag = f"{sec} s | {name}{' + S-norm' if sn else ''}"; r = P.run(E, spk, dev, rng, sn)
                res[tag] = {f: {k: v for k, v in rr.items() if k != "llr"} for f, rr in r.items()}
                if sec == 10: plots[tag] = (np.concatenate([rr["llr"][0] for rr in r.values()]), np.concatenate([rr["llr"][1] for rr in r.values()]))
                print(f"  {tag:75s} " + "  ".join(f"{f}: EER {rr['eer']:.3f} Cllr {rr['cllr']:.2f} min {rr['cllr_min']:.2f}" for f, rr in r.items()), flush=True)
    json.dump(res, open(f"{OUT}/e3_encoders.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    best = min((k for k in plots), key=lambda k: np.mean([v["eer"] for v in res[k].values()]))
    fig, ax = plt.subplots(figsize=(4.5, 3.4)); tippett(*plots[best], ax); m = res[best]
    ax.set_title(best + f"\nmean EER {np.mean([v['eer'] for v in m.values()]):.1%}  Cllr {np.mean([v['cllr'] for v in m.values()]):.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/e3_encoders_tippett.png", dpi=150)
