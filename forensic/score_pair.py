"""Forensic speaker comparison report for one questioned recording against one or more known recordings.

    cd forensic && ../.venv/bin/python -W ignore score_pair.py questioned.wav known1.wav [known2.wav ...] [--out report.json]
    cd forensic && ../.venv/bin/python -W ignore score_pair.py --demo

Uses the backend shipped by e3_backend.py (results/e3_backend_model.npz: encoders, backend, S-norm cohort, logistic
calibrator, validation figures). The JSON report carries what the paper's Sec. 4.6 asks every system to state:
software and model versions with SHA-256 hashes, per-file speech duration and segment count, the raw score, the
calibrated log10 LR with a verbal equivalent on the ENFSI scale, the validation set's EER / Cllr with speaker-bootstrap
confidence intervals and its conditions, and the limitations that follow from how small and homogeneous the
validation corpus is. It does not make a decision; it reports the strength of evidence and its uncertainty.
"""
import os, sys, json, hashlib, platform, subprocess, time
import numpy as np, soundfile as sf
from scipy.signal import stft, resample_poly
from sklearn.preprocessing import normalize
import e3_speaker as B, e3_speaker_plus as P, e3_backend as EB

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "results", "e3_backend_model.npz")
# ENFSI Guideline for Evaluative Reporting in Forensic Science (2015), Appendix: verbal equivalents of the LR.
ENFSI = [(1, "weak support"), (2, "moderate support"), (3, "moderately strong support"), (4, "strong support"), (6, "very strong support"), (np.inf, "extremely strong support")]


def verbal(log10lr):
    direction = "for the same-speaker proposition over the different-speaker proposition" if log10lr >= 0 else "for the different-speaker proposition over the same-speaker proposition"
    if abs(log10lr) < np.log10(2): return "the evidence provides no appreciable support for either proposition (LR ≈ 1)"
    return next(f"{w} {direction}" for lim, w in ENFSI if abs(log10lr) < lim)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def load_wave(path):
    x, sr = sf.read(path, dtype="float32"); ch = 1 if x.ndim == 1 else x.shape[1]; x = x if x.ndim == 1 else x.mean(1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x, sr, ch


def speech_segments(x, sec):
    """Same energy VAD and segmentation as e3_speaker_plus.speech_segments, on an in-memory 16 kHz waveform."""
    _, _, Z = stft(np.append(x[0], x[1:] - 0.97 * x[:-1]), nperseg=B.WIN, noverlap=B.WIN - B.HOP, nfft=512, window="hamming", boundary=None, padded=False)
    loge = np.log((np.abs(Z.T) ** 2).sum(1) + 1e-10); lo, hi = np.percentile(loge, 5), np.percentile(loge, 95)
    frames = np.where(loge > lo + 0.4 * (hi - lo))[0]
    speech = np.concatenate([x[f * B.HOP: f * B.HOP + B.HOP] for f in frames]) if len(frames) else np.zeros(0, np.float32)
    n = sec * 16000
    return speech, [speech[i:i + n] for i in range(0, len(speech) - n + 1, n)]


def snorm_pair(S, Cq, Ck, topk):
    """Adaptive S-norm for a questioned×known score matrix given each side's cohort scores."""
    def stats(C):
        v = np.sort(C, axis=1)[:, -topk:]; return v.mean(1), v.std(1) + 1e-9
    mq, sq = stats(Cq); mk, sk = stats(Ck)
    return 0.5 * ((S - mq[:, None]) / sq[:, None] + (S - mk[None, :]) / sk[None, :])


def load_model(path=MODEL):
    z = np.load(path, allow_pickle=False); meta = json.loads(str(z["meta"])); arrays = {k: z[k] for k in z.files if k != "meta"}
    backend = EB.BACKENDS[meta["backend"]].from_params(arrays); return meta, arrays, backend


def embed_files(meta, files, extractors):
    """Per file: waveform facts + fused embedding per 10 s speech segment."""
    out = []
    for path in files:
        x, sr, ch = load_wave(path); speech, segs = speech_segments(x, meta["segment_seconds"])
        info = dict(file=path, sha256=sha256(path), sample_rate=sr, channels=ch, duration_s=round(len(x) / 16000, 2), speech_s_after_vad=round(len(speech) / 16000, 2),
                    n_segments=len(segs), segment_seconds=meta["segment_seconds"], resampled_to_16k=sr != 16000, mixed_to_mono=ch > 1)
        E = None
        if segs:
            parts = []
            for name in meta["encoders"]:
                ex = extractors[name]; emb = []
                for s in segs:
                    st = ex.create_stream(); st.accept_waveform(16000, s); st.input_finished(); emb.append(np.array(ex.compute(st)))
                parts.append(normalize(np.array(emb)))
            E = normalize(np.hstack(parts)) if meta["fusion"] else parts[0]
        out.append((info, E))
    return out


def compare(questioned, knowns, meta=None, arrays=None, backend=None, model_path=MODEL):
    if meta is None: meta, arrays, backend = load_model(model_path)
    import sherpa_onnx
    mdir = os.path.join(HERE, "..", "models"); extractors, hashes = {}, {}
    for name in meta["encoders"]:
        p = os.path.join(mdir, meta["encoder_files"][name]); hashes[name] = sha256(p)
        extractors[name] = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=p, num_threads=4))
    files = embed_files(meta, [questioned] + list(knowns), extractors); q_info, Q = files[0]; k_infos = [f[0] for f in files[1:]]
    K = [f[1] for f in files[1:] if f[1] is not None]; K = np.vstack(K) if K else None
    try: commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception: commit = None
    val = meta["validation"]; be = val["boot_eval"]; bc = val.get("boot_cal", {})
    rep = dict(
        report_type="forensic speaker comparison, strength-of-evidence report (likelihood ratio); not a decision",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"), propositions=dict(H1="the questioned and known recordings were spoken by the same person", H2="they were spoken by different people"),
        system=dict(encoders=meta["encoders"], encoder_files={n: meta["encoder_files"][n] for n in meta["encoders"]}, encoder_sha256=hashes,
                    encoder_sha256_match_validation={n: hashes[n] == meta["encoder_sha256"][n] for n in meta["encoders"]},
                    fusion="L2-normalised embeddings of the encoders concatenated" if meta["fusion"] else "single encoder",
                    backend=meta["backend"], score_normalisation=f"adaptive S-norm, top-{meta['snorm_topk']} cohort of {len(arrays['cohort'])} training-fold segments" if meta["snorm"] else "none",
                    calibration=f"class-balanced logistic regression on {meta['calibration']}; log LR = {float(arrays['cal_a'][0]):.4f} · score + {float(arrays['cal_b'][0]):.4f} (natural log)",
                    training_data=meta["training"], labelling=meta["labelling"], backend_file=os.path.relpath(model_path, HERE), backend_sha256=sha256(model_path), backend_created=meta["created"],
                    code_sha256={f: sha256(os.path.join(HERE, f)) for f in ("score_pair.py", "e3_backend.py", "e3_speaker_plus.py", "e3_speaker.py", "metrics.py")}, git_commit=commit,
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=__import__("scipy").__version__, sklearn=__import__("sklearn").__version__, sherpa_onnx=sherpa_onnx.__version__,
                                  soundfile=sf.__version__, platform=platform.platform()), versions_at_validation=meta["versions"]),
        questioned=q_info, known=k_infos)
    if Q is None or K is None:
        rep["result"] = dict(status="insufficient speech", detail=f"every file needs at least {meta['segment_seconds']} s of speech after VAD to form one segment; see questioned/known speech_s_after_vad")
        return rep
    S = backend.score(Q, K)
    if meta["snorm"]: S = snorm_pair(S, backend.score(Q, arrays["cohort"]), backend.score(K, arrays["cohort"]), meta["snorm_topk"])
    a, b = float(arrays["cal_a"][0]), float(arrays["cal_b"][0]); raw = float(S.mean()); lnlr = a * raw + b; l10 = lnlr / np.log(10)
    pair_l10 = (a * S + b) / np.log(10); rng_ss, rng_ds = val["validated_log10lr_range"]["same_speaker"], val["validated_log10lr_range"]["different_speaker"]
    inside = min(rng_ss[0], rng_ds[0]) <= l10 <= max(rng_ss[1], rng_ds[1])
    rep["result"] = dict(
        status="ok", n_segment_pairs=int(S.size), raw_score=raw, raw_score_pooling="mean over all questioned × known segment-pair scores",
        raw_score_segment_pairs=dict(median=float(np.median(S)), std=float(S.std()), min=float(S.min()), max=float(S.max())),
        log10_lr=float(l10), lr=float(10 ** l10), ln_lr=float(lnlr), verbal_equivalent_enfsi=verbal(l10),
        segment_pair_log10_lr=dict(median=float(np.median(pair_l10)), share_supporting_same_speaker=float((pair_l10 > 0).mean()), min=float(pair_l10.min()), max=float(pair_l10.max())),
        within_validated_lr_range=bool(inside), validated_log10_lr_range=dict(same_speaker_trials=rng_ss, different_speaker_trials=rng_ds),
        note="The calibrator was validated on single 10 s segment pairs. Pooling by the mean score reports the LR of a typical segment pair; it does not combine the pairs as independent evidence, which would overstate the LR because segments of one recording share channel and session.")
    rep["validation"] = dict(
        corpus="InfoBay call_center_audio_dual_channel_as: 5 Assamese call-centre calls, dual channel (agent / customer), 16 kHz, 10 channels = 9 speakers after merging the repeated agent (round 4)",
        protocol="speaker-disjoint folds; backend and calibration fitted on calls 1–3 (6 speakers), evaluated on calls 4–5; same-speaker trials are distinct 10 s segments of one channel of one call (same session, same channel); different-speaker trials are capped at 12,000 random pairs",
        fold=val["fold"], n_calibration_speakers=val["n_train_speakers"], n_evaluation_speakers=val["n_eval_speakers"], n_same_speaker_trials=val["n_ss"], n_different_speaker_trials=val["n_ds"],
        eer=val["eer"], eer_ci95_speaker_bootstrap=be["eer_ci95"], cllr=val["cllr"], cllr_ci95_speaker_bootstrap=be["cllr_ci95"], cllr_min=val["cllr_min"],
        cllr_ci95_from_resampling_calibration_speakers=bc.get("cllr_ci95"), misleading_evidence_rate=dict(same_speaker_log10lr_below_minus1=val["misleading_ss"], different_speaker_log10lr_above_1=val["misleading_ds"]),
        other_fold_4_to_6=dict(n_calibration_speakers=val["other_fold_4→6"]["n_train_speakers"], n_evaluation_speakers=val["other_fold_4→6"]["n_eval_speakers"], eer=val["other_fold_4→6"]["eer"],
                               eer_ci95=val["other_fold_4→6"]["boot_eval"]["eer_ci95"], cllr=val["other_fold_4→6"]["cllr"], cllr_ci95=val["other_fold_4→6"]["boot_eval"]["cllr_ci95"]),
        n_bootstrap=500, system_selection=meta["selection_rule"],
        robustness={c + (" [clean-10 s system applied to degraded evaluation audio]" if k.startswith("mismatched") else ""): {f: dict(eer=r[f]["eer"], eer_ci95=r[f]["boot_eval"]["eer_ci95"], cllr=r[f]["cllr"]) for f in ("6→4", "4→6") if "skipped" not in r[f]}
                    for c, v in meta["robustness"].items() for k, r in v.items() if k.startswith("mismatched") or c.startswith("clean")})
    rb = meta["robustness"]; dur = ", ".join(f"{s} s: {rb[f'clean {s} s']['matched']['6→4']['eer']:.1%}" for s in (3, 5, 10, 20))
    lim = [
        f"Calibration rests on {val['n_train_speakers']} speakers and validation on {val['n_eval_speakers']} speakers (one evaluation fold). The 95% speaker-bootstrap interval for the EER is {be['eer_ci95'][0]:.1%}–{be['eer_ci95'][1]:.1%} and for Cllr {be['cllr_ci95'][0]:.2f}–{be['cllr_ci95'][1]:.2f}; the point estimates should not be quoted without them.",
        "All validation same-speaker trials come from one session and one channel per speaker. The validation therefore does not cover the within-speaker variability of casework (different sessions, handsets, days, health, speaking style), and same-speaker LRs obtained here are expected to be too high for cross-session pairs; the only cross-session same-speaker pair in the corpus (call 4 ch00 vs call 5 ch00) is the --demo case.",
        f"Validated conditions: Assamese call-centre conversational speech, 16 kHz dual-channel recordings, {meta['segment_seconds']} s of speech per segment after energy VAD. EER on fold 6→4 by speech per segment: {dur}. Band-limiting to the telephone band and 10 dB white noise were tested with the clean-trained system applied to degraded evaluation audio (see validation.robustness); other channels, codecs and noise types are untested.",
        "The speaker encoders were trained on Mandarin corpora (3D-Speaker, CN-Celeb) and used zero-shot on Assamese; no Assamese speaker data was used to train them. Language, dialect, gender and age effects are unmeasured: the corpus has no speaker metadata.",
        f"The system was chosen post hoc among the candidate configurations on the evaluation folds themselves ({meta['selection_rule']}), so its own validation figures are optimistically biased.",
        f"LRs outside the validated range (log10 LR {rng_ss[0]:.1f} to {rng_ss[1]:.1f} for same-speaker and {rng_ds[0]:.1f} to {rng_ds[1]:.1f} for different-speaker validation trials) are extrapolations of the logistic calibrator and have no empirical support.",
        "The trained backends (LDA, PLDA) were worse than cosine on this corpus because 3–6 training speakers cannot estimate a speaker subspace; the shipped backend is " + meta["backend"] + ". The remedy is a many-speaker Assamese corpus (IndicVoices, LDC-IL), not a different model.",
        "No automatic check for recording tampering, overlapping speakers, or non-speech content is performed; the energy VAD is not a speech detector and will pass music or noise bursts.",
    ]
    if any(f["speech_s_after_vad"] < 30 for f in [q_info] + k_infos): lim.insert(0, "At least one file has under 30 s of speech after VAD; the score is pooled over very few segment pairs and is unstable.")
    rep["limitations"] = lim
    return rep


if __name__ == "__main__":
    argv = sys.argv[1:]; out = None
    if "--out" in argv: i = argv.index("--out"); out = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if "--demo" in argv:
        D = os.path.join(HERE, "..", "data", "calls"); meta, arrays, backend = load_model()
        cases = {"same agent across calls (call4_ch00 vs call5_ch00; round-4 relabelling, neither call in the calibration set)": (f"{D}/call4_ch00.wav", [f"{D}/call5_ch00.wav"]),
                 "different speakers (call4_ch01 vs call5_ch01, the two customers)": (f"{D}/call4_ch01.wav", [f"{D}/call5_ch01.wav"]),
                 "same speaker, two known files from one call (call4_ch01 first half vs second half)": None}
        reports = {}
        for name, c in cases.items():
            if c is None:  # split call4_ch01 into halves to show the one-known / two-known interface on same-session data
                x, sr, _ = load_wave(f"{D}/call4_ch01.wav"); h = len(x) // 2; tmp = os.path.join(HERE, "results", "_demo_half")
                sf.write(tmp + "_a.wav", x[:h], 16000); sf.write(tmp + "_b.wav", x[h: 3 * h // 2], 16000); sf.write(tmp + "_c.wav", x[3 * h // 2:], 16000)
                c = (tmp + "_a.wav", [tmp + "_b.wav", tmp + "_c.wav"])
            reports[name] = compare(c[0], c[1], meta, arrays, backend)
            r = reports[name]["result"]; print(f"{name}\n   raw {r['raw_score']:.3f}  log10 LR {r['log10_lr']:+.2f}  → {r['verbal_equivalent_enfsi']}", flush=True)
        for s in ("_a", "_b", "_c"):
            p = os.path.join(HERE, "results", f"_demo_half{s}.wav"); os.path.exists(p) and os.remove(p)
        out = out or os.path.join(HERE, "results", "score_pair_demo.json"); json.dump(reports, open(out, "w"), indent=1, ensure_ascii=False); print(f"written {out}")
    else:
        if len(argv) < 2: sys.exit(__doc__)
        rep = compare(argv[0], argv[1:])
        s = json.dumps(rep, indent=1, ensure_ascii=False)
        if out: open(out, "w").write(s); print(f"written {out}")
        else: print(s)
        if rep["result"]["status"] != "ok": sys.exit(2)
