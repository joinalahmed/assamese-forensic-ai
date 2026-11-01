# Forensic AI for Assamese — experiment reproduction

Rebuilt from scratch from the paper's Section 5 (the original code was lost). Four experiments, one shared
evaluation module. Everything is CPU-only and runs in under ~20 minutes.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash forensic/get_data.sh            # public data from GitHub, ~190 MB
cd forensic && for e in e1_authorship e2_romanisation e3_speaker e4_machine_text; do ../.venv/bin/python -W ignore $e.py; done
```

Outputs (`forensic/results/`): `e1.json … e4.json`, figures `e1_tippett.png`, `e2.png`, `e3.png`, `e4.png`, logs.

| File | What |
|---|---|
| `forensic/metrics.py` | EER, Cllr, Cllr_min (PAV), class-balanced logistic calibration, Tippett plot |
| `forensic/e1_authorship.py` | closed-set attribution, query-length curve, open-set verification (Arambhani poems) |
| `forensic/e2_romanisation.py` | spelling variability + lookup normaliser (AssameseBackTranslit) |
| `forensic/e3_speaker.py` | MFCC → energy VAD → 64-comp GMM-UBM → MAP supervectors → cosine → calibration (call-centre audio) |
| `forensic/e4_machine_text.py` | ChatGPT vs Wikipedia/CC-100 sentences: in-genre, cross-genre, calibrated |

## Results vs. the paper

Numbers are single runs with seed 0; the paper's exact seeds, splits and some preprocessing were not
recoverable, so expect a few points of drift.

| Experiment | Paper | This reproduction |
|---|---|---|
| E1 closed-set, char1–4+word SVM | 76.1% (F1 0.694) | 78.1% (F1 0.721) |
| E1 char1–4 / word / zlib | 71.8 / 66.7 / 54.4% | 73.8 / 68.9 / 55.8% |
| E1 accuracy at 100 / 50 / 30 / 15 words | 63.5 / 58.7 / 47.3 / 33.6% | 71.8 / 63.2 / 51.5 / 37.3% |
| E1 verification, pairwise (EER / Cllr) | 37.3% / 0.93 | 36.2% / 0.92 |
| E1 verification, profile ≥3 poems | 27.3% / 0.81 | 24.4% / 0.75 |
| E2 spellings per word with ≥100 tokens (mean / median) | 22.4 / 18 | 22.6 / 18 |
| E2 lookup normaliser: token / in-vocab / unseen / sentence | 80.6 / 85.6 / 13.5 / 19.1% | 80.5 / 85.7 / 12.9 / 20.2% |
| E3 10 s EER (6→4, 4→6) | 16.3%, 7.6% | 26.0%, 13.0% |
| E3 3 s EER (6→4, 4→6) | 25.2%, 12.2% | 38.8%, 26.2% |
| E3 10 s Cllr / Cllr_min | 1.84, 1.37 / 0.52, 0.23 | 1.78, 1.17 / 0.75, 0.39 |
| E3 repeated-agent finding: call 4 ch00 ≈ call 5 ch00 most similar cross-call pair | cosine 0.271 | 0.283 ✔ |
| E3 re-scoring with those channels merged (10 s, 6→4) | EER 16.7→9.4%, Cllr_min 0.54→0.31 | EER 26.0→26.0%, Cllr_min 0.75→0.74 ✘ |
| E4 in-genre accuracy (Wiki / CC-100) | 97.0 / 97.7% | 96.9 / 96.7% |
| E4 calibrated EER / Cllr | 3.3% / 0.13 | 4.5% / 0.17 |
| E4 cross-genre human flagged as machine (Wiki→CC / CC→Wiki) | 33.6 / 55.7% | 29.4 / 19.9% |

**What reproduced:** every qualitative claim — closed-set ≫ verification, Cllr>1 for the speaker system
despite useful discrimination, the call 4/5 channel-00 similarity, Bengali `র` as the top machine-text tell,
in-genre detection near 97% and degradation across genres, and the lookup normaliser's ~80% ceiling.

**What did not match closely:**
- **E3 discrimination is weaker** than the paper (EER 26/13% vs 16/8% at 10 s). VAD thresholds, UBM training
  size and supervector scaling are unspecified; I tried three supervector variants (all within a few points)
  and kept the standard KL-style scaling rather than tuning toward the paper's numbers.
- **E3 merge check:** the suspected repeated agent is detected as the most similar cross-call pair, but merging
  the two channels does not improve the evaluation fold here, so the paper's "fixes the EER" claim is unconfirmed.
  Also 3 s Cllr is slightly <1 in this run (1.07, 0.86), where the paper had >1; the "Cllr > 1" result holds for 10 s only.
- **E4 cross-genre:** degradation reproduced but the paper's asymmetry (CC→Wiki flagged 55.7%) did not.
- **E1 query-length curve** is flatter than the paper's.

## Improved systems (the `*_plus.py` scripts)

Second pass: for each experiment, the technique the paper itself names as the way forward, implemented and
evaluated under the *same* author-/speaker-disjoint protocol as the baseline. Optional deps: `sherpa-onnx`, `torch`.

```bash
cd forensic && for e in e1_verification_plus e2_normaliser_plus e2_neural e3_speaker_plus e4_machine_text_plus; do ../.venv/bin/python -W ignore $e.py; done
```

| Experiment | Baseline (this repo) | Improved | Technique |
|---|---|---|---|
| E1 verification, profile ≥3 poems, EER / Cllr | 24.4% / 0.77 | **10.8% / 0.43** (fusion of S-normed cosine and SVM); 11.1% / 0.53 (SVM + S-norm); 14.3% / 0.51 (cosine + S-norm) | fused char + word + punctuation/layout streams; author-specific LinearSVC vs. an impostor pool; adaptive S-norm against a development-author cohort; equal-weight score fusion |
| E2 normaliser token / unseen / sentence acc. | 79.7 / 15.2 / 20.2% | 81.6 / 17.0 / 22.1% (Viterbi + bigram LM); **83.2 / 40.0 / 24.4%** (+ char transformer) | noisy-channel Viterbi decoding with an absolute-discount word-bigram LM; 2-layer char-level transformer (d=192, 6 epochs, ~9 min CPU) for unseen tokens, also fed as a candidate for seen tokens |
| E3 10 s EER (6→4, 4→6) / Cllr | 26.0%, 13.0% / 1.78, 1.17 | **10.1%, 2.5% / 0.44, 0.11** | pretrained ERes2Net speaker encoder (3D-Speaker, Mandarin training data) via sherpa-onnx, zero-shot on Assamese telephone speech |
| E3 3 s EER / Cllr | 38.8%, 26.2% / 1.07, 0.86 | **15.1%, 8.3% / 0.53, 0.36** | same |
| E4 cross-genre, human flagged (Wiki→CC / CC→Wiki) | 29.6 / 23.8% (char) | 22.8 / 20.4% (script-normalised char); 28.6 / **14.5%** (script-norm + fused + char-LM) | intake script normalisation (র→ৰ, য→য়, NFC); char-5-gram LM log-perplexity features; mixed-genre training as upper reference: 97.4–97.7% acc, 1.7–1.9% flagged |

Observations:
- **E1:** author-specific models and impostor normalisation each halve the gap the paper identified; the
  paper's "27% EER ceiling with current features" is a feature-and-scoring ceiling, not a data ceiling. A first
  attempt with the impostor pool drawn from the same authors as the development trials gave EER 11.9% but Cllr 1.27:
  the pool must be author-disjoint on the development side too, or the calibrator is fitted to leaked scores.
- **E3:** the pretrained encoder is the single largest gain in the repo and, unlike every classical backend here
  and in the paper, its log-LRs are *calibrated* (Cllr close to Cllr_min) from 4–6 development speakers.
  CAM++ (VoxCeleb) did not transfer (EER 25%/15%); the choice of encoder matters more than the backend.
  S-norm helps the weaker encoders and slightly hurts ERes2Net at 10 s. The encoder also settles the labelling
  question: the call 4 / call 5 channel-00 centroids have cosine 0.92, inside the 0.86–0.97 range of a channel
  compared with its own other half — the same agent, as the paper suspected.
- **E2:** the transformer takes unseen-token accuracy from 15% to 40%; in-vocabulary accuracy (86%) is now the
  ceiling, and most of those errors are genuine ambiguity a word-bigram LM cannot resolve (a sentence-level
  seq2seq model is the next step).
- **E4:** script normalisation removes the Bengali-র shortcut and improves every cross-genre number; the
  perplexity features help CC→Wiki and hurt Wiki→CC, so the genre gap is only partly a feature problem.
  Calibrated under genre mismatch (best config, calibration fitted on the training genre): Cllr 0.84 (Wiki→CC) and
  0.36 (CC→Wiki) against 0.17 in-genre — still a system that should be reported as an investigative aid only.

## Round 3 (`e1_plus2.py`, `e2_neural.py` big config, `e3_encoders.py`, `e4_plus2.py`)

| Experiment | Round 2 | Round 3 | What changed |
|---|---|---|---|
| E1 closed-set accuracy (20 authors) | 78.1% | **82.3%** (F1 0.78) | fused char + word + punctuation/layout streams, LinearSVC C=3 |
| E1 *pairwise* verification (one known poem), EER / Cllr | 36.2% / 0.92 (char cosine) | **26.4% / 0.77** | fused streams + S-norm against an author-disjoint cohort |
| E2 normaliser token / unseen acc. | 83.2 / 40.0% | **83.7 / 47.5%** (unseen 44.9% greedy top-1; sentence 25.4%) | d=256 transformer, 8 epochs, beam-3 candidates fed to the Viterbi decoder |
| E3 10 s EER (6→4, 4→6) / Cllr | 10.1%, 2.5% / 0.44, 0.11 | **9.0%, 2.5% / 0.32, 0.08** | fusion of the three best encoders (selected on development speakers only) + S-norm |
| E3 3 s EER / Cllr | 15.1%, 8.3% / 0.53, 0.36 | **11.8%, 5.9% / 0.41, 0.29** | same |
| E4 cross-genre flagged, Wiki→CC / CC→Wiki | 22.8 / 14.5% (different configs) | 14.5% (style-only view) / 15.4% (LM-only); no single config wins both | content-free views: letter-class "style" view, char-LM perplexity only, strong regularisation, rank ensembles |

Encoder sweep (E3, 10 s, mean of the two folds, no S-norm): 3D-Speaker CAM++ zh-en 7.1%, ERes2Net-base 6.3%,
ERes2Net-v2 6.9%, ERes2Net-large 6.7%, WeSpeaker ResNet34 CN-Celeb 10.2%, WeSpeaker ResNet34 VoxCeleb 18.7%,
WeSpeaker CAM++ VoxCeleb 19.9%. Everything trained on Mandarin corpora transfers to Assamese telephone speech;
both VoxCeleb models do not. Training data, not architecture, decides cross-lingual transfer here, which argues
for IndicVoices / LDC-IL fine-tuning as the next step rather than a bigger encoder.

E4 is a clean negative result: the style-only view (every Assamese letter replaced by its class, so only
punctuation, length and conjunct structure remain) fixes Wiki→CC (85.6% acc, 14.5% flagged) but collapses
CC→Wiki (34.7% flagged); the perplexity-only detector does the reverse. Fusions of the two inherit the worse
direction. The genre gap is not a feature-engineering problem; the paper's data recommendation (AS-Threat with
register coverage) stands.

## Round 4: testing what did not improve (`e3_merge_test.py`, `e3_encoder_probe.py`, `e4_domain.py`)

**1. The paper's merged-channel claim (E3) holds — the GMM-UBM was just too weak to show it.**
On fold 6→4 with calls 4–5 as evaluation speakers, 10 s segments:

| System | c400–c500 cross scores vs same-speaker / other cross-call | False alarms at EER from that one pair | EER as labelled → merged | Cllr |
|---|---|---|---|---|
| GMM-UBM supervector | 0.12 vs 0.15 / 0.06 | 150 / 496 | 26.1% → 25.9% | 1.78 → 1.61 |
| 3D-Speaker CAM++ zh-en | 0.79 vs 0.83 / 0.43 | **202 / 221** | **11.6% → 2.3%** | 0.36 → 0.12 |
| ERes2Net-v2 | 0.90 vs 0.92 / 0.76 | 180 / 197 | 10.4% → 2.0% | 0.40 → 0.15 |
| ERes2Net-base | 0.74 vs 0.79 / 0.47 | 173 / 192 | 10.1% → 3.2% | 0.44 → 0.14 |
| WeSpeaker ResNet34 CN-Celeb | 0.94 vs 0.94 / 0.79 | 251 / 288 | 15.1% → 4.0% | 0.49 → 0.19 |

For every encoder the c400–c500 scores sit inside the genuine same-speaker distribution, and for the good
encoders 90% of the false alarms at the EER point are that single pair, so relabelling removes almost all of
them (the paper reported 16.7% → 9.4%). The GMM-UBM's errors were spread over many pairs (30% from c400–c500),
which is why merging did nothing in the reproduction. Once the agent is relabelled, the best 10 s system is at
**EER 2.0–2.3%, Cllr 0.12–0.15** on real Assamese telephone speech. Corollary for AS-FSC: speaker identity
must be recorded independently of channel role, exactly as Sec. 5.4 argued.

**2. Why the VoxCeleb encoders fail (E3): not bandwidth, not noise, not duration.** Mean EER over both folds, 10 s unless stated:

| Condition | CAM++ zh-en | ERes2Net-base | ERes2Net-v2 | ResNet34 CN-Celeb | **CAM++ VoxCeleb** | **ResNet34 VoxCeleb** |
|---|---|---|---|---|---|---|
| clean 10 s | 7.0% | 6.3% | 6.7% | 10.2% | 19.8% | 18.7% |
| telephone band 300–3400 Hz | 8.8% | 5.8% | 6.8% | 10.1% | 27.0% | 16.3% |
| 8 kHz narrowband | 7.7% | 6.5% | 7.6% | 9.7% | 22.0% | 16.9% |
| white noise 10 dB | 6.9% | 6.7% | 8.0% | 8.7% | 15.0% | 12.9% |
| clean 30 s | 3.7% | 3.7% | 4.4% | 6.2% | 13.4% | 9.5% |

The Mandarin-trained encoders are essentially invariant to band-limiting and noise; the VoxCeleb encoders are not
rescued by any of it — at 30 s they are still 2.5–3.5× worse. The one suggestive result is that *adding* 10 dB
noise improves the VoxCeleb models (19.8 → 15.0%, 18.7 → 12.9%), which is what one expects if their embeddings
carry channel/noise-floor information that the noise masks. The residual explanation is training-corpus domain
(CN-Celeb and 3D-Speaker include telephone and far-field speech; VoxCeleb is wideband YouTube), i.e. the same
"data, not architecture" conclusion as round 3, now with the obvious alternatives ruled out.

**3. The E4 genre gap is closable with target-genre text, and it does not need labels.** Source-genre detector
(script-normalised char 1–4) tested on the other genre; "flagged" = target-genre human sentences called machine.

| Adaptation | Wiki→CC flagged | CC→Wiki flagged |
|---|---|---|
| none (source only) | 17.2% | 16.8% |
| A2 recalibrate only, any k | 12–14% | 12–14% |
| A1 retrain with k = 100 labelled target humans | 8.4% | 6.6% |
| A1 retrain with k = 500 | 4.6% | 2.2% |
| A1 retrain with k = 2500 | 2.2% | 1.0% |
| B1 self-training on 5000 unlabelled target sentences | 11.2% | 9.2% |
| **B2 importance weighting from 5000 unlabelled target sentences** | **6.2%** | **2.7%** |

Recalibration alone plateaus: the detector's *ranking* is genre-biased, not just its threshold. A hundred labelled
sentences from the target platform halve the false-flag rate; 500 bring it near the in-genre level. Most
usefully, importance weighting with only *unlabelled* target-genre text (a genre classifier re-weights the
source training sentences towards target-like ones) gets to 6.2% / 2.7% — comparable to 250–500 labelled
sentences — and that is data an investigator always has. Self-training helps less and is riskier (it trusts the
biased detector's own labels). For the framework: a triage detector should be adapted to the seized collection's
register before use, and the adaptation set and method must be part of the reported validation.

## Round 5: a many-author, multi-session Assamese authorship corpus from Wikipedia (`wiki_authors.py`, `e1_wiki.py`)

The dataset survey (`reports/Assamese forensic datasets survey.md`) found no public Assamese text with ≥100
authors and multiple sessions each; the only legal source is the Assamese Wikipedia revision history (CC BY-SA).
`wiki_authors.py` streams the full history dump (372 MB bz2, 127,716 pages, 602,701 revisions), keeps for every
registered non-bot editor only the *lines they added* in each revision, strips markup, keeps lines that are ≥60%
Assamese script, and groups an editor's contributions into sessions (gap > 6 h). Result: **35,958 sessions from
2,710 editors (median 309 words); 642 editors with ≥3 sessions, 221 with ≥10**; 8,870 sessions touch talk/user
namespaces. Output: `data/new/aswiki/editor_sessions.jsonl` (also the data script is re-runnable; the dump is
mirrored at dumps.wikimedia.your.org since the main host 404s on the file).

`e1_wiki.py` runs the same author-disjoint, two-fold, profile-protocol verification as E1, but every same-author
trial now compares a session with a profile from the editor's *other* sessions — a genuine cross-occasion test the
poetry corpus could not provide. 150 editors per run, ≤12 sessions each, ~1,000 same-author / ~76,000
different-author trials.

| Register | Questioned item | cosine fused | + S-norm | author SVM vs impostors |
|---|---|---|---|---|
| all namespaces (616 eligible editors) | first 75 words | 26.5% / 0.76 | 22.3% / 0.65 | **19.8% / 0.61** |
| all namespaces | full session (median 349 words) | 23.0% / 0.69 | 17.7% / 0.57 | **15.8% / 0.53** |
| talk/user namespaces (202 eligible) | first 75 words | 28.3% / 0.81 | 26.0% / 0.73 | **22.1% / 0.69** |
| talk/user namespaces | full session (median 400 words) | 23.7% / 0.72 | 18.5% / 0.59 | **14.0% / 0.53** |

(EER / Cllr, means of two folds.)

What this adds to the paper's claims:
- **The paper's E1 number generalises.** On 75-word questioned texts from 150 real editors with cross-session
  trials, plain cosine gives 26.5% EER — essentially the paper's 27.3% on 20 poets, so the poetry result was not an
  artefact of a small, homogeneous author set.
- **The same two techniques transfer.** Impostor normalisation and author-specific SVMs take 75-word queries to
  19.8% and full sessions to 14–16% EER, with calibration loss small in every cell (Cllr − Cllr_min ≤ 0.09).
- **The informal slice is not harder once there is enough text.** Talk-namespace sessions are slightly worse at 75
  words (22.1%) but the best at full length (14.0%), consistent with punctuation/layout habits carrying signal that
  short excerpts truncate.
- **Register caveat remains.** Encyclopedic prose and talk pages are not chat messages; Wikipedia editors are literate,
  standard-dialect writers. This corpus is a validation and pre-training resource for AS-Forensic-Text, not a
  substitute for it. Sessions touching namespace 0 and 3 in the same sitting are counted under "talk/user".

## Round 6: pretrained neural models for E1 and E2 (`e1_finetune.py`, `e2_seq2seq.py`)

Hardware note: everything here ran on an M5 Max via PyTorch MPS. Three MPS pitfalls cost most of a day and are
now handled in the scripts: (1) two GPU processes at once thrash each other and leak wired memory; (2) varying tensor
shapes make the caching allocator grow without bound (fixed-shape padding/bucketing and a fixed-shape greedy decoder
fix it); (3) when the battery is nearly empty macOS throttles the GPU ~10× even on AC (`run_when_charged.sh`).

**E1 — contrastive fine-tuning of IndicBERT-v2 on Wikipedia editors.** 466 training editors (25,732 sessions)
disjoint from the 150 evaluation editors of round 5; supervised InfoNCE over batches of 16 editors × 2 sessions,
random 40–220-word crops, 1,500 (v1) and 3,000 (v2) steps; long texts embedded in 150-word chunks and averaged.

| Query | TF-IDF author SVM (round 5) | IndicBERT zero-shot + S-norm | fine-tuned v1 + S-norm | fine-tuned v2 + S-norm | v2 fused with TF-IDF |
|---|---|---|---|---|---|
| 75 words, Wikipedia editors | 19.8% / 0.61 | 22.1% / 0.70 | **16.2% / 0.53** | 16.7% / 0.54 | 16.0% / 0.52 |
| full session, Wikipedia editors | 15.8% / 0.53 | 22.8% / 0.70 | 15.4% / 0.51 | 16.2% / 0.52 | **14.7% / 0.48** |
| poets, 75 words (transfer) | 10.8% / 0.43 (tuned on poets) | 21.6% / 0.67 | 19.5% / 0.60 | 19.7% / 0.61 | 16.4% / 0.54 |
| poets, full (transfer) | — | 20.6% / 0.65 | 17.9% / 0.56 | 18.7% / 0.57 | **14.5% / 0.46** |

(EER / Cllr.) Fine-tuning helps short texts (22 → 16%), and the neural and TF-IDF scores are complementary (fusion
is best everywhere), but doubling the steps did not help — the encoder saturates around 15–16% EER on this corpus.
Unlike E3, there is no pretrained *authorship* encoder to borrow; the contrastive signal from 466 editors is what
limits it, and more editors (not more steps) is the lever. This is the honest ceiling of what public Assamese text
allows today: ~15% EER on 75-word questioned texts with calibration loss under 0.03.

**E2 — sentence-level character transformer: a negative result so far.** A 17M-parameter encoder–decoder
(d=384, 4+4 layers) reading the whole Romanised sentence as characters and writing the whole native sentence,
trained from scratch on the 46k training sentences for 30 epochs (~20k updates, 70 min on MPS). Dev token accuracy
climbed 3% → 22% (epoch 6) → 44% (12) → 60% (21) → **67% (30)** and was flattening; **test token accuracy 67.4%**
(in-vocabulary 70.4%, unseen 27.9%, 18.7% of sentences exact). That is well below the word-level
systems (lookup 80%, Viterbi+LM 82%, per-word char transformer 84%). 46k sentences are not enough to learn
spelling, segmentation and context jointly from scratch; the paper's own from-scratch sentence transformer reached
WER 55 (≈45% word accuracy) on this corpus, so this is consistent, not a bug. What would change it: pre-training the
decoder side as a native-script char LM on Sangraha/IndicCorp (now on disk), or using the sentence model only to
*rerank* the word-level system's n-best lists — the word-level system already has the right answer in its candidate
set far more often than it picks it.

## Round 7: E2 pretrain + rerank (`e2_rerank.py`, `e2_rerank2.py`)

Three things were tried on top of the round-3 word-level system (83.7% token accuracy), all evaluated on the same
12,063-sentence test split; weights tuned on 1,000 dev sentences with the word-level decoder refitted without them.

| Step | Test token acc | Sentence acc | Note |
|---|---|---|---|
| round-3 word-level (Viterbi + in-domain bigram + char transformer beam-3) | 83.7% | 25.4% | starting point |
| + wider lattice (seen top-8, beam-5, copy) → **oracle recall 96.5%** | 83.8% | 25.7% | the right token is almost always *in* the lattice |
| + 113M-token Kneser-Ney bigram (IndicCorp v2 + Wikipedia + Sangraha), interpolated | 84.1% | 26.1% | +0.2: news-domain LM barely helps social-media disambiguation |
| + in-domain trigram, second-order Viterbi | 84.1% | 26.1% | no further gain |
| + 10-best rescoring with the sentence-level seq2seq log-likelihood (β=0.05) | 84.1% | 26.2% | +0.1: 10-best oracle sentence accuracy is only 50% |
| + **Unicode NFC on the native side** | **84.9%** | **28.1%** | +0.8: the corpus mixes precomposed য়/ড়/ঢ় with base + nukta (441 of 37k forms) |

So pre-training a bigger LM and reranking with the sentence model are both near-zero gains, while a one-line Unicode
fix is the largest single improvement of the round. The error analysis explains why: in the *training* data, among
(previous word, Roman token, next word) contexts seen ≥3 times, **20% have conflicting reference readings and 8.2%
of tokens disagree with their own context's majority reading** — e.g. `tu` → টো vs তো, `koi` → কৈ vs কয়, `sir` kept
as `sir` vs ছাৰ, `plz` → please vs plz. No model can learn both answers to the same question; the paper's κ = 0.83
inter-annotator agreement says the same thing. The realistic ceiling on this corpus is therefore ~90–92% token
accuracy, and the remaining 5–7 points are annotation-guideline questions (how to treat Latin words, the
য়/য distinction, টো/তো), not modelling ones. Recommendation for AS-Forensic-Text: fix a normalisation standard
(NFC; explicit keep-Latin rule) before annotation, and double-annotate a 2k-sentence slice to measure the floor.

## Round 8: E2 discriminative reranker (`e2_discrim.py`)

Round 7 ended with a lattice whose oracle token recall is 96.5% but whose decoder picks the right reading 84.9% of
the time, scored by a hand-set emission plus λ × bigram LM. This round replaces that score with a trained one:
every lattice edge (Roman token, candidate, previous word) gets 27 features — direct-model log P(n|r) and raw count,
char-transformer beam log-prob and rank, copy flag, Aksharantar flag and count, in-domain and big-corpus bigram and
unigram log-probs, length ratio, Latin-script and apostrophe flags, and the round-7 decoder's own forward–backward
posterior and 1-best flag — and a classifier trained on the dev lattices (label = candidate equals the reference)
provides the emission; test sentences are decoded with Viterbi over its log-odds. Protocol as in round 7: same
80/20 split and seed, dev = first 2,000 training sentences, NFC on the native side everywhere, and *everything* the
reranker sees on dev is fitted without dev — the word-level decoder, the in-domain bigram, and now also the char
transformer (`NFC=1 NODEV=1 TAG=_big_nodev_nfc e2_neural.py`, 55 min CPU). Model selection is 2-fold by sentence
within dev; the selected model is refitted on all 2,000 dev sentences and applied once to the 12,063 test sentences.

**What could not be run here.** The session's network policy denies `huggingface.co` (and the AI4Bharat object
store and `dumps.wikimedia.org`), so none of the `data/new/download_hf.py` resources were available: **no Aksharantar
lexicon, no Bhasha-Abhijnaanam external test, no IndicCorp/Sangraha big LM.** The script has all three code paths
(Aksharantar readings as extra candidates with flag/count features and an oracle-recall-with/without comparison;
Bhasha-Abhijnaanam decoded with the test-side system; the round-7 113M-token LM via `build_big_lm`) and degrades
to a documented fallback when the files are absent — the "big" LM below is a 2.0M-token Kneser-Ney bigram from the
two corpora `get_data.sh` fetches from GitHub (Wikipedia 100k sentences + the 12 MB CC-100 prefix). Re-running
`e2_discrim.py` on a machine where `download_hf.py` has run fills those columns in; the numbers below are therefore
the *reranker-only* result, and the Aksharantar and external-test questions remain open.

| System (test, 134,793 tokens, 12,063 sentences) | Token | In-vocab | Unseen (7.0%) | Sentence |
|---|---|---|---|---|
| round-7 published (lattice + 113M-token LM + seq2seq 10-best, NFC) | 84.9% | – | – | 28.1% |
| round-7 lattice reproduced here (2M-token fallback LM, transformer trained without dev) | 84.7% | 87.3% | 50.3% | 27.3% |
| + logistic-regression reranker (dev 2-fold: 84.5% vs baseline 84.4%) | not run on test | | | |
| + **gradient-boosted reranker** (dev 2-fold: 84.9%) | **85.3%** | **87.7%** | **53.6%** | **28.3%** |

So the discriminative decoder is worth +0.6 token / +1.0 sentence over the same lattice and +0.4 over the best
published round-7 number, despite a 50× smaller LM and no sentence-level rescoring. The gain is spread thin:
in-vocabulary +0.4, unseen +3.3. Permutation importance is dominated by one feature — the round-7 forward–backward
posterior (0.68 of the log-loss drop) — with the interpolated bigram, neural log-prob, in-domain unigram, copy flag
and direct-model probability each contributing 0.004–0.006; the classifier is essentially re-calibrating the old
decoder's posterior with a handful of corrections, which is why logistic regression (a linear re-weighting of the
same scores) buys almost nothing and the trees buy half a point.

**A leak worth recording.** The first run of this script reported 86.0% on dev and only 85.1% on test, with unseen-
token accuracy jumping 51 → 64% on dev and not moving on test. The cause was the `big_oov` feature: the word-id
dictionary shared between the LM and the in-domain decoders is extended with every training-sentence word, so by the
time the dev lattices were built it also contained the dev *reference* words, and "candidate not in vocabulary"
quietly became "candidate is not the answer". Defining the flag against the big LM's own counts removed the dev/test
gap (dev 84.9 / test 85.3). The pattern — a feature computed from a structure that was fitted on the data it is
evaluated on — is exactly the one the round-7 note about refitting without dev was meant to prevent; it is easy to
reintroduce through a shared mutable dictionary.

**Error analysis (19,780 test errors; `errors.top20` in the JSON).** 77% of all errors are a hypothesis that is
itself an attested training reading of the same Roman token — the model chose a reading that some annotator also
chose. The round-7 inconsistency classes account for: Latin kept vs transliterated **26.1%** (`assam`→অসম vs `assam`,
`sir`→ছাৰ vs `sir`, `bjp`→বিজেপি vs `bjp`), টো/তো **2.6%** (`tu`), and য়/য after NFC only **0.1%** (NFC already settled
it). Two further classes became visible once the big ones were named: the apostrophe that marks /o/ (হল vs হ'ল, হব
vs হ'ব, গল vs গ'ল) **6.3%**, and ি/ী, ু/ূ vowel length **3.1%**. One token alone, the name `dimpu`, produces 949
errors (4.8% of the total) because the corpus writes it three ways (`dimpu`, ডিম্পু, দিম্পু) and the test split
happens to prefer the two the training split does not. The 20 most frequent (Roman, reference, hypothesis) triples:

| Roman | Reference | Hypothesis | n | Class |
|---|---|---|---|---|
| dimpu | dimpu | দিম্পু | 564 | Latin kept |
| dimpu | ডিম্পু | দিম্পু | 325 | other (ড/দ) |
| a | য়ে | এ | 100 | other |
| hol | হল | হ'ল | 96 | apostrophe |
| tu | তো | টো | 89 | টো/তো |
| hoi | হৈ | হয় | 87 | other |
| assam | অসম | assam | 83 | Latin kept |
| hobo | হব | হ'ব | 78 | apostrophe |
| sir | ছাৰ | sir | 77 | Latin kept |
| koi | কয় | কৈ | 71 | other |
| bjp | বিজেপি | bjp | 69 | Latin kept |
| dimpu | দিম্পু | dimpu | 60 | Latin kept |
| nki | নিকি | নেকি | 59 | other |
| asa | আছা | আছে | 59 | other |
| o | ওঁ | ও | 57 | other |
| gol | গল | গ'ল | 54 | apostrophe |
| or | ৰ | অ'ৰ | 52 | other |
| e | য়ে | এ | 51 | other |
| o | অ | ও | 48 | other |
| na | নে | না | 48 | other |

Every row is a guideline question, not a modelling one: whether names and English words stay in Latin script,
whether the /o/ apostrophe is written, ট vs ত in টো/তো, and how the clitics `a`/`e`/`o`/`or` are spelt. The 38% of
errors in the three named classes plus the apostrophe and vowel-length classes would be removed by a normalisation
standard, not by a better reranker, which is consistent with the ~90–92% ceiling estimated in round 7.

Outputs: `forensic/results/e2_discrim.json` (oracle recall with/without Aksharantar, dev selection, test, feature
importances, error table), `e2_discrim.log`, `e2_neural_big_nodev_nfc.{json,log}`. Run order (from `forensic/`):
`EPOCHS=8 D=256 LAYERS=2 BEAM=5 NFC=1 NODEV=1 TAG=_big_nodev_nfc ../.venv/bin/python e2_neural.py` then
`../.venv/bin/python e2_discrim.py` (neural beams are cached to `results/e2_discrim_neural.pkl`; `SMOKE=150` runs the
pipeline on 150 sentences). `e2_neural.py` gained the `NFC` and `NODEV` flags; its default behaviour is unchanged.

## Choices inferred where the paper was silent

- **Poem corpus parsing:** headers appear in three styles (`কবিতা :: t --- a`, `t -- a`, `t - কবি a`). With a
  16-word minimum this gives 853 poems / 181 authors / median 75 words and exactly the paper's 351 poems
  across 20 authors with ≥10 poems (paper states 840 / 178 overall; author-name merging probably explains the gap).
- **E1 baselines:** zlib = extra compressed bytes of the questioned text given each author's training
  profile; NB uses α=0.01 with uniform priors; logistic regression uses C=20 (default C=1 gave 30%).
  These were adjusted after seeing poor first results — flagged so they aren't mistaken for pre-registered settings.
- **E2 variability** counts only native-script word types (Latin words that stay Roman are excluded) with
  lower-cased Roman forms. Most-variable word differs from the paper (মজা, 126 forms, vs বহুত, 110).
- **E3:** each channel is one speaker; same-speaker trials are distinct segments of one channel;
  different-speaker trials are capped at 12,000 random pairs; UBM is fitted on ≤60k development frames.
- **E4:** machine = all 15,497 sentences of `output_100k_split_1.jsonl`; humans are length-matched
  Wikipedia (100K sentence file) and CC-100 (first 12 MB, split on `।`) sentences, one per machine sentence per genre.
  Calibrated test uses a 50/25/25 train/calibration/test split with mixed-genre humans.
