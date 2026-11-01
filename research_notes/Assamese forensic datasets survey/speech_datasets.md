# Public Assamese and Transferable Indic/Multilingual Speech Datasets and Pretrained Models for Forensic Speaker Comparison and Anti-Spoofing

Scope note: compiled 2026-10-31 via direct fetches of primary pages (Hugging Face dataset cards, arXiv, project sites, LDC/LDC-IL catalogues). No general web-search tool was available in this session, so coverage is biased toward sources with known URLs; items that could not be reached are listed under Gaps. Tags: **[ASM]** = Assamese-specific; **[XFER]** = transferable (non-Assamese Indic/multilingual).

## Key Question 1 — Speaker-labelled Assamese speech (IndicVoices, VAANI, LDC-IL, IndicTTS, Kathbath, Common Voice, Shrutilipi, Rasa, IndicVoices-R, SPRING-INX, call-centre corpora, HF long-tail)

### Takeaway
Three sources dominate for many-speaker Assamese audio with speaker metadata: IndicVoices (CC BY 4.0, gated HF; ~684 train + 163 test Assamese speakers in the v1 paper split; mobile 44.1 kHz read/extempore plus 8 kHz telephony-bridge conversations), VAANI (CC BY 4.0, gated HF; 387 h Assamese within 786 h from Assam, 4,217 Assam speakers, district-labelled), and LDC-IL's Assamese Raw Speech Corpus (54 h, 304 speakers, four dialect regions, 48 kHz; free to Indian academic institutions under undertaking, tiered pricing commercially). No Assamese corpus was found that is explicitly multi-session per speaker; Kathbath/IndicSUPERB and Shrutilipi do not contain Assamese. The only Assamese telephone-band resources are the IndicVoices 8 kHz conversational subset and InfoBay AI's call-centre sample (full set is paid/enterprise, no demographic metadata).

### Cited Findings

**IndicVoices (AI4Bharat) [ASM + XFER]**
- HF card: "a total of 23.7K hours of read (8%), extempore (76%) and conversational (15%) audio from 51K speakers" across 22 languages, "11.2K hours have already been transcribed" (update dated 23 Dec 2025); licence CC BY 4.0; repository is gated ("you have to accept the conditions to access its files"); full-length audio also hosted at `https://iv-release.objectstore.e2enetworks.net/dmu_release/v{1–5}_<LANGUAGE>_train.tgz`, each audio accompanied by a `.json` metadata file; Assamese config exists (`load_dataset("ai4bharat/IndicVoices","assamese",split="valid")`) — [HF ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)
- Paper (v1, ACL Findings 2024): 7,348 h, 16,237 speakers, 22 languages, 145 districts; 1,639 h transcribed — [arXiv 2403.01926](https://arxiv.org/abs/2403.01926)
- Assamese split in paper Table 6: Train 684 speakers / 100 h; Test 163 speakers / 5 h; Valid 1 h. Assamese prompt pool: 9,146 read sentences, 2,401 extempore questions, 1,185 role-play scenarios — [arXiv 2403.01926 HTML](https://arxiv.org/html/2403.01926v1)
- Recording: participants' own phones via a modified Microsoft Karya Android app, .wav; single-speaker recordings "over a 44.1kHz channel"; two-party conversations "over a 8kHz channel" via a cloud telephony bridge; devices deliberately varied "to cover a wider range of devices and microphones of varying quality" — [arXiv 2403.01926 HTML](https://arxiv.org/html/2403.01926v1)
- Metadata: registration captured phone, name, age, gender, primary language, proficiency, occupation, location, education; released metadata described as "district id, speaker id, language id, list of entities"; "No PII data will be shared externally"; dialect is not a field; QC checks include "Wrong gender (metadata)" and "Wrong age group (metadata)" — [arXiv 2403.01926 HTML](https://arxiv.org/html/2403.01926v1)
- Sessions: each participant did one collection procedure of 1–4 h with ≥20 min mandated (25–35 min typical); multi-session re-recording is not described — [arXiv 2403.01926 HTML](https://arxiv.org/html/2403.01926v1)
- Licence statement in paper: tools MIT, "the dataset will be released with CC-BY-4.0 license", "allowing commercial usage" — [arXiv 2403.01926 HTML](https://arxiv.org/html/2403.01926v1)

**IndicVoices-R (AI4Bharat) [ASM + XFER]**
- 22 languages, 1,704 h, 10,496 speakers, 9–175 h per language; 48 kHz .wav; CC BY 4.0; gated; 1.05 TB; derived from IndicVoices via HTDemucs demixing → VoiceFixer dereverberation → DeepFilterNet3 enhancement → filtering → volume normalisation; ~93.25% extempore; per-entry metadata JSON "Includes speaker ID, gender, age group, duration, and language" — [HF ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r)

**VAANI (IISc SPIRE Lab / ARTPARK) [ASM + XFER, incl. Nagamese/Sylheti/Karbi]**
- Current release: 31,278.37 h total, 2,122.21 h transcribed, 156,208 speakers, 105 languages, 165 districts, 31 states/UTs; gender 45.63% M / 54.32% F; target ">150,000 hours ... 773 districts"; licence CC-BY-4.0; "exclusively available on Hugging Face" (`ARTPARK-IISc/Vaani`), future distribution via Bhashini; open to "any individual or organization"; contact vaanicontact@gmail.com — [vaani.iisc.ac.in](https://vaani.iisc.ac.in/)
- Assam: 786.32 h, 4,217 speakers, 63.33 h transcribed, 9 languages — Assamese 387.19 h, Bengali 192.83, Karbi 83.71, Hindi 72.31, English 49.22, Sylheti 0.60, Odia 0.22, Nepali 0.13, Sadri 0.10 — [vaani.iisc.ac.in](https://vaani.iisc.ac.in/)
- Neighbouring NE states: Nagaland 467.22 h / 2,643 spk (Nagamese 296.26 h, English 104.14, plus Lotha, Sumi, Angami, Chakhesang, Ao, Rengma); Meghalaya 474.65 h / 2,663 spk (Garo 467.74, plus Nagamese, Hajong, Assamese); Arunachal 645.04 h / 3,147 spk (includes Assamese); Mizoram 218.17 h (incl. Nagamese); Tripura 668.38 h (Chakma 479.99, Bengali 133.58, Kokborok 42.95); Manipur 215.33 h; Sikkim 213.55 h — [vaani.iisc.ac.in](https://vaani.iisc.ac.in/)
- Assamese appears as a single label; Sylheti, Nagamese, Karbi are separate language labels, no Assamese sub-dialect labels — [vaani.iisc.ac.in](https://vaani.iisc.ac.in/)
- HF card: gated (login + accept conditions + token); ~7.21 TB; organised by language subsets plus "images" config; no train/dev/test splits; per-district and per-language tables present but per-row schema (speaker id, gender, age) not documented on card; Assam districts identifiable in the district table: Hailakandi (mostly Bengali, some Assamese, Sylheti), Kamrup Metropolitan (Assamese-dominant), Karbi Anglong (Karbi-dominant), Sonitpur (Assamese-dominant) — [HF ARTPARK-IISc/Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)
- Recording device, sampling rate, sessions per speaker: not stated on either page — [vaani.iisc.ac.in](https://vaani.iisc.ac.in/); [HF ARTPARK-IISc/Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)

**LDC-IL (CIIL Mysuru) Assamese corpora [ASM]**
- Assamese Raw Speech Corpus (SKU LDCIL-68, catalogue 1273, ISBN 978-81-948885-5-0, released 15 Jun 2021): "54:21:12 Hours" (spec table says 54:21:18), "Total Speakers 304 (154 Female and 150 Male)", 37,570 segments, 32.5 GB, "48 kHz | 16 bit wav", "Data Source: On Field"; dialects grouped into "the Eastern Group and the Western Group" with LDC-IL dividing areas into "four regions Xiboxagoria, Central Assam, Kamrupi, Goalparia"; domain breakdown e.g. Contemporary Text (News) 17:23:25, Sentence 7,593 segments; detailed documentation in "AssameseRawSpeechDocumentation.pdf"; access via "Commercial User Undertaking" or "Non-Commercial User Undertaking" — [LDC-IL Assamese Raw Speech Corpus](https://data.ldcil.org/speech/speech-raw-corpus/assamese-raw-speech-corpus)
- Also listed: Assamese Sentence Aligned Speech Corpus (SKU LDCIL-233; description field broken) and Assamese Text to Speech Corpus (SKU LDCIL-256; 44:49:34 h, 2 speakers, 32,594 segments, 28.85 GB) — [LDC-IL speech listing](https://data.ldcil.org/speech)
- Terms (FAQ): commercial tiered pricing — MNC/foreign = base price; non-MNC Indian company 80%; MSME/SAARC 60%; startups/MSMEs with turnover < Rs 5 crore 20%; some datasets free for research scholars and recognised Indian academic/research institutions under Non-Commercial Undertaking (derivatives "should not be used for commercial purposes"; must destroy data after leaving institution); documents (ID + HoD letter on letterhead) posted to Director, CIIL, Mysuru; payment by cheque/DD only, no refunds; delivery of speech data (>2 GB) on physical media: 2–110 GB on 128 GB pen drive Rs 2,000, >110 GB on external HDD Rs 4,500 (within India), 7–10 working days minimum — [LDC-IL FAQ](https://data.ldcil.org/faq)
- Absolute base prices are not shown on any fetched LDC-IL page ("Login for the applicable fee" pattern) — [LDC-IL speech listing](https://data.ldcil.org/speech)
- A third-party HF mirror `bijaykumarsingh/LDCIL_AssameseSentenceAlignedSpeechCorpus` exists (53 downloads/month, no licence tag) — redistribution legitimacy unverified — [HF search API](https://huggingface.co/api/datasets?search=assamese&limit=60&full=false)

**IndicTTS (IIT Madras) Assamese [ASM]**
- HF mirror SPRINGLab/IndicTTS_Assamese: "2 male, 2 female native Assamese speakers", 16,358 rows, 48 kHz WAV, "Studio-quality recordings"; hours stated inconsistently ("~27.4 hours (Male: 5.16 hours, Female: 5.18 hours)"); licence is the IIT Madras "License For Use of Indic TTS" (not a standard open licence; PDF at iitm.ac.in/donlab/indictts/downloads/license.pdf) — [HF SPRINGLab/IndicTTS_Assamese](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Assamese)
- Indic Parler-TTS card lists IndicTTS as 382 h / 12 languages / CC BY 4.0 and gives Assamese training data as 69.78 h / 41,210 utterances (across its component corpora) — [HF ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)
- IITM IndicTTS database page timed out during fetch — [iitm.ac.in/donlab/indictts/database](https://www.iitm.ac.in/donlab/indictts/database)

**Rasa (AI4Bharat) [ASM]**
- All 22 languages, 1,145.44 h, 640,950 utterances, 44 speaker-language pairs (one F + one M per language); Assamese Female 29.09 h / 15,084 utt, Assamese Male 29.23 h / 16,614 utt; neutral + six Ekman emotions + command-style + conversation/news/narration; 48 kHz mono; CC-BY-4.0, gated, ~388 GB — [HF ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa)
- Paper: "10 hours of neutral speech and 1-3 hours of expressive speech for each of the 6 Ekman emotions" for Assamese, Bengali, Tamil — [arXiv 2407.14056](https://arxiv.org/abs/2407.14056)

**Kathbath / IndicSUPERB [XFER — no Assamese]**
- 12 languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu; "1,684 hours of labelled speech data"; "1,218 contributors located in 203 districts"; gated; 174 GB; card metadata tag cc-by-4.0 but text says packaging under CC0 — inconsistent — [HF ai4bharat/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath)

**Shrutilipi [XFER — no Assamese]**
- 12 languages (same list as Kathbath), >6,400 h mined from All India Radio news bulletins; CC BY 4.0; gated; no speaker labels mentioned — [HF ai4bharat/Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi)

**SPRING-INX (IIT Madras SPRING Lab) [ASM]**
- "about 2000 hours of legally sourced and manually transcribed speech data" across Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil; open-sourced under NLTM/MeitY — [arXiv 2310.14654](https://arxiv.org/abs/2310.14654)
- HF SPRINGLab/SPRING_INX_Assamese_R1: 29,530 rows (train 24.6k / val 2.49k / test 2.45k), clips 2–29.7 s, 5.98 GB; empty README; only `utterance_id` (e.g. "KaInDe_utt00000004"), `text`, `audio` — **no speaker column**; transcripts are spontaneous two-party conversational Assamese with heavy Assamese–English code-switching; licence, sampling rate, device not stated — [HF SPRINGLab/SPRING_INX_Assamese_R1](https://huggingface.co/datasets/SPRINGLab/SPRING_INX_Assamese_R1)

**Common Voice Assamese [ASM]**
- A "Mozilla Common Voice 24.0-Assamese corpus" exists and was used to fine-tune Whisper (WER 43.17%, CER 13.18%) — [arXiv 2607.17164](https://arxiv.org/abs/2607.17164)
- HF `mozilla-foundation/common_voice_17_0` repo is now empty: "as of October 2025, Common Voice datasets are only available via Mozilla Data Collective" — [HF common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- Mozilla Data Collective lists 1,207 datasets; Assamese entry not located on the first page — [mozilladatacollective.com/datasets](https://mozilladatacollective.com/datasets)

**Assamese call-centre / telephone corpora [ASM]**
- InfoBay AI dual-channel sample (HF): "It contains only sample data", 33.4 MB; full Assamese catalogue claimed as 58,478 h (and 3.65 M h across 54 languages); 8,000 Hz; dual-channel "separate customer and agent recordings"; schema fields id, audio_filepath, duration, language, sample_rate, format, num_speakers, domain, metadata(source, recording_condition) — **no demographic/speaker-ID fields**; no transcripts; sample licence cc-by-4.0 but gated and "for research and educational purposes only"; data "Procured through formal agreements and generated in the ordinary course of business", PII muted; full set via "enterprise licensing options" (datareq@infobay.ai, +91 9236398619), price not stated — [HF InfoBayAI dual channel](https://huggingface.co/datasets/InfoBayAI/Assamese_Call_Center_Audio_Dataset_Dual_Channel)
- A single-channel InfoBay sample also exists (`InfoBayAI/Assamese-Call-Center-Audio-Dataset-Single-Channel`, cc-by-4.0, gated) — [HF search API](https://huggingface.co/api/datasets?search=assamese&limit=60&full=false)

**Hugging Face long tail (Assamese audio) [ASM]**
- Other Assamese audio datasets found: madhabpaul/assamese_speech_corpus (apache-2.0; 24,212 rows in subsets default/dual_1/female_1/male_1/male_2; no hours, source or sampling rate stated), ananddey/assamese-asr-dataset (apache-2.0, 100K–1M rows, gated), Speech-data/Assamese-Speech-Dataset (cc-by-nc-nd-4.0, <1K rows), SPRINGLab/SPRING_INX_Assamese_R2, saurabh1043/noisy_assamese, dianavdavidson/Vaani-* copies, multiple small TTS train/val sets — [HF search API](https://huggingface.co/api/datasets?search=assamese&limit=60&full=false); [HF madhabpaul/assamese_speech_corpus](https://huggingface.co/datasets/madhabpaul/assamese_speech_corpus)
- HF model search filtered to `language:as` with "speaker" returned an empty array — no Assamese-tagged speaker models on the Hub — [HF models API](https://huggingface.co/api/models?search=speaker&limit=100&filter=language:as)

### Inferences
- For fine-tuning a speaker encoder, IndicVoices Assamese (~850 speakers, many microphones, speaker IDs in metadata) plus VAANI Assam (4,217 speakers, district labels) is the realistic many-speaker pool; both are CC BY 4.0 but gated, and neither documents a per-row schema on its card, so speaker-ID availability must be verified after accepting the gate (IndicVoices-R explicitly confirms speaker ID, gender, age group per entry, which implies the parent IndicVoices metadata carries them too).
- IndicVoices' 8 kHz conversational subset is the only open Assamese telephone-channel speech found; pairing a speaker's 44.1 kHz solo tasks with their 8 kHz conversation would give within-speaker cross-channel trials if speaker IDs are consistent across tasks (unverified).
- LDC-IL's 304-speaker corpus is uniquely dialect-labelled but is studio/field 48 kHz read speech; it suits dialect profiling more than telephone calibration. Indian academic affiliation makes it free; a commercial/foreign requester pays an undisclosed base price plus media fee.
- Session multiplicity is a hole across every Assamese corpus found; synthetic "sessions" would have to be approximated by task/channel diversity within one visit.

### Gaps
- IndicVoices: exact Assamese hours in the current (23.7K h) release, per-row metadata schema, and whether speaker IDs are consistent across the 44.1 kHz and 8 kHz tasks — not stated on card; paper gives only v1 split counts.
- VAANI: sampling rate, recording app/device, per-row metadata fields, full list of Assam districts, speaker-ID availability — not stated on site or card.
- LDC-IL: absolute prices and the text of the undertakings not retrievable (pages 404 or login-gated).
- Common Voice Assamese hours/speakers/validated hours — Mozilla Data Collective listing not reached.
- IndicTTS official database page timed out; Svarah (Indian English) and NPTEL/YouTube-derived Assamese speaker-labelled sets not searched for lack of a web-search tool.
- Shaip, Defined.ai, Appen Assamese catalogues not fetched.

## Key Question 2 — Indic/multilingual speaker-verification corpora for fine-tuning or cohort populations

### Takeaway
IndicSUPERB/Kathbath is the only openly released Indic speaker-ID/ASV benchmark found (12 languages, no Assamese; 1,218 speakers; mobile-recorded; x-vector EERs 2–3% on known speakers but 11–15% on unseen speakers). Telephone-band multilingual SV data (SRE16 Tagalog/Cantonese/Cebuano/Mandarin, CTS Superset) exists only via paid LDC licences. The I-MSV 2022 Indian-language challenge set (100 speakers, 5 sensors, multi-session) is the closest multi-session Indic SV corpus but access is via challenge registration. Bengali resources on HF are diarization sets, not speaker-ID corpora.

### Cited Findings

**IndicSUPERB / Kathbath [XFER]**
- Speakers per language (M/F) in release: Bengali 18/28, Gujarati 44/35, Hindi 58/63, Kannada 53/26, Malayalam 12/20, Marathi 82/61, Odia 10/32, Punjabi 65/77, Sanskrit 95/110, Tamil 116/42, Telugu 53/51, Urdu 36/31; speaker and gender "present in the name of the audio file" (e.g. `844424931537866-594-f.m4a`); audio distributed as m4a; splits Train (~85 GB), Valid, Test Known, Test Unknown plus noisy variants; ASV trial lists in meta_data.zip — [GitHub AI4Bharat/IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB)
- Recording via Microsoft Karya Android app, participants' own phones, remotely — [arXiv 2208.11761 HTML](https://arxiv.org/html/2208.11761v2)
- ASV protocol: x-vector trained on SID data, cosine scoring, "50000 pairs of utterances" per split, 50/50 target/non-target — [arXiv 2208.11761 HTML](https://arxiv.org/html/2208.11761v2)
- ASV EER averages across 12 languages — Test Known: FBANK 3.0, IndicW2V 2.1, XLS-R 2.1; Test Unknown: FBANK 11.2, IndicW2V 15.3, XLS-R 12.1; Bengali: known 2.1/2.0/1.3, unknown 13.3/16.6/10.7 — [arXiv 2208.11761 HTML](https://arxiv.org/html/2208.11761v2)
- SID-mono accuracy avg: FBANK 77.2, IndicW2V 95.6, XLS-R 94.2; SID-multi clean-known: 36.79 / 79.26 / 70.71 — [arXiv 2208.11761 HTML](https://arxiv.org/html/2208.11761v2)
- Authors note FBANK beats SSL features on unknown-speaker ASV and XLS-R generalises better than IndicW2V — [arXiv 2208.11761 HTML](https://arxiv.org/html/2208.11761v2)

**I-MSV 2022 (O-COCOSDA Multi-lingual Speaker Verification Challenge, Indian track) [XFER]**
- Fixed training set "1000 audio recordings spoken by 100 speakers, about 155 hours in total"; development data "collected in multiple sessions using five different sensors"; enrolment recorded "using only a headphone"; test sets have "language and recording device mismatch"; constrained/unconstrained sub-tasks; languages not listed; access procedure not described — [arXiv 2211.16694 HTML](https://arxiv.org/html/2211.16694v2)
- NPU-HC system EERs: 0.223% public, 2.123% constrained / 0.630% unconstrained private — [arXiv 2211.16694](https://arxiv.org/abs/2211.16694)

**NIST SRE telephone corpora [XFER, paid]**
- SRE16 Test Set (LDC2019S20): Tagalog, Cantonese, Cebuano, Mandarin; 220 speakers × 10 calls of 8–10 min; ~340 h; telephone, "range of handsets and acoustic environments"; 8 kHz a-law SPHERE; from Call My Net 2015; LDC User Agreement for Non-Members, 2019 member-year, "Login for the applicable fee" — [LDC2019S20](https://catalog.ldc.upenn.edu/LDC2019S20)
- NIST SRE CTS Superset: ">6800 speakers", telephony segments uniformly 10–60 s, drawn from Greybeard, Switchboard and Mixer (LDC) — [arXiv 2108.07118](https://arxiv.org/abs/2108.07118)
- KASET (LDC2024S01, NIST SRE project): Kurmanji/Sorani Kurdish; 21 speakers each making ≥10 calls; 147 h incl. 289 telephone recordings; CTS two-channel 8-bit μ-law 8 kHz; LDC non-member fee on login — [LDC2024S01](https://catalog.ldc.upenn.edu/LDC2024S01)

**VoxBlink2 (multilingual, in-the-wild) [XFER]**
- "approximately 10M utterances with videos from 110K+ speakers in the wild"; VoxCeleb1-O EER 0.170%; resources at voxblink2.github.io; language breakdown not in abstract — [arXiv 2407.11510](https://arxiv.org/abs/2407.11510)

**Bengali speaker resources [XFER]**
- HF search "bengali speaker": bitwisemind/Bengali_Speaker_Diarization_Dataset (MIT, <1K rows), AdilShamim8/Bengali-Speaker-Diarization_dataset (MIT), psdn-ai/bengali-multi-speaker-speech-samples (gated, licence "other") — diarization/ASR, not speaker-ID corpora — [HF search API](https://huggingface.co/api/datasets?search=bengali%20speaker&limit=30)
- Bengali region-of-origin recognition: 633 speakers, >45 h, 8 Bangladesh divisions, 85.44% accuracy; no public release stated — [arXiv 2404.15168](https://arxiv.org/abs/2404.15168)
- Lipi-Ghor-882: 882-hour multi-speaker Bengali ASR/diarization dataset (2026) — [arXiv 2602.23070 via search](http://export.arxiv.org/api/query?search_query=(all:Bengali+OR+all:Bangla+OR+all:Hindi+OR+all:Indian)+AND+all:speaker+AND+(all:verification+OR+all:recognition)+AND+all:dataset&max_results=25&sortBy=submittedDate)
- DISPLACE 2024: 158 h far-field + 12 h close-field, 5 Indian languages (diarization) — [arXiv 2406.09494 via search](http://export.arxiv.org/api/query?search_query=(all:Bengali+OR+all:Bangla+OR+all:Hindi+OR+all:Indian)+AND+all:speaker+AND+(all:verification+OR+all:recognition)+AND+all:dataset&max_results=25&sortBy=submittedDate)

### Inferences
- Kathbath Bengali (46 speakers, mobile) is the nearest phonological cohort to Assamese that comes with ready ASV trial lists; the 11–15% unknown-speaker EERs reported there are in the same band as the project's zero-shot 2–10% on Assamese, suggesting Indic mobile speech is simply hard for VoxCeleb-style x-vectors without in-domain adaptation.
- For telephone-band cohort/impostor populations, SRE16 is the only multilingual non-English CTS set with a public catalogue entry; cost is undisclosed and requires LDC licensing.

### Gaps
- CTS Superset language list and LDC catalogue ID: the arXiv PDF could not be text-extracted (no pdftotext/pypdf) and the abstract does not list languages.
- I-MSV 2022 languages, sensor list, and data-access route: official evaluation plan not located.
- No "VoxCeleb-Indic" or CN-Celeb-style Hindi/Bengali corpus was found; absence may reflect the missing search tool rather than non-existence.
- SRE18/19 LDC catalogue entries not fetched.

## Key Question 3 — Dialect-labelled Assamese and neighbouring varieties

### Takeaway
LDC-IL's Assamese Raw Speech Corpus is the only corpus found with explicit Assamese dialect-region labels (Xiboxagoria/Eastern, Central Assam, Kamrupi, Goalparia). VAANI provides district labels (Kamrup Metropolitan, Sonitpur, Hailakandi, Karbi Anglong) and separate language labels for Nagamese (~296 h), Sylheti (0.6 h in Assam), Karbi, Hajong and Kokborok rather than dialect tags. The Bodoland University CNN dataset and SLTU 2018 Guwahati LID data could not be located as downloadable resources.

### Cited Findings
- LDC-IL: 304 speakers across "four regions Xiboxagoria, Central Assam, Kamrupi, Goalparia" grouped into Eastern and Western dialect groups — [LDC-IL Assamese Raw Speech Corpus](https://data.ldcil.org/speech/speech-raw-corpus/assamese-raw-speech-corpus)
- VAANI Assam district table includes Hailakandi (Bengali/Assamese/Sylheti), Kamrup Metropolitan (Assamese), Karbi Anglong (Karbi), Sonitpur (Assamese) — [HF ARTPARK-IISc/Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)
- VAANI Nagaland: Nagamese 296.26 h; Meghalaya and Mizoram also list Nagamese; Sylheti 0.60 h in Assam; no Rajbanshi listed under Assam on the fetched page — [vaani.iisc.ac.in](https://vaani.iisc.ac.in/)
- SLTU 2018 papers: "Language Identification of Assamese, Bengali and English Speech" (Chakraborty, Nath, Nirmala, Samudravijaya K); "Dialect Identification Using Tonal and Spectral Features in Two Dialects of Ao" (Tzudir, Sarmah, Prasanna); "Development of Assamese Continuous Speech Recognition System" (Deka, Nirmala, Samudravijaya K) — no dataset release URLs on the index — [ISCA SLTU 2018](https://www.isca-archive.org/sltu_2018/index.html)
- Rhythm study of Mising and Assamese (2024) — no corpus release stated — [arXiv 2410.20095 via search](http://export.arxiv.org/api/query?search_query=all:Assamese+AND+all:speech&max_results=40&sortBy=submittedDate)
- Mendeley Data search for "assamese dialect" returned no relevant speech datasets — [Mendeley Data search](https://data.mendeley.com/research-data/?search=assamese%20dialect)
- arXiv search for Assamese + dialect/speaker/spoof/deepfake returned only two text-only papers; no Assamese dialect-ID speech dataset papers — [arXiv API](http://export.arxiv.org/api/query?search_query=all:Assamese+AND+(all:dialect+OR+all:speaker+OR+all:spoof+OR+all:deepfake)&max_results=25)

### Inferences
- A dialect-profiling model would have to be trained on LDC-IL (four regions, 304 speakers) and validated on VAANI district labels as a proxy (Kamrup Metro ≈ Kamrupi/standard urban; Sonitpur ≈ Central/Eastern; Hailakandi ≈ Barak valley Bengali/Sylheti contact zone). This mapping is the author's inference, not documented by either corpus.

### Gaps
- Bodoland University CNN Assamese-dialect dataset: not found (likely journal-only, no repository located).
- SLTU 2018 Guwahati LID data: paper located but no download/access info.
- Sylheti, Rajbanshi, Goalpariya-specific corpora beyond VAANI: none found.

## Key Question 4 — Anti-spoofing datasets and Assamese-capable TTS/VC for building an AS-Spoof corpus

### Takeaway
No existing anti-spoofing corpus contains Assamese: IndicSynth covers 12 Indic languages without Assamese (CC BY-NC 4.0, xtts_v2 + VITS + FreeVC24 on Kathbath speakers); MLAAD v11 (54 languages) has Bangla 14.1 h, Hindi 22.5 h, Urdu, Tamil, Marathi, Kannada, Malayalam but not Assamese and no bona-fide Indic audio; SEA-Spoof covers Tamil/Hindi + four SEA languages; HAV-DF is Hindi. An Assamese spoof set must be generated: open Assamese-capable generators are MMS-TTS asm (VITS, CC-BY-NC 4.0), Indic Parler-TTS (Apache 2.0, 4 Assamese voices), IndicF5 (MIT, zero-shot voice cloning, Assamese), with FreeVC-style VC usable language-agnostically; XTTS-v2 does not support Assamese.

### Cited Findings

**IndicSynth [XFER]**
- 12 languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu; "4,000+ hours of synthetic audio"; 1,912,665 rows; ~845 GB; bona fide source/target speakers drawn from IndicSUPERB; generators: xtts_v2 (fine-tuned for 10 languages), vits (not fine-tuned), freevc24 (VC, not fine-tuned); Mimicry subset (anti-spoofing) and Diversity subset (deepfake detection); only a "train" split; CC BY-NC 4.0 "non-commercial, academic research only"; ACL 2025 Long Papers pp. 22037–22060, Outstanding Paper Award; authors Sharma, Ekbote, Gupta (IIIT-Delhi) — [HF vdivyasharma/IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth)
- No arXiv preprint found (API query returned 0 results) — [arXiv API](http://export.arxiv.org/api/query?search_query=all:IndicSynth&max_results=5)

**MLAAD (Fraunhofer AISEC) [XFER]**
- v11: 1,153.6 h, 205 TTS models, 54 languages; arXiv versions track dataset releases v1 (17 Jan 2024) → v11 (28 Aug 2026) — [arXiv 2401.09512](https://arxiv.org/abs/2401.09512)
- Language hours include Bangla 14.1, Hindi 22.5, Kannada 9.4, Malayalam 1.8, Marathi 3.0, Tamil 5.0, Urdu 2.1; Assamese, Telugu, Gujarati, Punjabi, Nepali absent; only 8 languages have genuine M-AILABS source audio — South-Asian bona fide audio not included; 205 models / 127 architectures incl. Bark, VITS, XTTS v2, VITS-MMS, F5-TTS, IndicF5, Parler-TTS, ElevenLabs, OpenAI TTS-1 HD; audio 22,050 Hz 16-bit WAV; version hours: v1 63.0 … v10 1002.9, v11 1153.6; cross-lingual evaluation run on v1 only — [arXiv 2401.09512 HTML v11](https://arxiv.org/html/2401.09512v11)
- Licence CC BY-NC 4.0, "Commercial use is not permitted"; gated HF `mueller91/MLAAD`, 183 GB; bona fide from M-AILABS separately; `reference_speaker` field since v8 — [HF mueller91/MLAAD](https://huggingface.co/datasets/mueller91/MLAAD); [deepfake-total.com/mlaad](https://deepfake-total.com/mlaad)

**SEA-Spoof [XFER]**
- Tamil, Hindi, Thai, Indonesian, Malay, Vietnamese; "300+ hours of paired real and spoof speech"; spoofs from "state-of-the-art open-source and commercial systems"; reports "severe cross-lingual degradation" of existing detectors; release location/licence not in abstract; paper CC BY-NC-ND 4.0 — [arXiv 2509.19865](https://arxiv.org/abs/2509.19865)

**HAV-DF and IAB-Rubric deepfake hate-speech set [XFER]**
- HAV-DF: Hindi audio-video deepfakes via faceswap, lip-sync, voice cloning; size/licence not in abstract — [arXiv 2411.15457 via search](http://export.arxiv.org/api/query?search_query=(all:HAV-DF+OR+all:SEA-Spoof+OR+(all:Hindi+AND+all:deepfake+AND+all:audio)+OR+(all:Indian+AND+all:languages+AND+all:spoof))&max_results=25&sortBy=submittedDate)
- Deepfake hate speech (Interspeech 2025): English + Hindi, Bengali, Marathi, Tamil, Telugu; 127,290 paired text/synth-speech samples; at iab-rubric.org/resources — same search result

**ASVspoof series [XFER, English]**
- ASVspoof 2019: VCTK-derived, 107 speakers (46 M / 61 F); train 20 / dev 10 / eval 48 speakers, speaker-disjoint; LA (TTS+VC) 7.12 GB, PA (replay) 16.45 GB; FLAC, "no telephone or mobile codecs applied"; Edinburgh DataShare DOI 10.7488/ds/2555 — [DataShare 10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
- ASVspoof 2021 DF eval set: 34.5 GB, ODC-ODbL v1.0, Zenodo DOI 10.5281/zenodo.4835108; keys from asvspoof.org — [Zenodo 4835108](https://zenodo.org/records/4835108)
- ASVspoof 5: ~2,000 speakers (plus auxiliary ~30,000 for attack speaker encoders), 32 attack algorithms incl. adversarial, seven speaker-disjoint partitions; Zenodo record 14498691, HF mirror jungjee/asvspoof5 — [arXiv 2502.08857](https://arxiv.org/abs/2502.08857)
- asvspoof.org downloads via /database with login — [asvspoof.org](https://www.asvspoof.org/)

**In-the-Wild and Codecfake [XFER]**
- In-the-Wild: 20.8 h bona fide, 17.2 h spoof, 58 politicians/celebrities; Apache 2.0; HF mueller91/In-The-Wild — [deepfake-total.com/in_the_wild](https://deepfake-total.com/in_the_wild)
- Codecfake: English and Chinese, >1 M samples; train 740,747, dev 92,596, test conditions C1–C7 + ALM A1–A3; CC BY-NC-ND 4.0; Zenodo six parts; CSAM countermeasure avg EER 0.616% — [GitHub Codecfake](https://github.com/xieyuankun/Codecfake); [arXiv 2405.04880](https://arxiv.org/abs/2405.04880)

**Assamese-capable generators for an AS-Spoof corpus [ASM]**
- facebook/mms-tts-asm: VITS, 36.3M params, ISO `asm`, CC-BY-NC 4.0, sampling rate in `model.config.sampling_rate` — [HF facebook/mms-tts-asm](https://huggingface.co/facebook/mms-tts-asm)
- ai4bharat/indic-parler-tts: 21 official languages incl. Assamese (also in emotion-prompt set); 4 Assamese voices (Amit, Sita, Poonam, Rakesh); 0.9B params; trained on 1,806 h (GLOBE, IndicTTS, LIMMITS, Rasa); Apache 2.0, gated; Assamese NSS 87.36 ± 1.81; card recommends newer "Indic-Speak" (bodhan-ai) — [HF ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)
- ai4bharat/IndicF5: F5-TTS-based zero-shot voice cloning from reference audio + transcript; 11 languages incl. Assamese; 1,417 h training (Rasa, IndicTTS, LIMMITS, IndicVoices-R); 24 kHz; MIT but gated with condition "only clone voices for which you have explicit permission" — [HF ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5)
- coqui/XTTS-v2: 17 languages, Hindi only Indian language, no Assamese/Bengali; Coqui Public Model License — [HF coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
- MMS-TTS Assamese output has been audited for ATR vowel-harmony fidelity (phonology-informed TTS evaluation, 2026) — [arXiv 2607.01965 via search](http://export.arxiv.org/api/query?search_query=all:Assamese+AND+all:speech&max_results=40&sortBy=submittedDate)

### Inferences
- A minimal AS-Spoof recipe mirroring IndicSynth: bona fide = IndicVoices/VAANI Assamese speakers; TTS attacks = MMS-TTS asm (single voice), Indic Parler-TTS (4 voices), IndicF5 (zero-shot clones of target speakers); VC attack = FreeVC24 (language-agnostic as used in IndicSynth). Passing all outputs through an 8 kHz telephone codec chain would match the project's call-centre channel, which none of the public spoof sets (except ASVspoof 2021 LA/DF compression conditions) model.
- IndicSynth's Bengali subset is the closest available spoof data to Assamese for pre-training a detector before Assamese fine-tuning; its NC licence forbids commercial forensic use.

### Gaps
- Bhashini/ULCA TTS models and Indic-Speak (bodhan-ai): not fetched; licences unknown.
- SEA-Spoof and HAV-DF download locations/licences not stated in abstracts.
- ASVspoof 5 language and codec conditions not in abstract (source corpus widely reported as MLS English but unverified here).

## Key Question 5 — Pretrained speaker encoders and anti-spoofing checkpoints (Indic coverage and cross-lingual behaviour)

### Takeaway
No public speaker encoder trained or fine-tuned on IndicVoices, Kathbath or VAANI was found; all surveyed zoos (3D-Speaker, WeSpeaker, NeMo TitaNet, SpeechBrain ECAPA) are Mandarin- or English-trained, with WeSpeaker's VoxBlink2 SimAMResNet models the only ones labelled "multilingual". The best open anti-spoofing checkpoint is the XLS-R + AASIST model (MIT; EER 0.82% LA / 2.85% DF on ASVspoof 2021), trained on English only; MLAAD results show cross-dataset generalisation is complementary rather than solved.

### Cited Findings
- 3D-Speaker zoo: ERes2Net / ERes2NetV2 / CAM++ / ERes2Net-base checkpoints "trained on a Mandarin dataset of 200k labeled speakers" (`iic/speech_eres2net_sv_zh-cn_16k-common` etc.), plus CN-Celeb and 3D-Speaker models and VoxCeleb SDPN; recipe EERs (Vox1-O / CNCeleb / 3D-Speaker): CAM++ 0.65/6.78/7.75, ERes2Net-large 0.52/6.17/6.34, ERes2NetV2 0.61/6.14/6.52; toolkit Apache 2.0; "No Indic or broader multilingual training data" — [GitHub 3D-Speaker](https://github.com/modelscope/3D-Speaker)
- WeSpeaker pretrained: VoxCeleb (EN) ResNet34/152/221/293, CAM++, ECAPA512/1024, Gemini_DFResNet114, W2V-BERT2.0; CN-Celeb (CN) ResNet34; VoxBlink2 "Multilingual" SimAMResNet34/100 and VoxBlink2-pretrain + VoxCeleb2-finetune variants (pt + onnx); 16 kHz input; VoxCeleb models under CC BY 4.0, others inherit dataset licence — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- NeMo TitaNet-Large: trained on VoxCeleb1/2, Fisher, Switchboard, LibriSpeech, SRE 2004–2010 ("several thousand hours of English speech"); VoxCeleb1 EER 0.66%; ~23M params; 16 kHz mono; CC-BY-4.0 — [HF nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- SpeechBrain ECAPA: VoxCeleb1+2; Vox1 cleaned EER 0.80%; Apache-2.0; 16 kHz — [HF speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- sherpa-onnx speaker-recognition release lists 38 assets, "Each model has its own license"; asset table failed to render — [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)
- No HF models tagged Assamese match "speaker" — [HF models API](https://huggingface.co/api/models?search=speaker&limit=100&filter=language:as)
- IndicSUPERB x-vector ASV on Indic mobile speech: unknown-speaker EER 11–15% avg (see KQ2) — [arXiv 2208.11761 HTML](https://arxiv.org/html/2208.11761v2)
- SSL_Anti-spoofing (Tak et al., Odyssey 2022): wav2vec 2.0 XLS-R 300M front-end + back-end in model.py, RawBoost augmentation; checkpoints `best_SSL_model_LA.pth`, `best_SSL_model_DF.pth` (Google Drive); trained on ASVspoof 2019 LA train; ASVspoof 2021 LA EER 0.82% / min t-DCF 0.2066, DF EER 2.85%; MIT — [GitHub TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
- MLAAD cross-lingual: MLAAD-trained SSL-W2V2 and WhisperDF best on ASVspoof19/21-LA/Voc.v/WaveFake, ASVspoof19-trained best on the other four ("complementary"); MLAAD-trained models "the only ones that do not exhibit performance significantly below 50%" — [arXiv 2401.09512 HTML v11](https://arxiv.org/html/2401.09512v11)
- SEA-Spoof: detectors trained on high-resource languages show "severe cross-lingual degradation"; fine-tuning on in-language spoof data largely recovers performance — [arXiv 2509.19865](https://arxiv.org/abs/2509.19865)

### Inferences
- The project's observation that Mandarin-trained 3D-Speaker/WeSpeaker CN-Celeb encoders transfer better than VoxCeleb ones is consistent with CN-Celeb's harder, more varied-channel recipe EERs (6–8% vs <1% on Vox1-O); the WeSpeaker VoxBlink2 SimAMResNet models are the obvious additional zero-shot candidates to test since they are the only zoo entries labelled multilingual.
- For anti-spoofing, the XLS-R-AASIST checkpoint is the sensible warm-start; the SEA-Spoof and MLAAD findings imply it should be expected to degrade on Assamese until fine-tuned on an in-language spoof set.

### Gaps
- No Indic-trained speaker encoder located (AI4Bharat, IISc, IITM model pages for speaker tasks not found); absence not confirmed exhaustively because no web search was available.
- WavLM-SV (microsoft/wavlm-base-plus-sv) and AASIST (clovaai) licences/results not fetched.
- sherpa-onnx asset list unreadable; VoxBlink2 language distribution and licence not in abstract.
