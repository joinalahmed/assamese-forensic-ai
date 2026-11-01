# Machine-generated Assamese text: detection datasets, generation resources, and genre-diverse human corpora

Scope note: research conducted 2026-10-31 via arXiv (abstract + HTML), Hugging Face Hub API/dataset cards, GitHub READMEs, Semantic Scholar API (rate-limited; several queries failed), and shared-task pages. No general web search tool was available, so coverage of non-indexed sources (Kaggle, newspaper sites, government portals) is weaker. Each entry is tagged **[AS]** = Assamese-specific, **[BN]** = Bengali (script-sharing, transferable with ৰ/ৱ caveat), **[IND]** = pan-Indic incl. Assamese, **[ML]** = multilingual, no Assamese/Bengali.

---

## KQ1. Multilingual machine-generated-text (MGT) detection benchmarks: do any include Assamese or Bengali?

### Takeaway
No public MGT-detection benchmark found contains Assamese; none of the major multilingual benchmarks (M4/M4GT-Bench, SemEval-2024 Task 8, MULTITuDE, COLING-2025 GenAI Detection Task 1) contains Bengali either. The only Indic coverage is Urdu (M4GT-Bench, SemEval-2024, COLING-2025) and a 1,200-text Hindi test-only slice (COLING-2025), plus 2025–26 single-language studies for Bengali (paraphrase detection, unreleased), Urdu (3,600 texts) and Hindi/Telugu/Tamil (IndicDetect). Cross-lingual results in these benchmarks show the exact failure mode the project observes: near-chance accuracy on held-out languages/genres.

### Cited Findings
**M4GT-Bench [ML, Urdu only] — ACL 2024**
- Multilingual portion (Table 2) totals 28,000 human + 37,100 machine = 65,100 texts across Arabic (Wikipedia 3k/3k ChatGPT; news 1k/1k ChatGPT + 100 Jais-30B), Bulgarian (true & fake news, 3k human / 3k davinci-003 + 3k ChatGPT), Chinese (Baike/Web QA, same split), Indonesian (id_newspapers_2018, 3k/3k ChatGPT), Russian (RuATD, 3k / 3k+3k), Urdu (Urdu-news, 3k human / 3k ChatGPT), Italian (CHANGE-it news, 3k / 3k camoscio-70B i.e. LLaMA-2-70B fine-tune), German (news 3k/3k ChatGPT; Wikipedia 3k/3k ChatGPT). Hindi, Bengali and Assamese are absent — [arXiv 2402.11175 HTML](https://arxiv.org/html/2402.11175v2)
- Generators (9): davinci-003, ChatGPT (GPT-3.5-Turbo), GPT-4, Cohere, Dolly-v2, BLOOMz, Jais-30B, LLaMA-2 (7B/13B/70B), camoscio-70B. English domains: Wikipedia, WikiHow, Reddit ELI5, arXiv abstracts, PeerRead reviews, OUTFOX student essays; non-English adds news and Baike/Web QA — [arXiv 2402.11175 HTML](https://arxiv.org/html/2402.11175v2)
- Licences are per-source (CC BY-SA-3.0 Wikipedia sources; MIT Bulgarian/Chinese; CC BY-NC-SA-4.0 Indonesian/Italian/German news; Apache-2.0 RuATD and Arabic news; CC BY 4.0 Urdu-news); the COLING-2025 overview records M4GT-Bench as a whole as CC BY-SA-4.0 — [arXiv 2402.11175 HTML](https://arxiv.org/html/2402.11175v2); [arXiv 2501.11012 HTML](https://arxiv.org/html/2501.11012v2)
- Cross-lingual result (XLM-R trained on all other languages): mixed split 94.5% accuracy but held-out Bulgarian 52.7%, Russian 53.7%, Indonesian 55.8% (near chance); Urdu 94.4%, Arabic 92.2%; the detector over-predicts "machine". Cross-generator (English): RoBERTa 99.3% random split → 60.3% on unseen BLOOMz. Cross-domain attribution: RoBERTa 97.0% in-domain → 36.6% on unseen arXiv — [arXiv 2402.11175 HTML](https://arxiv.org/html/2402.11175v2)
- Data hosted via Google Drive link from the repo; repo has no LICENSE file — [GitHub mbzuai-nlp/M4GT-Bench](https://github.com/mbzuai-nlp/M4GT-Bench)

**SemEval-2024 Task 8 [ML, Urdu only]**
- Subtask A multilingual: train = English, Chinese, Urdu, Bulgarian, Indonesian (76,863 machine / 80,994 human); dev = Russian, Arabic, German (2,000/2,000); test = English, Arabic, German, Italian (22,100 machine / 20,200 human), Italian as surprise language. Urdu: 2,899 ChatGPT + 3,000 human, train only. Hindi/Bengali absent — [arXiv 2404.14183 HTML](https://arxiv.org/html/2404.14183)
- Generators: davinci-003, ChatGPT, LLaMA2 (Italian), Jais (Arabic, 100 samples), Cohere, Dolly-v2, BLOOMz; GPT-4 surprise generator in monolingual track — [arXiv 2404.14183 HTML](https://arxiv.org/html/2404.14183)
- Best multilingual systems: USTC-BUPT 95.99 acc, FI Group 95.85, KInIT 95.00; baseline 80.89 — [arXiv 2404.14183 HTML](https://arxiv.org/html/2404.14183)
- GitHub repo for the task returned HTTP 451 (unavailable for legal reasons) at fetch time — [GitHub mbzuai-nlp/SemEval2024-task8](https://github.com/mbzuai-nlp/SemEval2024-task8)

**MULTITuDE [ML] — EMNLP 2023**
- 74,081 authentic + machine texts in 11 languages (ar, ca, cs, de, en, es, nl, pt, ru, uk, zh) from 8 multilingual LLMs; no Indic language; paper CC BY 4.0; dataset licence recorded as GPL-3.0 in the COLING-2025 overview — [arXiv 2310.13606](https://arxiv.org/abs/2310.13606); [arXiv 2501.11012 HTML](https://arxiv.org/html/2501.11012v2)

**GenAI Content Detection Task 1, COLING 2025 [ML, Urdu + Hindi test-only]**
- Train/dev (15 languages: ar, bg, ca, zh, cs, nl, en, de, id, it, pt, ru, es, uk, ur) = HC3 + M4GT-Bench + MAGE merged, deduplicated, split 7:3: train 257,968 human / 416,115 MGT; dev 110,166 / 178,728; dev-test = MULTITuDE 7,992 / 66,089; test 73,634 / 77,791 (151,425) across 15 languages with six unseen (Hebrew, Hindi, Japanese, Kazakh, Norwegian, Vietnamese). Total 1,188,483 — [arXiv 2501.11012 HTML](https://arxiv.org/html/2501.11012v2)
- Indic: Urdu in train via M4GT Urdu-news (train 2,085 human / 1,676 machine; dev 853 / 720) and test News 13,497 human / 17,472 MGT (GPT-4o-2024-08-06). Hindi test-only News 600 human / 600 MGT (GPT-4o-2024-08-06). Bengali and Assamese not included — [arXiv 2501.11012 HTML](https://arxiv.org/html/2501.11012v2)
- Test generators: GPT-4o variants, GPT-4o-mini, GPT-4-0613, GPT-3.5, Claude-3.5-sonnet, GLM-4-9b, ChatGLM3-6B, Qwen/Qwen-2.5 72B, Baichuan, Ace-GPT, Vikhr-Nemo-12B, Llama 3.1 405B, dictalm2.0, LeoLM, normistral-7b, T-lite — [arXiv 2501.11012 HTML](https://arxiv.org/html/2501.11012v2)
- Results: winner Grape 79.16 macro-F1; XLM-R baseline 74.16 (only 7 teams beat it). Best Urdu accuracy 68.7; best Hindi only 51.8 ("the top-performing team achieved only 51.8 on Hindi") — [arXiv 2501.11012 HTML](https://arxiv.org/html/2501.11012v2)
- Data: HF `Jinyan1/COLING_2025_MGT_multingual` (962,977 rows train+dev, 913 MB parquet, 43 `model` classes, 9 `lang` classes, README empty, no licence tag) and Google Drive; repo code Apache-2.0, dataset licence not stated — [HF Jinyan1/COLING_2025_MGT_multingual](https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_multingual); [GitHub](https://github.com/mbzuai-nlp/COLING-2025-Workshop-on-MGT-Detection-Task1)

**Indic-language single-language studies (2025–2026)**
- IndicDetect (arXiv 2608.29919, v2 Oct 2026): Hindi, Telugu, Tamil; "highly curated human-written texts matched with LLM-generated counterparts across various domains and generators"; tests domain shift, generator shift, adversarial perturbation; supervised neural detectors strong in-distribution, training-free methods "degrade considerably under unseen generators and adversarial attacks"; Hindi shows largest adversarial degradation. Sizes, generators and release URL not in abstract — [arXiv 2608.29919](https://arxiv.org/abs/2608.29919)
- Urdu case study (arXiv 2510.16573): 1,800 human + 1,800 AI texts from Gemini, GPT-4o-mini, Kimi AI; no release URL in abstract — [arXiv listing via export API](http://export.arxiv.org/api/query?search_query=%28all:hindi+OR+all:indic+OR+all:urdu+OR+all:marathi+OR+all:%22indian+languages%22%29+AND+%28all:%22machine-generated+text%22+OR+all:%22AI-generated+text%22+OR+all:%22LLM-generated+text%22%29&max_results=40&sortBy=submittedDate)
- Bengali [BN]: "Detecting AI-Generated Paraphrases in Bengali" (arXiv 2512.21709, ICCIT 2025): human vs AI-paraphrased Bengali; zero-shot XLM-R-Large / mDeBERTaV3 / BanglaBERT / IndicBERT / mBERT ≈50% accuracy; fine-tuned XLM-R, mDeBERTa, mBERT ≈91% accuracy/F1; IndicBERT lags. Dataset size, generating LLM, and release not stated in abstract — [arXiv 2512.21709](https://arxiv.org/abs/2512.21709)
- Bengali [BN]: "Detection of Bangla AI-Generated Texts Using a Quantum Kernel SVM", Applied AI Letters 2026, DOI 10.1002/ail2.70051, claims "first application of a quantum-kernel SVM to detecting AI-generated" Bangla; dataset details not retrieved (paywalled redirect) — [Semantic Scholar record](https://api.semanticscholar.org/graph/v1/paper/search?query=Bangla+AI-generated+text+detection)
- Bengali LLM-output corpora that are *not* detection sets but contain released machine Bengali: CAP (arXiv 2510.22395) — 900 questions, >7,000 answers from 16 open models in 9 languages incl. Bengali, with hallucination/fluency labels and token logits — [arXiv export](http://export.arxiv.org/api/query?search_query=all:bengali+AND+%28all:%22machine-generated%22+OR+all:%22AI-generated%22+OR+all:%22LLM-generated%22%29&max_results=30)
- English-only benchmarks (RAID, MGTBench, Ghostbuster) were not re-verified here; the COLING-2025 overview's source table lists only HC3/M4GT/MAGE/MULTITuDE as multilingual sources, none with Bengali/Assamese — [arXiv 2501.11012 HTML](https://arxiv.org/html/2501.11012v2)

### Inferences
- The project's Assamese detector cannot be validated against any existing MGT benchmark; the closest transferable assets are M4GT Urdu-news (ChatGPT, news genre), COLING-2025 Hindi/Urdu news (GPT-4o) and the Bengali paraphrase study design (paraphrase of human text = prompt-matched pairs).
- The COLING-2025 Hindi result (51.8% on an unseen language with a strong XLM-R-based system) and M4GT held-out-language results (52–56%) corroborate that genre/language confounds dominate and that unlabelled target-domain adaptation (as the project uses) is a reasonable response.
- A released Assamese MGT benchmark would be novel; publishing one (even a few thousand prompt-matched pairs across genres) fills a documented gap.

### Gaps
- Could not obtain the M4GT-Bench/SemEval dataset licence as a single statement (per-source only).
- Bengali paraphrase-detection dataset (2512.21709) and Bangla QSVM paper: sizes, generators, and availability unknown (abstract-only / paywalled).
- Semantic Scholar rate limits (HTTP 429) blocked several targeted queries (Assamese hate-speech dataset, Bangla MGT dataset); an "Indic MGT" dataset from IIIT/IIT groups was not found via arXiv — may not exist publicly as of Oct 2026.

---

## KQ2. Assamese LLM-generated or synthetic text already released

### Takeaway
Released machine Assamese is almost entirely *machine-translated* (IndicTrans2) rather than natively LLM-generated: Sangraha Synthetic (11.7B Assamese tokens of IndicTrans2-translated English Wikimedia), IndicAlign (Llama2-70B-Chat English responses translated by IndicTrans2), Alpaca/GPTeacher Assamese translations of unknown provenance. The only confirmed natively LLM-generated Assamese set is the project's existing indian-nlp ChatGPT word-prompted file (~10k lines per the paper). Aya Collection has no Assamese split.

### Cited Findings
**indian-nlp/assamese-dataset [AS] — the project's current machine source**
- GitHub repo (MIT licence, 17 commits, linked to arXiv 2410.11291): ChatGPT (version unspecified in README; paper says "ChatGPT 3.5") sent each word from an Assamese Wikipedia v2 word list (1/10 split) "along with a prompt... to generate valid dataset that describes that word"; outputs `output_100k_split_1.jsonl` (word, sentence) and `output_100k_split_1_chat.jsonl` (chat-style `text`) — [GitHub indian-nlp/assamese-dataset](https://github.com/indian-nlp/assamese-dataset)
- Paper describes the ChatGPT set as "10,000 lines of JSONL file, with 'word' and 'sentence' keys" (the project's 15k count is not contradicted but not confirmed by the paper) — [arXiv 2410.11291 HTML](https://arxiv.org/html/2410.11291v2)
- Human portions: Assamese Wikipedia 2021 dump (10k and 100k document versions), CC-100 Assamese (7.6 MB download), Kaggle Assamese Sentiments (Ritik Kumar Jain), Assamese Poem collection (Sani Kamal, scraped from arambhani.com) — [GitHub indian-nlp/assamese-dataset](https://github.com/indian-nlp/assamese-dataset); [arXiv 2410.11291 HTML](https://arxiv.org/html/2410.11291v2)
- The HF mirror `indian-nlp/assamese-dataset` returns HTTP 401 and an empty API record (gated/private or removed) — [HF API](https://huggingface.co/api/datasets/indian-nlp/assamese-dataset)

**AI4Bharat Sangraha — Synthetic split [IND]**
- Assamese token counts (millions): Verified 292.1 (Web 128.6, PDFs 162.9, Speech 0.6), Synthetic 11,696.4, Unverified 17.5, total 12,006.0 — [HF ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha); [arXiv 2403.06350 HTML](https://arxiv.org/html/2403.06350)
- Synthetic = all English Wikimedia translated into 14 Indic languages with IndicTrans2 via Setu-Translate (~90B tokens), then romanised with IndicXlit (~72B); no quality filtering applied to translated output (perplexity filtering applies only to Unverified) — [arXiv 2403.06350 HTML](https://arxiv.org/html/2403.06350)
- Assamese synthetic appears as `asm_Beng` and `asm_Latn` splits of ~5.63M rows each; whole dataset CC BY 4.0, parquet, ~705 GB, not gated — [HF ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)

**AI4Bharat IndicAlign [IND]**
- CC-BY-4.0, ~97.4M rows / 28.2 GB; Assamese listed among 15 language tags. Subsets: Indic-ShareLlama (~21k; ShareGPT first prompts, responses by Llama2-70B-Chat), Dolly-T (15k), OpenAssistant-T (19.9k), WikiHow (20.3k), IndoWordNet (~74–97M), Anudesh (36.8k), Wiki-Conv (~141–144k, ~9 turns), Wiki-Chat (~198–202k), HHRLHF-T (~32.6k), Toxic-Matrix (~90.3k; Mistral-7B-Instruct toxic prompts + Llama2-70B refusals). "We use IndicTrans2 ... for the translation of the datasets" — [HF ai4bharat/indic-align](https://huggingface.co/datasets/ai4bharat/indic-align)
- Responses were generated natively in **English** by Llama2-70B-Chat (Wiki-Chat also uses Mixtral-8x7B in a multi-agent simulation) and then translated/transliterated into 14 languages with Setu-Translate/IndicTrans2; Anudesh uses a translate-test loop — [arXiv 2403.06350 HTML](https://arxiv.org/html/2403.06350)

**Alpaca / instruction translations [AS]**
- `Telugu-LLM-Labs/assamese_alpaca_yahma_cleaned_filtered`: 28,910 rows, parquet, 45.6 MB; parallel English + Assamese instruction/input/output columns; README empty; **translation model not stated**; no licence; used to train Indic-gemma Navarasa-2.0 — [HF](https://huggingface.co/datasets/Telugu-LLM-Labs/assamese_alpaca_yahma_cleaned_filtered)
- `Tensoic/GPTeacher-Assamese`: 43,654 rows, 81.5 MB JSON; no dataset card, no licence, creation method undocumented — [HF](https://huggingface.co/datasets/Tensoic/GPTeacher-Assamese)
- `saillab/alpaca_assamese_taco` and `saillab/alpaca-assamese-cleaned` (10K–100K rows each, language tag `as`) exist; cards not inspected — [HF API search](https://huggingface.co/api/datasets?search=assamese_alpaca&limit=50)
- `ananddey/assamese-sft-dataset-v1`: 61,929 rows (58,833 train / 3,096 val), Apache-2.0; aggregation of the Telugu-LLM-Labs Alpaca set + IndicAlign Dolly_T + OpenAssistant_T, filtered for >20% Assamese-script ratio, MD5-deduplicated (1,706 duplicates removed); preview shows translation artefacts (stray spaces around apostrophes) — [HF](https://huggingface.co/datasets/ananddey/assamese-sft-dataset-v1)
- `pritamdeka/stsb-assamese-indictrans2-rope` and `pritamdeka/stsb-assamese-translated` (1K–10K): STS-B translated to Assamese (IndicTrans2 named in one id) — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)
- `murkho/MedQA_tran_COT_Assamese`, `iit-patna-cse-ai/MedQA_tran_COT_Assamese_train` (1K–10K): translated MedQA with chain-of-thought — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)

**Aya [IND/ML]**
- `CohereLabs/aya_collection_language_split` has 132 configs; only `bengali` matches — **no Assamese config** — [HF datasets-server splits](https://datasets-server.huggingface.co/splits?dataset=CohereLabs/aya_collection_language_split)
- `CohereLabs/aya_dataset` full-text search for "Assamese" returned 0 rows — [HF datasets-server search](https://datasets-server.huggingface.co/search?dataset=CohereLabs/aya_dataset&config=default&split=train&query=Assamese&offset=0&length=2)

**Other synthetic Assamese (2026)**
- IndicMedDialog (BioNLP@ACL 2026): LLM-generated synthetic medical consultations extending MDDial, translated with TranslateGemma into Assamese, Bengali, Gujarati, Hindi, Marathi, Punjabi, Tamil, Telugu, Urdu, native-speaker verified with "script-aware post-processing"; generating LLM, size and release URL not in abstract — [arXiv 2605.13292](https://arxiv.org/abs/2605.13292)
- `promptmafiainc/assamese-bengali-contrastive-eval`: paper+code release ("Making Large Language Models Speak Assamese: Dual-Script Structured Prompting and Phonemic Anchor-and-Project Conditioning"); 1 row (PDF); MIT code; explicitly "not a live-model evaluation" — [HF](https://huggingface.co/datasets/promptmafiainc/assamese-bengali-contrastive-eval)

### Inferences
- Translation-based machine text (IndicTrans2/TranslateGemma) is a distinct "machine register" from native LLM generation; a detector trained only on ChatGPT word-prompted sentences will not have seen it, yet it is by far the most abundant machine Assamese in the wild (12B tokens in Sangraha Synthetic alone). Including it as a machine class (or as a separate "MT" class) would broaden coverage.
- Sangraha Synthetic is parallel to English Wikipedia and therefore genre-matched to the project's human Assamese Wikipedia set — a near-ideal controlled pair for isolating MT artefacts from genre.
- No released set offers multiple *native* LLM generators in Assamese; this must be built locally (see KQ3).

### Gaps
- Translation model for Telugu-LLM-Labs/Tensoic/saillab Assamese Alpaca sets is undocumented.
- Exact Assamese row counts for each IndicAlign subset were not extracted (paper table truncated).
- Whether `indian-nlp/assamese-dataset` on HF is gated or deleted could not be determined (401).

---

## KQ3. Open LLMs that can generate fluent Assamese (for a multi-generator machine set)

### Takeaway
Confirmed Assamese support (by official language lists): Qwen3 (explicitly lists Assamese among 119 languages), Krutrim-2-instruct (language tag `as`), BLOOM/BLOOMZ (`as`), IndicTrans2/Sangraha pipeline (translation route), TranslateGemma (used for Assamese in IndicMedDialog). Not supported officially: Sarvam-M (10 Indic languages, Assamese excluded), Llama 3.1 (8 languages), Aya Expanse (23 languages, no Assamese). Tokenizer efficiency for Assamese ranks SUTRA > GPT-4o > Gemma 2 > Llama 3.1 > Mistral. Quantitative Assamese *generation quality* evidence is thin: IndicGenBench (29 languages incl. Assamese) is the main benchmark; no per-model Assamese scores were retrieved.

### Cited Findings
- Qwen3 blog lists Assamese by name under Indo-European among 119 supported languages/dialects ("Luxembourgish, Venetian, Assamese, Welsh, Silesian, Asturian, Chhattisgarhi, Awadhi, Maithili, Bhojpuri, Sindhi, Irish"); Qwen3-32B is Apache-2.0 — [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/); [HF API Qwen/Qwen3-32B](https://huggingface.co/api/models/Qwen/Qwen3-32B)
- Krutrim-2-instruct card languages: en, hi, bn, mr, te, ta, kn, ml, gu, **as**, pa, sa, ur; licence "Krutrim Community License Agreement Version 1.0" (custom); no per-language Assamese benchmark in text — [HF API](https://huggingface.co/api/models/krutrim-ai-labs/Krutrim-2-instruct); [model card](https://huggingface.co/krutrim-ai-labs/Krutrim-2-instruct)
- BLOOMZ language list includes `as` and `bn` (46 languages + code) — [HF API bigscience/bloomz](https://huggingface.co/api/models/bigscience/bloomz)
- Sarvam-M supports Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu + English; Assamese not included; Apache-2.0; "Indic Vibe Check" Bengali 8.17 vs Gemma 3 7.29, Llama 4 Scout 7.59, Llama 3.3 70B 7.01 — [Sarvam blog](https://www.sarvam.ai/blogs/sarvam-m); [HF API](https://huggingface.co/api/models/sarvamai/sarvam-m)
- Llama-3.1-8B-Instruct official languages: en, de, fr, it, pt, hi, es, th (no Assamese/Bengali) — [HF API](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct)
- Aya Expanse 8B: 23 languages (hi is the only Indic), CC-BY-NC-4.0, gated — [HF API](https://huggingface.co/api/models/CohereLabs/aya-expanse-8b)
- Gemma-3-27b-it: gated ("manual"), Gemma licence, no language list in card metadata — [HF API](https://huggingface.co/api/models/google/gemma-3-27b-it)
- Assamese tokenizer NSL (lower better): SUTRA 0.45, GPT-4o 0.54, Gemma 2 0.82, Llama 3.1 1.4, Mistral Large 2407 1.48; paper does not evaluate generation quality — [arXiv 2410.03718](https://arxiv.org/abs/2410.03718)
- IndicGenBench (Google, ACL 2024): 29 languages incl. Assamese (`as`, Bengali script); tasks CrossSum-IN (2.9k/2.9k/14.5k), Flores-IN (dev/devtest 28.9k/29.3k), XQuAD-IN (12 langs), XorQA-IN (28 langs); evaluated GPT-3.5, GPT-4, PaLM-2, mT5, Gemma, BLOOM, LLaMA, with PaLM-2 best and "a notable gap between every Indic language and English"; licences CC BY-SA 4.0 (XQuAD-IN, Flores-IN), MIT (XorQA-IN), CC BY-NC-SA 4.0 (CrossSum-IN); repo archived Apr 2026 — [GitHub](https://github.com/google-research-datasets/indic-gen-bench); [arXiv 2404.16816](https://arxiv.org/abs/2404.16816)
- IndicTrans2 is the translation engine behind Sangraha Synthetic and IndicAlign (Assamese included as a target) — [arXiv 2403.06350 HTML](https://arxiv.org/html/2403.06350)
- Yes-MT WMT24 compared prompted/fine-tuned Llama 3 and Mixtral vs IndicTrans2 LoRA for En↔Assamese (MT only) — [arXiv 2512.15226 via export API](http://export.arxiv.org/api/query?search_query=all:assamese+AND+%28all:%22generated%22+OR+all:%22LLM%22+OR+all:%22synthetic%22%29&max_results=40&sortBy=submittedDate)
- SurakshaEval (2026) reports strong LLMs "still fail in native scripts" on human-written Assamese safety prompts; CompositeHarm translation finds attack success "rises sharply" in Indic languages incl. Assamese — same arXiv export listing as above
- The "Making LLMs Speak Assamese" release frames "Bengali-contaminated baselines" as the core problem of LLM Assamese output — [HF promptmafiainc](https://huggingface.co/datasets/promptmafiainc/assamese-bengali-contrastive-eval)

### Inferences
- A locally buildable multi-generator set: Qwen3 (Apache-2.0), Krutrim-2 (custom licence), BLOOMZ (RAIL) plus API models (GPT-4o, Claude, Gemini) and translation-based generation (IndicTrans2/TranslateGemma from English or Bengali seeds). Gemma 3 likely works (140-language claim) but no Assamese-specific evidence was retrieved.
- Expect Bengali-script contamination (র for ৰ, missing ৱ) in weaker generators; this is both a detection "tell" and a confound (see KQ5).
- IndicGenBench's CrossSum-IN/Flores-IN Assamese references give human-written Assamese parallel to English sources, enabling prompt-matched human/machine pairs (generate from the same English input).

### Gaps
- No retrieved per-model Assamese generation-quality scores (IndicGenBench per-language table not extracted; Sarvam/Krutrim report aggregates only).
- Gemma 3, GPT-4o, Claude, Gemini Assamese fluency: no published Assamese-specific evaluation found via arXiv.

---

## KQ4. Genre-diverse human Assamese text (news, social media, chat, Wikipedia, literature, legal)

### Takeaway
Large general/news-heavy corpora exist (IndicCorp v2 3.38M Assamese lines, CC-0; Sangraha Verified 292M tokens incl. 163M from PDFs; MWirelabs 1.6M sentences / 77M tokens CC BY-SA). Literature is covered by Xahitya (9,430 articles, prose/poetry) and the Arambhani poem scrape. Social media is thin: HASOC 2023 Assamese (Twitter/Facebook/YouTube, password-gated via Google Form, sizes not published in text) and an 8,986-review movie-sentiment set (60% translated). Conversational: IndicVoices Assamese ~106 transcribed hours (role-play telephone dialogues + extempore), gated CC BY 4.0. Legal/government Assamese: no dedicated public corpus found; proxies are Samanantar/BPCC government sources (anuvaad_pib, pmi) and Sangraha Verified PDFs.

### Cited Findings
**Web/news/general**
- IndicCorp v2 (ACL 2023): `asm_Beng` split 3.38M rows; corpus-wide 20.9B tokens / 24 languages; licence exact text: "All the datasets created as part of this work will be released under a CC-0 license"; samples carry datelines/bylines (ডিজিটেল ডেস্কঃ, ষ্টাফৰিপ'র্টাৰ) suggesting news crawl, but sources undocumented; not gated — [HF ai4bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
- Sangraha Verified Assamese: Web 128.6M, PDFs 162.9M (Internet Archive main PDF source, OCR), Speech transcripts 0.6M tokens; websites manually screened, discarding "toxic, adult, and poorly machine-translated content"; CC BY 4.0, not gated — [arXiv 2403.06350 HTML](https://arxiv.org/html/2403.06350); [HF ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
- MWirelabs/assamese-monolingual-corpus: 1,613,879 sentences, 77.4M IndicBERTv2 tokens, CC BY-SA 4.0; sources IITB-IndicMonoDoc, Samanantar Assamese side, "Assamese poetry and civic texts"; single `text` column, **no genre labels**; preview shows news, Wikipedia/Wikisource, boilerplate and some Kannada/Hindi fragments — [HF](https://huggingface.co/datasets/MWirelabs/assamese-monolingual-corpus) (mirror `jintz0/assamese-monolingual-corpus`)
- ananddey/assamese-news-corpus: 1,369 articles (1,298/71), 348K tokens, title+text, licence "other", outlets unnamed (one body mentions দৈনিক জনমসূচি), OCR-like artefacts — [HF](https://huggingface.co/datasets/ananddey/assamese-news-corpus)
- Samanantar Assamese–English: ~141k pairs, `data_source` field (anuvaad_pib, anuvaad_newsonair, pmi, wikipedia, indiccorp, news outlets, coursera, khan_academy, nptel, ocr…); preview mixes religious text, news, government releases, software UI strings; CC BY-NC-4.0 — [HF ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar)
- BPCC: ~230M bitext pairs for 22 languages (BPCC-Mined ~228M, BPCC-Human 2.2M); `bpcc-seed-latest/asm_Beng.tsv` present; gated (auto), no licence in metadata — [HF API ai4bharat/BPCC](https://huggingface.co/api/datasets/ai4bharat/BPCC)
- Wikipedia dumps: `ffatty/plaintext-wikipedia-full-assamese` (CC BY-SA 3.0, 10K–100K), `marsh-mellow/assamese_wikipedia` (CC BY 4.0), `ananddey/assamese-wiki-corpus` (CC BY-SA 4.0) — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)
- CC-100 Assamese: 7.6 MB download (per indian-nlp README) — [GitHub indian-nlp/assamese-dataset](https://github.com/indian-nlp/assamese-dataset)

**Literature / poetry**
- Ranjit89/xahitya-assamese-corpus: 9,430 articles scraped from Xahitya.org (prose, essays, stories, poems, interviews, serialized pieces; dates 2011–2026), 292 MB, fields url/title/date/source/text, tagged CC0-1.0 but "rights remain with the original authors/publishers" with takedown offer; some noise (stray shell command in a URL field) — [HF](https://huggingface.co/datasets/Ranjit89/xahitya-assamese-corpus)
- Arambhani poems: "Assamese Poem - kobita" scraped from arambhani.com (Kaggle, Sani Kamal), size unstated — [arXiv 2410.11291 HTML](https://arxiv.org/html/2410.11291v2); indian-nlp/nanoGPT-assamese was trained on `arambhani.txt` from Kaggle — [GitHub org listing](https://api.github.com/orgs/indian-nlp/repos)
- naruttamboruah/assamese-idioms-proverbs: expert-generated idioms/proverbs with English, CC BY 4.0, <1K — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)

**Social media / informal**
- HASOC 2023 Task 4 "Annihilate Hates" (Assamese, Bengali, Bodo): binary HOF/NOT; data "primarily collected from Twitter, Facebook, or youtube comments"; Assamese/Bodo prepared with CIT Kokrajhar students; files password-protected, passwords via Google Form registration; no licence stated; Google Drive train/test links on page — [task site](https://sites.google.com/view/hasoc-2023-annihilate-hates)
- Participant paper: datasets "fairly balanced"; Assamese and Bengali contain emojis/repeated punctuation, all contain code-mixed English words; best Assamese macro-F1 0.70525 (mBERT cased), Bengali 0.75625 (BanglaBERT), Bodo 0.83009; counts only in figures — [arXiv 2312.05671 HTML](https://arxiv.org/html/2312.05671)
- HASOC 2023 overview paper: "Overview of the HASOC Subtracks at FIRE 2023: Hate Speech and Offensive Content Identification in Assamese, Bengali, Bodo, Gujarati and Sinhala", DOI 10.1145/3632754.3633278 — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Assamese+Bengali+script+language+identification)
- AvinabhDutta-Dev/assamese-movie-reviews-sentiment: 8,986 reviews from IMDb (English), Reddit r/assam (mostly Romanised Assamese), YouTube comments on Assamese films (mixed script); 5,677 pos / 3,309 neg; two native annotators, κ 0.942; ~60% "required translation or script normalization" via Google Translate/QuillBot; CC BY 4.0; some rows are Bengali/Bangladeshi-drama content — [HF](https://huggingface.co/datasets/AvinabhDutta-Dev/assamese-movie-reviews-sentiment)
- nabakamal/assamese_sentiment: 120,921 rows, 3-class, MIT, README empty; samples look like localisation strings, Jehovah's Witness religious text, news/government — not social media; labels appear noisy — [HF](https://huggingface.co/datasets/nabakamal/assamese_sentiment)
- Nagamese (Assamese-lexified creole) sentiment lexicon of 1,195 words (2025) — not Assamese proper — [arXiv export](http://export.arxiv.org/api/query?search_query=all:assamese+AND+%28all:hate+OR+all:offensive+OR+all:%22social+media%22+OR+all:tweets+OR+all:%22code-mixed%22+OR+all:youtube+OR+all:sentiment%29&max_results=40&sortBy=submittedDate)
- HF searches for "assamese youtube", "assamese chat", "translit assamese" returned zero datasets — [HF API](https://huggingface.co/api/datasets?search=assamese%20youtube&limit=30)

**Conversational / speech transcripts**
- IndicVoices Assamese: ASR splits train 100 h (684 speakers), test 5 h (163 speakers), val 1 h; prompt inventory 9,146 read sentences, 2,401 extempore questions, 1,185 role-play scenarios, 103 customer-care scenarios; role-plays are two-party telephone (8 kHz) occupation-pair dialogues (customer/shopkeeper etc.) of 2–3 minutes; dataset-wide 9% read / 74% extempore / 17% conversational; collection partner Pragyam Foundation — [arXiv 2403.01926 HTML](https://arxiv.org/html/2403.01926)
- HF release CC BY 4.0, gated (accept conditions + token), 745 GB, 11.2K of 23.7K hours transcribed as of Dec 2025 — [HF ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)
- InfoBayAI Assamese call-center audio datasets (dual/single channel, CC BY 4.0, <1K) — audio only tags — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)
- Afuu-coder/asteria-bhojpuri-assamese-civic-qa: civic/government-scheme QA, CC BY 4.0, 1K–10K (creation method not inspected) — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)

**Legal / government**
- HF searches "assamese legal" returned none; "indic legal" hits (LegalBrain-Indic-Legal-Corpus, indic-legal-benchmark, legalTransEn_Indic) are tagged English only — [HF API](https://huggingface.co/api/datasets?search=indic%20legal&limit=20)
- Bhashini-related HF hits are Hindi/Tamil/Malayalam/Nepali; only `Bhashini-IITJ/Bharat-Scene-Text-Dataset` tags `as` (scene-text images) — [HF API](https://huggingface.co/api/datasets?search=bhashini&limit=20)
- Government-register Assamese exists inside Samanantar (`anuvaad_pib`, `pmi`, `anuvaad_newsonair` sources) and IndicVoices read-speech prompts drawn from Wikipedia sentences in "legal, governance, health, STEM, sports and religion" domains — [HF samanantar](https://huggingface.co/datasets/ai4bharat/samanantar); [arXiv 2403.01926 HTML](https://arxiv.org/html/2403.01926)

**Other human evaluation sets with Assamese**
- SurakshaEval human-written safety prompts in 10 Indian languages incl. Assamese, GitHub debobanerjee/SurakshaEval; BLEnD 52.6k cultural QA pairs incl. Assamese; MQM/DA MT test sets for Assamese (arXiv 2406.03893) — [arXiv export](http://export.arxiv.org/api/query?search_query=all:assamese+AND+%28all:%22generated%22+OR+all:%22LLM%22+OR+all:%22synthetic%22%29&max_results=40&sortBy=submittedDate)
- WhatdTECH/Assamese_SCERT_Class_Science_Dataset (GPL, <1K) — textbook register — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)

### Inferences
- Registers with adequate volume: news/web (IndicCorp v2, Sangraha), encyclopaedic (Wikipedia), literature (Xahitya, Arambhani), spoken-conversational (IndicVoices transcripts). Weak: native social-media text (HASOC Assamese is the only genuine comment corpus, gated by registration), chat/messaging (none), legal (none dedicated).
- Samanantar's `data_source` field and IndicVoices' task labels are the only per-document genre metadata found; most monolingual corpora lack genre tags, so genre-stratified evaluation requires source-level reconstruction (e.g., Sangraha `verified` by source type, Samanantar by `data_source`).
- Movie-review and nabakamal sentiment sets contain translated text and should not be used as "human Assamese" negatives without filtering.

### Gaps
- HASOC 2023 Assamese train/test counts: only in figures/overview paper (ACM, not fetched). The "5,044-comment Assamese hate-speech corpus" mentioned in the brief could not be located (Semantic Scholar 429; arXiv has no such paper) — treat as unverified.
- "AssameseBackTranslit" dataset: no HF/arXiv hit under that name.
- Assam Gazette, Assam Legislative Assembly proceedings, Wikisource Assamese sizes: not researched (no web search tool); no HF packaging found.
- IITB-IndicMonoDoc Assamese size/licence not inspected.

---

## KQ5. Assamese/Bengali studies on LLM-text detection and script-level tells (ৰ vs র, ৱ)

### Takeaway
No peer-reviewed study of machine-generated *Assamese* text detection was found. The only Assamese-specific work on LLM output orthography is a 2026 preprint/code release about forcing LLMs to produce Assamese rather than "Bengali-contaminated" output; Bengali detection work (two 2025–26 papers) exists but without released data. Script-letter tells are therefore undocumented in the literature, though the Bengali-contamination framing and TranslateGemma's need for "script-aware post-processing" indirectly confirm the phenomenon.

### Cited Findings
- arXiv full-text search (Assamese + generated/LLM/synthetic, 25 results) found "no paper addresses machine-generated text detection" for Assamese — [arXiv export](http://export.arxiv.org/api/query?search_query=all:assamese+AND+%28all:%22generated%22+OR+all:%22LLM%22+OR+all:%22synthetic%22%29&max_results=40&sortBy=submittedDate)
- Semantic Scholar query for Assamese MGT detection returned 0 results — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/search?query=assamese+corpus+legal+OR+assamese+literature+corpus+OR+assamese+code-mixed+social+media+corpus+OR+assamese+dialogue+dataset&fields=title&limit=20)
- "Making Large Language Models Speak Assamese: Dual-Script Structured Prompting and Phonemic Anchor-and-Project Conditioning" (PromptMafia, 2026): targets "Bengali-contaminated baselines" with "deterministic Assamese–Bengali contrastive artifacts"; self-test "is not a live-model evaluation"; MIT code, CC BY 4.0 manuscript — [HF](https://huggingface.co/datasets/promptmafiainc/assamese-bengali-contrastive-eval)
- IndicMedDialog applied a "script-aware post-processing pipeline" to fix "phonetic, lexical, and spacing errors" in TranslateGemma output for Assamese and other languages — [arXiv 2605.13292](https://arxiv.org/abs/2605.13292)
- `anrikus/lexical_diff_bangla_assamese` (v1, v2; 1K–10K, image modality) exists on HF — content not inspected — [HF API search](https://huggingface.co/api/datasets?search=assamese&limit=200&full=false)
- The movie-review sentiment set's own translation pipeline introduced at least one Bengali-language row into an "Assamese" dataset — [HF AvinabhDutta-Dev](https://huggingface.co/datasets/AvinabhDutta-Dev/assamese-movie-reviews-sentiment)
- Bengali detection: fine-tuned XLM-R/mDeBERTa/mBERT ≈91% on AI paraphrases, zero-shot ≈50% — [arXiv 2512.21709](https://arxiv.org/abs/2512.21709); Bangla QSVM detection paper (Applied AI Letters 2026) — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Bangla+AI-generated+text+detection)
- Translation-artefact evidence in Assamese SFT data: "stray spaces around apostrophes in Assamese text" in IndicTrans2-derived subsets — [HF ananddey/assamese-sft-dataset-v1](https://huggingface.co/datasets/ananddey/assamese-sft-dataset-v1)

### Inferences
- Script-level features (frequency of র vs ৰ, presence of ৱ, Bengali-only lexis) are likely strong, cheap discriminators for weaker generators but will also fire on human Bengali-influenced or OCR'd text, and will vanish for strong generators or post-processed pipelines — consistent with the project's finding that no single feature set is genre-robust.
- Because no prior Assamese MGT-detection study exists, the project's detector and any released Assamese benchmark would be first-of-kind; the Bengali paraphrase-detection setup (paraphrase human text → prompt-matched pairs) is the closest methodological precedent.

### Gaps
- No quantitative study of ৰ/র or ৱ substitution rates in LLM Assamese output was found.
- Contents of `anrikus/lexical_diff_bangla_assamese` and the PromptMafia PDF were not read.
- Bangla QSVM paper details (dataset, generators) unavailable (paywalled).
