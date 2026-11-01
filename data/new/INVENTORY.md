# Newly acquired resources (31 Oct 2026) — "download today" tier from the dataset survey

Re-fetch everything with `python download_hf.py && python download_models.py` (HF) plus the two curl commands below.
Total on disk ≈ 3.4 GB. Gated resources (IndicVoices, VAANI, IndicF5) are NOT here: their click-through licences
must be accepted by the account holder on huggingface.co before `hf_hub_download` will serve them.

| Resource | Path | What arrived | Licence | Status / use |
|---|---|---|---|---|
| Aksharantar asm | `aksharantar/asm_{train,valid,test}.json` | 178,630 / 3,788 / 5,506 Roman↔native word pairs; sources IndicCorp 177,393, Samanantar 814, Wikidata 423 (fields: native word, english word, source) | CC0 (mined) / CC-BY (manual) | **Ready** — E2 vocabulary coverage; register is news, not social media |
| Bhasha-Abhijnaanam | `bhasha/bhasha-abhijnaanam.json` | 88,014 sentences over 22 languages with `native sentence`, `romanized sentence`, `language`, `script` fields; filter `language == "Assamese"` | CC0 | **Ready** — only external sentence-level E2 test set |
| IndicCorp v2 asm | `indiccorp/data/as.txt` | 3,375,484 lines native-script Assamese (1.18 GB) | CC0 | **Ready** — native word LM for E2 reranking; news-genre humans for E4 |
| Sangraha verified asm | `sangraha/verified/asm/data-{0,1,2}.parquet` | 3 × 110,238 docs (`doc_id`, `text`, `type`); first 3 of the verified shards | CC BY 4.0 | **Ready** — LM data; `type` gives source genre for E4 stratification |
| Sangraha unverified asm | `sangraha/unverified/asm/data-0.parquet` | 18,713 docs | CC BY 4.0 | Ready |
| Sangraha synthetic asm_Beng | `sangraha/synthetic/asm_Beng/wiki_asm_Beng_000{0,1}_of_0063.parquet` | 2 × 87,914 IndicTrans2-translated Wikipedia docs (of 63 shards) | CC BY 4.0 | **Ready** — E4 machine-translation class, genre-matched to Wikipedia humans |
| Sangraha synthetic asm_Latn | `sangraha/synthetic/asm_Latn/wiki_asm_Latn_0000_of_0063.parquet` | 87,914 IndicXlit-romanised docs | CC BY 4.0 | Ready — romanised-encoder pre-training (canonical spellings only) |
| Xahitya | `xahitya/xahitya_dump/articles.jsonl` | 9,430 literary articles (`url`, `title`, `date`, `source`, `text`), 2011–2026 | tagged CC0; author rights caveat | **Ready** — E4 literature genre; check for author bylines for E1 |
| Assamese Wikipedia stub history | `aswiki/aswiki-latest-stub-meta-history.xml.gz` | revision metadata only (49 MB; superseded by the full history below): 586,380 revisions by 5,840 registered users; **238 users with ≥100 revisions, 353 with ≥50**; namespaces: articles 424,791, user-talk 41,490, talk 17,809, user 21,370; ~98k revisions from bot-named accounts | CC BY-SA | **Sizing done** — enough editors for an open-set E1 split. Next: fetch the full `pages-meta-history` dump (~390 MB bz2) and extract per-editor diffs, talk namespaces first |
| Assamese Wikipedia full history | `aswiki/aswiki-20261001-pages-meta-history.xml.bz2` (372 MB, from mirror `dumps.wikimedia.your.org` — main host returns 404) → `aswiki/editor_sessions.jsonl` | 35,958 editor sessions from 2,710 non-bot editors (added text only, markup stripped); 642 editors with ≥3 sessions | CC BY-SA | **Built and evaluated** (`forensic/wiki_authors.py`, `forensic/e1_wiki.py`; README round 5) |
| IndicXlit (CTranslate2 fp32) | `models/indicxlit_ct2/indicxlit_ct2_fp32/` | 47 MB en→indic transliteration model + vocabularies | MIT | Ready — n-best candidate generator for E2 (needs `pip install ctranslate2`) |
| IndicXlit Roman→Assamese ONNX int8 | `models/indicxlit_as_onnx/` | encoder 6 MB + decoder 8 MB + vocabs | MIT (as tagged) | Ready — lightweight alternative |
| **AAALC** | `aaalc/translated_output.csv` | 9,730 rows, 16 authors (Benudhar Sarma 892 … Hemaprava Das 37), columns Author / Text / Translated_Text | CC BY 4.0 | **UNUSABLE as hosted**: the Assamese `Text` column contains **zero Bengali-script characters** — 15.3 M of 19.5 M characters are `?` (encoding destroyed before upload). Only the English machine translation survives, itself 46% `?`. Author counts confirm this *is* the paper's 9,730-sample / 16-author corpus. Action: contact the Mendeley contributor for the original-encoding file |

Downloads outside Hugging Face:
```bash
curl -fsSL -o aswiki/aswiki-latest-stub-meta-history.xml.gz https://dumps.wikimedia.org/aswiki/latest/aswiki-latest-stub-meta-history.xml.gz
curl -fsSL -o aaalc/translated_output.csv "https://data.mendeley.com/public-files/datasets/m24w2n2jtb/files/b2a34ca8-21da-4a84-bd78-4960b2105916/file_downloaded"
```

Not downloaded (by design): remaining 60 Sangraha synthetic shards (~16 GB), remaining verified shards, the
full aswiki page history. Gated: IndicVoices, IndicVoices-R, VAANI, IndicF5 — accept the licence on the Hub, then
`hf_hub_download(..., token=...)`.
