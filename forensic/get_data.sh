#!/usr/bin/env bash
# Fetch the public datasets used by E1-E4 into ../data (~190 MB; CC-100 is a 12 MB prefix).
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p data/{poems,backtranslit,text,calls}
GH=https://raw.githubusercontent.com
curl -fsSL -o data/poems/arambhani.txt "$GH/indian-nlp/nanoGPT-assamese/HEAD/data/assamese-poem-arambhani/assamese-poem-arambhani.txt"
curl -fsSL -o data/backtranslit/sentence-aligned.txt "$GH/osintg-iitghy/LREC-COLING-2024-OSINTG-IITG/HEAD/sentence-aligned.txt"
A=$GH/indian-nlp/assamese-dataset/HEAD
curl -fsSL -o data/text/chatgpt.jsonl "$A/asm_wikipedia_2021_100K/output_100k_split_1.jsonl"
curl -fsSL -o data/text/wiki100k_sentences.txt "$A/asm_wikipedia_2021_100K/asm_wikipedia_2021_100K-sentences.txt"
curl -fsSL -r 0-12000000 -o data/text/cc100_head.txt "$A/fine-tuning-datasets/cc-100/as.txt"
# the repo's file names contain a literal "%20", hence the double encoding
for i in 1 2 3 4 5; do for s in 00 01; do
  curl -fsSL -o "data/calls/call${i}_ch${s}.wav" "$GH/InfoBay-AI/call_center_audio_dual_channel_as/HEAD/SAMPLE/SAMPLE-$i/Assamese_Sample%2520-%25200${i}__SPEAKER_$s.wav"
done; done
# pretrained speaker-embedding models for e3_speaker_plus.py (~70 MB)
mkdir -p models
for m in wespeaker_en_voxceleb_CAM++.onnx 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx; do
  curl -fsSL -o "models/$m" "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$m"
done
# further encoders for e3_encoders.py (~270 MB)
for m in 3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx 3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx wespeaker_zh_cnceleb_resnet34_LM.onnx wespeaker_en_voxceleb_resnet34_LM.onnx; do
  curl -fsSL -o "models/$m" "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$m"
done
