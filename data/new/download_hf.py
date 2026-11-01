import sys, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); root = os.path.dirname(os.path.abspath(__file__))
def get(repo, fn, sub, rt="dataset"):
    p = hf_hub_download(repo, fn, repo_type=rt, local_dir=f"{root}/{sub}"); print("ok", p, flush=True)
get("ai4bharat/Aksharantar", "asm.zip", "aksharantar")
get("ai4bharat/Bhasha-Abhijnaanam", "bhasha-abhijnaanam.zip", "bhasha")
get("Ranjit89/xahitya-assamese-corpus", "xahitya_dump/articles.jsonl", "xahitya")
info = api.dataset_info("ai4bharat/sangraha")
files = sorted(s.rfilename for s in info.siblings if "asm" in s.rfilename)
ver = [f for f in files if "synthetic" not in f]
print("sangraha asm non-synthetic files:", len(ver), ver[:40], flush=True)
for f in ver[:4]: get("ai4bharat/sangraha", f, "sangraha")  # first shards of verified/unverified
for f in [x for x in files if "synthetic/asm_Beng" in x][:2] + [x for x in files if "synthetic/asm_Latn" in x][:1]: get("ai4bharat/sangraha", f, "sangraha")
get("ai4bharat/IndicCorpV2", "data/as.txt", "indiccorp")
