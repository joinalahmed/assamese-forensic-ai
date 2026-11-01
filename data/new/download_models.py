import os
from huggingface_hub import snapshot_download
root = os.path.dirname(os.path.abspath(__file__))
print(snapshot_download("Singla0009/all-indic-transliteration", allow_patterns=["indicxlit_ct2_fp32/*", "README.md"], local_dir=f"{root}/models/indicxlit_ct2"), flush=True)
print(snapshot_download("aaiiaxom/indicxlit-roman-to-assamese-onnx-int8", local_dir=f"{root}/models/indicxlit_as_onnx"), flush=True)
