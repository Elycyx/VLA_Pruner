# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

model_name = "models/spatial_wanda"
# Load Processor & VLA
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_name, 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda")

# # 替换语言模型为本地的 Llama2
# from transformers import LlamaForCausalLM
# llama_model = LlamaForCausalLM.from_pretrained(
#     "models/llama2_7b",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True
# ).to("cuda")

# vla.language_model = llama_model
print(vla.language_model)