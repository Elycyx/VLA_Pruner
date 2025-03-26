# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from transformers import AutoModel, PrunedLlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, PrunedLlamaConfig

import torch
AutoConfig.register("pruned_llama", PrunedLlamaConfig)
AutoModelForCausalLM.register(PrunedLlamaConfig, PrunedLlamaForCausalLM)

# model_path = "openvla/openvla-7b-finetuned-libero-spatial"
model_path = "models/spatial_shortenedllm"
# Load Processor & VLA
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path, 
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

print(vla)
# 读取本地图片
image_path = "test.jpg"  # 替换为您的图片路径
image = Image.open(image_path)

# 设置提示词
prompt = "In: What action should the robot take to fold the cloth from top right to center?\nOut:"  # 根据需要修改指令

# 预测动作
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

# 打印预测的动作值
print("Predicted action:", action)
print("Action shape:", action.shape)  # 查看动作向量的维度


# Execute...
# robot.act(action, ...)  # 注释掉实际执行部分