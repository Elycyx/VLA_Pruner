# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor, PrunedLlamaForCausalLM, AutoTokenizer
from PIL import Image
import torch
import dill

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b-finetuned-libero-spatial", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial", 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda")

# # 加载并替换语言模型
# pruned_dict = torch.load('models/vla_llama2_llmpruner/pytorch_model.bin', map_location='cuda')
# llama_tokenizer, llama_model = pruned_dict['tokenizer'], pruned_dict['model']
llama_model = PrunedLlamaForCausalLM.from_pretrained("models/vla_llama2_flap", torch_dtype=torch.bfloat16, device_map="cuda")

# 替换语言模型
vla.language_model = llama_model

output_dir = "models/spatial_flap"

vla.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"模型已保存到 {output_dir}")

# print(vla)

# image_path = "test.jpg"  # 替换为您的图片路径
# image = Image.open(image_path)

# # 设置提示词
# prompt = "In: What action should the robot take to fold the cloth from top right to center?\nOut:"

# # 使用新的processor处理输入
# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# torch.save(vla, f'{output_dir}/model.pth')
# print(f'模型已保存到 {output_dir}')

# loaded_vla = torch.load(f'{output_dir}/model.pth')
# print(f'模型已加载到 {loaded_vla}')

# print(vla.config)

# # 打印替换前后的权重检查
# print("\n检查模型权重:")
# for name, param in vla.language_model.named_parameters():
#     if 'weight' in name:
#         print(f"{name}: {param.shape}")

# 保存替换后的完整模型到本地
# output_dir = "models/spatial_llmpruner"

# # print(f'vla: {vla}')

# # # 验证：从本地加载保存的模型
# with open(f"{output_dir}/pytorch_model.pkl", "wb") as f:
#     dill.dump(vla, f)
# print(f"模型已保存到 {output_dir}")

# # 读取本地图片
# image_path = "test.jpg"  # 替换为您的图片路径
# image = Image.open(image_path)
# loaded_processor = AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)
# with open(f"{output_dir}/pytorch_model.pkl", "rb") as f:
#     loaded_vla = dill.load(f)
# # 设置提示词
# prompt = "In: What action should the robot take to fold the cloth from top right to center?\nOut:"

# # 使用新的processor处理输入
# inputs = loaded_processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# # 打印调试信息
# print("Input shape:", inputs.input_ids.shape)
# print("Input sequence length:", inputs.input_ids.size(1))

# # 预测动作
# action = loaded_vla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

# # 打印预测的动作值
# print("Predicted action:", action)

# 手动构建state_dict
# state_dict = vla.state_dict()


# # 保存模型和配置
# torch.save({
#     'model_state_dict': state_dict,
#     'config': vla.config
# }, f"{output_dir}/pytorch_model.bin")

# # 保存processor
# processor.save_pretrained(output_dir)
# print(f"模型已保存到 {output_dir}")




# # 验证：从本地加载保存的模型并检查权重
# print("\n加载保存的模型并检查权重:")
# loaded_state_dict = torch.load(f"{output_dir}/pytorch_model.bin")
# loaded_processor = AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)
# loaded_vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla/openvla-7b-finetuned-libero-spatial",
#     state_dict=loaded_state_dict['model_state_dict'],
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to("cuda")

# for key, value in loaded_state_dict['model_state_dict'].items():
#     if 'language_model.model.layers.14' in key:
#         print(f'{key}: {value.shape}')

# # 测试模型
# image_path = "test.jpg"  # 替换为您的图片路径
# image = Image.open(image_path)
# prompt = "In: What action should the robot take to fold the cloth from top right to center?\nOut:"
# inputs = loaded_processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# action = loaded_vla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
# print("\nPredicted action:", action)

