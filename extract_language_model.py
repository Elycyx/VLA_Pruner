from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import os
import shutil

# 加载VLA模型
print("正在加载VLA模型...")
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 提取语言模型
print("正在提取语言模型...")
language_model = vla.language_model

# 加载并保存tokenizer
print("正在加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial",
    trust_remote_code=True
)

# 设置保存路径
save_dir = "models/vla_language_model"
os.makedirs(save_dir, exist_ok=True)

# 保存模型状态字典为pytorch_model.bin
print(f"正在保存模型状态到 {save_dir}/pytorch_model.bin")
torch.save(language_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

# 保存配置文件
print("正在保存模型配置...")
language_model.config.save_pretrained(save_dir)

# 保存tokenizer
print("正在保存tokenizer...")
tokenizer.save_pretrained(save_dir)

# 保存特殊token配置
special_tokens = {
    "pad_token": tokenizer.pad_token,
    "eos_token": tokenizer.eos_token,
    "bos_token": tokenizer.bos_token,
    "unk_token": tokenizer.unk_token
}

# 保存原始VLA配置
original_config = vla.config
config_save_path = os.path.join(save_dir, "original_vla_config.json")
original_config.save_pretrained(config_save_path)

# 验证模型配置
print("\n模型信息:")
print(f"词表大小: {language_model.config.vocab_size}")
print(f"隐藏层维度: {language_model.config.hidden_size}")
print(f"模型类型: {language_model.config.model_type}")
print(f"特殊token: {special_tokens}")

print("\n保存完成!")

# 验证是否可以重新加载
print("\n验证加载...")
try:
    # 加载配置
    config = AutoConfig.from_pretrained(save_dir)
    
    # 创建模型实例
    loaded_model = AutoModelForCausalLM.from_config(config)
    
    # 加载状态字典
    state_dict = torch.load(os.path.join(save_dir, "pytorch_model.bin"))
    loaded_model.load_state_dict(state_dict)
    
    # 加载tokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
    
    print("模型和tokenizer加载成功!")
except Exception as e:
    print(f"加载验证失败: {str(e)}") 