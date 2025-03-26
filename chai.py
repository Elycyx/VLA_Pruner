import torch

pruned_dict = torch.load('models/vla_llama2_llmpruner/pytorch_model.bin', map_location='cuda')
tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

# 保存模型和分词器
save_directory = 'models/vla_llama2_llmpruner2'  # 定义保存路径
model.save_pretrained(save_directory)  # 保存模型
tokenizer.save_pretrained(save_directory)  # 保存分词器

