import torch
from transformers import AutoModel, PrunedLlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, PrunedLlamaConfig


AutoConfig.register("pruned_llama", PrunedLlamaConfig)
AutoModelForCausalLM.register(PrunedLlamaConfig, PrunedLlamaForCausalLM)
# 加载模型
model = AutoModelForCausalLM.from_pretrained("models/vla_llama2_shortenedllm", torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("models/vla_llama2_shortenedllm")
# 打印 Hugging Face 自带的统计摘要
# input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to("cuda")
# outputs = model.generate(input_ids, max_new_tokens=1024)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# pruned_dict = torch.load('models/vla_llama2_llmpruner/pytorch_model.bin', map_location='cuda')
# llama_tokenizer, llama_model = pruned_dict['tokenizer'], pruned_dict['model']
print(model)
# # print(pruned_dict)
# # print(llama_tokenizer)
# print(llama_model)