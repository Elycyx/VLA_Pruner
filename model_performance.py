"""
测试 OpenVLA 在简单样本上的性能指标。
包括:
- 模型总参数量
- GPU 显存占用
- 推理吞吐量
- 平均推理时间
"""

import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from PIL import Image
import GPUtil
from transformers import AutoModel, PrunedLlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, PrunedLlamaConfig
AutoConfig.register("pruned_llama", PrunedLlamaConfig)
AutoModelForCausalLM.register(PrunedLlamaConfig, PrunedLlamaForCausalLM)

# 添加项目根目录到 Python 路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.robot.openvla_utils import get_vla, get_processor, OPENVLA_V01_SYSTEM_PROMPT
from experiments.robot.robot_utils import get_action


@dataclass
class TestConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""  # OpenVLA checkpoint path
    local_llm_path: Optional[str] = None  # 可选的本地语言模型路径
    num_test_samples: int = 5  # 测试样本数
    num_warmup_runs: int = 3  # 预热运行次数
    num_test_runs: int = 10  # 每个样本测试次数
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    unnorm_key: str = "libero_spatial"  # 用于动作反归一化的键
    max_sequence_length: int = 256  # 添加这一行


def get_gpu_memory():
    """获取当前 GPU 显存使用情况 (MB)"""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 0
    return gpus[0].memoryUsed


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def create_dummy_image(size=(224, 224)):
    """创建测试用的虚拟图像"""
    img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return img  # 直接返回 numpy 数组


def main(cfg: TestConfig):
    print("\n=== OpenVLA Performance Test ===")
    
    # 1. 加载模型
    print("\n[1] Loading model...")
    initial_gpu_mem = get_gpu_memory()
    
    model = get_vla(cfg)
    processor = get_processor(cfg)
    
    model_gpu_mem = get_gpu_memory() - initial_gpu_mem
    total_params = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"GPU memory usage: {model_gpu_mem:.1f} MB")

    # 在加载模型后添加
    model.config.max_position_embeddings = cfg.max_sequence_length
    
    # 2. 准备测试数据
    print("\n[2] Preparing test data...")
    
    # 创建一些测试样本
    test_samples = []
    test_tasks = [
        "pick up the red cube and place it on the blue cube",
        "move the green block to the right of the yellow block",
        "stack the blue cube on top of the red cube",
        "push the yellow cube towards the green cube",
        "grasp the red cube and lift it up"
    ]
    
    # 创建观察字典
    observation = {
        'state': np.zeros(9),  # 假设状态向量是9维的
    }
    
    for task in test_tasks[:cfg.num_test_samples]:
        obs = observation.copy()
        obs['full_image'] = create_dummy_image()
        test_samples.append({
            **obs,
            'task_description': task,
        })

    # 3. 性能测试
    print("\n[3] Running performance test...")
    
    # 预热运行
    print("Warming up...")
    for _ in range(cfg.num_warmup_runs):
        sample = random.choice(test_samples)
        with torch.no_grad():
            _ = get_action(cfg, model, sample, sample['task_description'], processor=processor)

    # 正式测试
    print("Testing...")
    inference_times = []
    
    for i, sample in enumerate(test_samples):
        sample_times = []
        
        for _ in range(cfg.num_test_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                _ = get_action(cfg, model, sample, sample['task_description'], processor=processor)
            
            torch.cuda.synchronize()
            end_time = time.time()
            sample_times.append(end_time - start_time)
        
        avg_time = np.mean(sample_times)
        inference_times.append(avg_time)
        print(f"Sample {i+1}/{len(test_samples)}: {avg_time*1000:.1f} ms")
        print(f"Task: {sample['task_description']}")

    # 4. 输出结果
    print("\n=== Performance Summary ===")
    print(f"Model: OpenVLA")
    print(f"Checkpoint: {cfg.pretrained_checkpoint}")
    if cfg.local_llm_path:
        print(f"Local LLM: {cfg.local_llm_path}")
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"GPU Memory Usage: {model_gpu_mem:.1f} MB")
    print(f"Average Inference Time: {np.mean(inference_times)*1000:.1f} ms")
    print(f"Throughput: {1.0/np.mean(inference_times):.1f} samples/second")
    print("==========================")


if __name__ == "__main__":
    import draccus
    cfg = draccus.parse(TestConfig)
    main(cfg)