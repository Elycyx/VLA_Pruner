import torch
import torch.nn as nn
from FLAP.lib.prune import prune_with_predefined_ratio, check_sparsity
from FLAP.lib.eval import eval_ppl
from FLAP.lib.data import get_loaders
import numpy as np
from tqdm import tqdm
import math
import copy
import time
from simulation.openvla.experiments.robot.libero.run_libero_eval import GenerateConfig, eval_libero
from transformers import AutoTokenizer
import os
import glob
import re
import contextlib


class LLMPruningEnv:
    """
    Environment for LLM pruning search
    """
    def __init__(self, model, tokenizer, args, preserve_ratio=0.5, device=torch.device("cuda:0")):
        """
        初始化LLM剪枝环境
        
        Args:
            model: 要剪枝的LLM模型
            tokenizer: 分词器
            args: 配置参数，需要包含以下字段：
                - libero_task_suite: LIBERO任务集名称
                - libero_num_trials: 每个任务的评估次数
                - libero_num_workers: 评估时的worker数量
                - libero_center_crop: 是否使用中心裁剪
                - wandb_project: W&B项目名称（可选）
                - wandb_entity: W&B实体名称（可选）
            preserve_ratio: 目标保留比例
            device: 运行设备
        """
        # 保存基本设置
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = device
        self.preserve_ratio = preserve_ratio
        
        # 获取模型层数
        self.num_layers = len(self.model.language_model.model.layers)
        
        # 计算原始模型大小
        self.org_param_size = sum(p.numel() for p in model.parameters())
        print(f'=> Original model size: {self.org_param_size/1e9:.2f}B parameters')
        
        # 设置LIBERO评估配置
        self.eval_config = GenerateConfig(
            model_family="openvla",
            pretrained_checkpoint=args.model,
            task_suite_name=args.libero_task_suite,
            num_trials_per_task=args.libero_num_trials,
            # num_workers=args.libero_num_workers,
            # center_crop=args.libero_center_crop,
            use_wandb=False,  # 在评估时不使用wandb
            wandb_project=args.wandb_project if hasattr(args, 'wandb_project') else None,
            wandb_entity=args.wandb_entity if hasattr(args, 'wandb_entity') else None,
            run_id_note="pruning_eval"
        )

        
        # 设置搜索范围
        self.lbound = args.lbound if hasattr(args, 'lbound') else 0.1  # 最小保留比例
        self.rbound = args.rbound if hasattr(args, 'rbound') else 1.0  # 最大保留比例
        
        # 记录最佳结果
        self.best_reward = -math.inf
        self.best_strategy = None
        
        # 构建状态嵌入
        self._build_state_embedding()
        
        # 初始化统计信息
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0

    def _build_state_embedding(self):
        """构建状态嵌入，参考原始模型的特征提取方式"""
        layer_embedding = []
        for i in range(self.num_layers):
            layer = self.model.language_model.model.layers[i]
            
            # 提取每层的特征
            this_state = [
                i / self.num_layers,                    # 层的相对位置
                layer.self_attn.num_heads,             # 注意力头数量
                layer.mlp.gate_proj.weight.size(0),    # MLP大小
                0.0,                                    # 当前已减少的参数比例
                1.0,                                    # 剩余可减少的参数比例
                1.0                                     # 上一层的动作
            ]
            layer_embedding.append(this_state)
            
        # 归一化处理
        layer_embedding = np.array(layer_embedding, dtype=np.float32)
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)
                
        self.layer_embedding = layer_embedding
        print(f'=> State embedding shape: {self.layer_embedding.shape}')

    def step(self, action):
        """执行一个剪枝动作并返回新的状态、奖励和是否完成"""
        # 更新当前层的策略
        self.strategy.append(float(action))  # 确保action是float类型
        
        # 执行剪枝
        layer_idx = len(self.strategy) - 1
        
        # 更新状态嵌入
        self.layer_embedding[layer_idx, 3] = sum(self.strategy) / len(self.strategy)  # 当前已减少的参数比例
        self.layer_embedding[layer_idx, 4] = 1.0 - (sum(self.strategy) / len(self.strategy))  # 剩余可减少的参数比例
        self.layer_embedding[layer_idx, 5] = float(action)  # 当前动作
        
        # 判断是否完成所有层的剪枝
        done = len(self.strategy) == self.num_layers
        
        if done:
            # 重置模型到原始状态
            self.pruned_model = copy.deepcopy(self.model)
            # print(self.pruned_model.language_model)
            self.pruned_model.language_model.seqlen = 128
            
            # 使用当前完整策略进行实际剪枝
            prune_with_predefined_ratio(
                self.args,
                self.pruned_model.language_model,
                self.tokenizer,
                self.strategy,
                self.device
            )
            #self.pruned_model.language_model.save_pretrained('results/pruned_model')
            # print(self.pruned_model.language_model)
            #tokenizer = AutoTokenizer.from_pretrained('openvla/openvla-7b-finetuned-libero-spatial', trust_remote_code=True)
            #tokenizer.save_pretrained('results/pruned_tokenizer')

            # calculate the number of parameters
            num_params = sum(p.numel() for p in self.pruned_model.parameters())
            print(f'number of parameters: {num_params/1e9:.2f}B')
            print('current strategy: ', self.strategy)
            print('avarage strategy: ', sum(self.strategy) / len(self.strategy))
            print('pruning ratio: ', 1 - (num_params / self.org_param_size))
            
            # 评估剪枝后的性能
            # print('start to evaluate model')
            # print(self.eval_config)
            current_success_rate = self._evaluate_model(self.pruned_model)
            print('evaluate model done')
            current_ratio = float(num_params / self.org_param_size)
            # 计算奖励
            reward = float(self._calculate_reward(current_success_rate, current_ratio))
            
            # 更新最佳策略
            if reward > self.best_reward:
                self.best_reward = float(reward)
                self.best_strategy = [float(x) for x in self.strategy]
                print(f'New best reward: {reward:.4f}, success rate: {current_success_rate:.4f}, ratio: {current_ratio:.4f}')
                print(f'New best strategy: {self.best_strategy}')
        else:
            reward = 0.0
            current_success_rate = 0.0
            current_ratio = 0.0
            
        # 返回下一个状态（如果未完成）或当前状态（如果完成）
        next_observation = self.layer_embedding[len(self.strategy), :].copy() if not done else self.layer_embedding[layer_idx, :].copy()
        
        info = {
            'success_rate': float(current_success_rate) if done else 0.0,
            'compress_ratio': float(current_ratio) if done else 0.0
        }
        
        return next_observation, float(reward), done, info

    def reset(self):
        """重置环境到初始状态"""
        # 恢复原始权重
        self.pruned_model = self.model
        
        # 重置当前层索引
        self.current_layer = 0
        
        # 重置策略
        self.strategy = []
        
        # 重置状态嵌入
        self.layer_embedding[:, 3] = 0.0  # 重置已减少比例
        self.layer_embedding[:, 4] = 1.0  # 重置剩余比例
        self.layer_embedding[:, 5] = 1.0  # 重置上一动作
        
        # 重置时间统计
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0
        
        # 如果是第一次reset且没有初始化过FLAP策略
        if not hasattr(self, 'flap_init_done'):
            self.flap_preserve_ratios = self.initialize_with_flap()
            self.flap_init_done = True
            # 将FLAP的结果作为初始最佳策略
            self.best_strategy = self.flap_preserve_ratios
            # 评估FLAP策略的效果
            for ratio in self.flap_preserve_ratios:
                _, reward, _, info = self.step(ratio)
            self.best_reward = reward
            print(f"=> Initial FLAP strategy reward: {reward:.4f}")
            # 重置环境以开始RL训练
            return self.reset()
        
        return self.layer_embedding[0].copy()

    def _evaluate_model(self, model):
        """
        评估模型在LIBERO任务上的表现
        
        Args:
            model: 要评估的模型
            
        Returns:
            float: 平均成功率
        """
        # 创建临时配置，避免修改原始配置
        eval_cfg = copy.deepcopy(self.eval_config)
        
        # 确保评估时不使用wandb
        eval_cfg.use_wandb = False
        
        # 使用eval_libero评估模型
        eval_results = eval_libero(cfg=eval_cfg, model=model)
        
        # 从评估结果字典中获取成功率
        success_rate = eval_results.get('success_rate', 0.0)
        
        return success_rate
        

    def _calculate_reward(self, current_success_rate, compress_ratio):
        """计算奖励值
        
        Args:
            current_success_rate: 当前模型的成功率
            compress_ratio: 压缩比例
            
        Returns:
            reward: 奖励值
        """

        beta = 0.3  # 超参数，可以调整压缩率的权重
        # 总奖励 = 成功率比例 + beta * 压缩率
        reward = current_success_rate + beta * compress_ratio
        
        return reward

    def get_best_strategy(self):
        """返回搜索到的最佳策略"""
        return self.best_strategy

    def get_statistics(self):
        """返回搜索过程的统计信息"""
        return {
            'extract_time': self.extract_time,
            'fit_time': self.fit_time,
            'val_time': self.val_time
        } 

    def initialize_with_flap(self):
        """
        使用FLAP进行初始剪枝，并返回每层的保留率作为初始策略
        
        Returns:
            List[float]: 每层的保留率列表
        """
        print("=> Initializing with FLAP pruning...")
        
        # 创建模型副本进行FLAP剪枝
        flap_model = copy.deepcopy(self.model)
        flap_model.language_model.seqlen = 128
        
        # 记录原始每层的参数数量
        original_sizes = []
        for layer in flap_model.language_model.model.layers:
            # 记录attention和mlp的参数数量
            attn_params = sum(p.numel() for p in layer.self_attn.parameters())
            mlp_params = sum(p.numel() for p in layer.mlp.parameters())
            original_sizes.append(attn_params + mlp_params)
        
        # 执行FLAP剪枝
        from FLAP.lib.prune import prune_flap
        prune_flap(
            self.args,
            flap_model.language_model,
            self.tokenizer,
            self.device
        )
        
        # 计算每层的保留率
        preserved_ratios = []
        for i, layer in enumerate(flap_model.language_model.model.layers):
            # 计算剪枝后的参数数量
            current_attn_params = sum(p.numel() for p in layer.self_attn.parameters())
            current_mlp_params = sum(p.numel() for p in layer.mlp.parameters())
            current_total = current_attn_params + current_mlp_params
            
            # 计算保留率
            preserve_ratio = float(current_total) / original_sizes[i]
            preserved_ratios.append(preserve_ratio)
        
        print(f"=> FLAP initialization complete. Layer-wise preserve ratios: {preserved_ratios}")
        return preserved_ratios 