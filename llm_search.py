import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True

from env.llm_pruning_env import LLMPruningEnv
from lib.agent import DDPG
from lib.utils import get_output_folder
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
from FLAP.models.hf_llama.modeling_llama import LlamaForCausalLM

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='LLM pruning search script')
    
    # env
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--cache_dir', type=str, default="llm_weights",
                       help='Directory to cache model weights')
    parser.add_argument('--preserve_ratio', type=float, default=0.5,
                       help='Target ratio to preserve weights')
    parser.add_argument('--lbound', default=0.5, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--metrics', type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN"])
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--unstr', action='store_true', help='Whether to use unstructured pruning')
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
    
    # ddpg
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for critic')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=0, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='discount factor')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    
    # noise
    parser.add_argument('--init_delta', default=0.5, type=float, help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float, help='delta decay during exploration')
    
    # training
    parser.add_argument('--train_episode', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--output', type=str, default='./pruned_models',
                       help='Output directory')
    parser.add_argument('--suffix', default=None, type=str, help='suffix for output directory')
    
    parser.add_argument('--seqlen', type=int, default=128,
                       help='Sequence length for calibration')
    
    # LIBERO evaluation parameters
    parser.add_argument('--libero_task_suite', type=str, default='libero_spatial',
                       choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                       help='LIBERO task suite for evaluation')
    parser.add_argument('--libero_num_trials', type=int, default=10,
                       help='Number of trials per task for evaluation')
    
    # Optional W&B parameters
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='W&B project name for logging')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity name for logging')
    
    args = parser.parse_args()
    return args


# def get_llm(model, cache_dir="llm_weights"):
#     """加载LLM模型"""
#     model = LlamaForCausalLM.from_pretrained(
#         model, 
#         torch_dtype=torch.float16, 
#         cache_dir=cache_dir, 
#         low_cpu_mem_usage=True
#     )
    
#     # 初始化bias
#     for i in range(len(model.model.layers)):
#         model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
#             torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device='cpu'))
#         model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
#             torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device='cpu'))
#         torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
#         torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
        
#     model.seqlen = 128
#     return model
def get_model_config(model) -> dict:
    """Get model configuration including attention features and intermediate sizes"""
    attention_features = []
    intermediate_sizes = []
    
    for layer in model.model.layers:
        attention_features.append(layer.self_attn.q_proj.out_features)
        intermediate_sizes.append(layer.mlp.gate_proj.out_features)
    
    text_config = {
        "model_type": "pruned_llama",
        "pad_token_id": 32000,
        "torch_dtype": "bfloat16",
        "vocab_size": 32064,
        "per_layer_attention_feature_size": attention_features,
        "per_layer_intermediate_size": intermediate_sizes
    }
    return {"text_config": text_config}

def train_and_prune(agent, env, num_episode, output_dir):
    """训练DDPG代理并执行剪枝"""
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    
    tfwriter = SummaryWriter(logdir=output_dir)
    text_writer = open(os.path.join(output_dir, 'log.txt'), 'w')
    
    while episode < num_episode:
        # 重置环境
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # 选择动作
        if episode == 0 and hasattr(env, 'flap_preserve_ratios'):
            # 第一个episode使用FLAP的结果
            action = env.flap_preserve_ratios[episode_steps]
        elif episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # 执行动作
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # 更新状态
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # 回合结束
            print('#{}: Last episode_reward:{:.4f} success_rate: {:.4f}, compress_ratio: {:.4f}'.format(
                episode, episode_reward, info['success_rate'], info['compress_ratio']))
            text_writer.write(
                '#{}: Last episode_reward:{:.4f} success_rate: {:.4f}, compress_ratio: {:.4f}\n'.format(
                    episode, episode_reward, info['success_rate'], info['compress_ratio']))
            
            final_reward = T[-1][0]
            
            # 更新策略
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    agent.update_policy()

            # 重置环境
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            # 记录日志
            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', env.best_reward, episode)
            tfwriter.add_scalar('info/success_rate', info['success_rate'], episode)
            tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)
            
            # 记录每层的保留率
            for i, preserve_rate in enumerate(env.strategy):
                tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(env.best_reward))
            text_writer.write('best policy: {}\n'.format(env.best_strategy))
    
    text_writer.close()
    tfwriter.close()
    
    # 使用最佳策略进行最终剪枝
    print("=> Using best strategy for final pruning:", env.best_strategy)
    env.reset()  # 重置环境
    for ratio in env.best_strategy:
        env.step(ratio)
    
    print(env.model)
    # 保存剪枝后的模型
    env.model.save_pretrained(output_dir)
    env.tokenizer.save_pretrained(output_dir)
     # 获取并保存新的配置
    import json
    config = get_model_config(env.model.language_model)
    
    # 如果已存在配置文件，先读取它
    config_path = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
            # 只更新text_config部分
            existing_config['text_config'] = config['text_config']
            config = existing_config
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"=> Pruned model saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, 
        # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = torch.device(args.device)
    model = model.to(device)

    # 创建环境
    env = LLMPruningEnv(
        model=model,
        tokenizer=tokenizer,
        args=args
    )

    # 创建输出目录
    base_folder_name = '{}_r{}'.format(
        args.model.split('/')[-1], 
        args.preserve_ratio
    )
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix
        
    output_dir = get_output_folder(args.output, base_folder_name)
    print('=> Output directory: {}'.format(output_dir))

    # 初始化DDPG代理
    nb_states = env.layer_embedding.shape[1]
    nb_actions = 1  # 每层一个动作（剪枝比例）
    args.rmsize = args.rmsize * len(model.language_model.model.layers)  # 为每层分配经验回放内存
    
    agent = DDPG(nb_states, nb_actions, args)
    
    # 训练并剪枝
    train_and_prune(agent, env, args.train_episode, output_dir) 