from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from simulation.openvla.experiments.robot.libero.run_libero_eval import GenerateConfig, eval_libero
import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b-finetuned-libero-spatial", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

cfg = GenerateConfig(
    pretrained_checkpoint="openvla/openvla-7b-finetuned-libero-spatial",
    task_suite_name="libero_spatial",
    center_crop=True,
)
eval_libero(cfg=cfg, model=vla)