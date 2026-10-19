import torch
from micro_vllm import LLM, LLMConfig, ModelConfig, SchedulerConfig, CacheConfig, SamplingParams
from torch.profiler import profile, ProfilerActivity

def main():
    # 1. 正常初始化配置（尽量简单，跑一次即可）
    llm = LLM(
        LLMConfig(
            model_config=ModelConfig(
                model="/home/lenovo/huggingface/Qwen3-0.6B", # 换成你的绝对路径
                device="cuda",
                dtype=torch.float16,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                max_num_batched_tokens=64,
                max_num_seqs=2,
            ),
            cache_config=CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.8,
            ),
        )
    )

    prompts = [
        "Write a long detailed story about a robot.",
        "What is the capital of France?",
    ]
    params = SamplingParams(max_tokens=16, temperature=0.7)

    print(">>> 开始 Warmup (热身，防止初始化干扰 Profile 结果)...")
    llm.generate(prompts, params) # 先跑一次预热

    print(">>> 开始 Profile...")
    # 2. 开启 Profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        llm.generate(prompts, params)

    # 3. 打印结果（按 CUDA 耗时排序，看 GPU 到底在干嘛）
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    llm.shutdown()

if __name__ == "__main__":
    main()