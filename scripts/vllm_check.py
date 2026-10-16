"""vLLM start-up check (accept step of verify.sh): load Qwen3-0.6B and generate 4 short answers. A file, not stdin:
vLLM's engine starts a subprocess that re-imports the main module, which fails for code piped in through stdin."""
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    import os; llm = LLM(model=os.environ.get("STUDENT_06B", "Qwen/Qwen3-0.6B"), gpu_memory_utilization=0.5, max_model_len=4096)
    out = llm.generate(["What is 7*8? Put the answer in \\boxed{}."] * 4, SamplingParams(max_tokens=256, temperature=0.6))
    print("VLLM_OUT", [o.outputs[0].text[-60:] for o in out])
