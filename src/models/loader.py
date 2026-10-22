"""Chat-model wrapper. One `.chat(messages)` interface so the agent never cares which model runs."""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_CODER_1_5B = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class ChatModel:
    def __init__(self, model_id: str = QWEN_CODER_1_5B, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # GPU: bfloat16 (native on RTX 40xx, same exponent range as fp32 so no overflow like fp16).
        # CPU: float32, since laptop CPUs have no fast bf16 matmul and 1.5B fp32 is only ~6 GB.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def chat(self, messages: List[Message], max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        # apply_chat_template wraps the turns in the model's own <|im_start|> format.
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]  # strip the prompt
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
