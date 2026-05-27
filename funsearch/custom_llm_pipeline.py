import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import logging
import time
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Mxfp4Config
# HOSTNAME = os.uname().nodename


# ANSI color codes
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
# YELLOW = "\033[93m"
# CYAN = "\033[1;96m"   # Bold + Cyan


def print_color(msg, color):
    print(f"{color}{msg}{RESET}")
    
class CustomLLM(torch.nn.Module):
    def __init__(self, 
                samples_per_prompt: int, 
                seed: int,
                model_name=None, 
                log_path=None):
        super().__init__()
        self._samples_per_prompt = samples_per_prompt
        self.prompt_count = 0
        self.log_path = log_path

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Step 1: Quantization config
        start = time.perf_counter()
        print_color("🔧 Setting quantization config...", CYAN)

        if "oss" in model_name: 
            quant_config = Mxfp4Config(dequantize=False)
        else:
            # quant_config = BitsAndBytesConfig(load_in_8bit=True)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # bnb_4bit_quant_type="nf4",       # Optional: NF4 (NormalFloat4) quantization type recommended
                # bnb_4bit_use_double_quant=True,  # Optional: use double quantization to save memory
                # bnb_4bit_compute_dtype=torch.bfloat16  # Optional: computation dtype for faster training
            )

        
        print_color(f"⏱️ Quantization config time: {time.perf_counter()-start:.2f}s", YELLOW)

        # Step 2: Load tokenizer
        start = time.perf_counter()
        print_color("📦 Loading tokenizer...", CYAN)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print_color(f"⏱️ [{model_name}] Tokenizer load time: {time.perf_counter()-start:.2f}s", YELLOW)
        

        # Step 3: Device and memory info
        start = time.perf_counter()
        print_color("🧠 Inferring device map across GPUs...", CYAN)
        device_count = torch.cuda.device_count()
        # Build a conservative per-GPU memory budget based on real device memory.
        # This avoids overcommitting VRAM on smaller cards (e.g., 12GB GPUs).
        max_mem = {}
        for i in range(device_count):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            budget_gib = max(4.0, total_gib - 2.0)
            max_mem[i] = f"{budget_gib:.1f}GiB"
        # max_mem = {}
        # max_mem[0] = "20.0GiB"
        # max_mem[1] = "17.0GiB"
        
        print_color(f"🖥️ Found {device_count} CUDA devices: {max_mem}", MAGENTA)
        print_color(f"⏱️ Device info time: {time.perf_counter()-start:.2f}s", YELLOW)

        # Step 4: Load model
        start = time.perf_counter()
        print_color("�� Loading model weights with device map and quantization...", CYAN)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map="cuda:0" if "Qwen3-1.7B" in model_name else "auto",
                max_memory=max_mem,
                low_cpu_mem_usage=True,
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print_color("⚠️ OOM while loading model; retrying with reduced GPU budget.", YELLOW)
            reduced_mem = {i: "8.0GiB" for i in range(device_count)}
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map="auto",
                max_memory=reduced_mem,
                low_cpu_mem_usage=True,
            )

        # # model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        # # Qwen2.5-Coder-7B-Instruct does not have a padding token.
        # #  Asking to pad but the tokenizer does not have a padding token. 
        # # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.

        print_color("✅ Model loaded successfully.", GREEN)
        print_color(f"⏱️ Model load time: {time.perf_counter()-start:.2f}s", YELLOW)

        # Step 5: Print device map
        device_map = getattr(self.model, 'hf_device_map', None)
        if device_map:
            print_color("✅ Device map created:", GREEN)
            for k, v in device_map.items():
                print_color(f"  - {k} → {v}", GREEN)
        else:
            print_color("ℹ️ Device map not available as an attribute.", MAGENTA)

        # Step 6: Set to eval mode for inference
        print_color("🛠️ Setting model to inference mode...", CYAN)
        self.model.eval()

        print("Compiling model...")
        start_time = time.time()
        # https://huggingface.co/docs/transformers/en/perf_torch_compile
        try:
            print("Compiling model with reduce-overhead...")
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True) # "max-autotune", "reduce-overhead"
        except Exception as e:
            print("Compiling model with max-autotune...")
            self.model = torch.compile(self.model, mode="max-autotune", fullgraph=False) # "max-autotune", "reduce-overhead"
        end_time = time.time()
        print(f"Model compiled in {end_time - start_time:.2f} seconds")

        self.system_prompt = ""
        self.response_prompt = "\n### Response\n"

    def forward(self, input_ids):
        return self.model(input_ids)
    
    def draw_samples(self, prompt: str, max_new_tokens=500):

        start = time.perf_counter()
        print_color(f"🚀 Running test inference... max_new_tokens={max_new_tokens}", CYAN)
        
        # NOTE: the system prompt is in the doc of the file
        # This remove the things we do not care
        prompt_to_encode = self.system_prompt + prompt[4:].replace('\n###\n"""', '').replace('import funsearch\n', '').replace('@funsearch.run\n', '') + self.response_prompt

        # import pdb; pdb.set_trace()
        logging.info(f"######################### {prompt_to_encode} #########################")

        prompt = prompt_to_encode
        # inputs are sent to gpu0, where the tokenizer is 
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", padding=True).to("cuda:0") 
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, 
                # max_length=max_length, 
                max_new_tokens=max_new_tokens,
                num_return_sequences=self._samples_per_prompt, 
                # no_repeat_ngram_size=2,
                do_sample=True, 
                top_k=30,  # it was 40
                top_p=0.92, # it was 0.95
                repetition_penalty=1.1,
                temperature=0.8,
                # pad_token_id=self.tokenizer.pad_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
        print_color(f"⏱️ Inference time: {time.perf_counter()-start:.2f}s", YELLOW)
        print_color("\n📄 Output:", CYAN)

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        elapsed = time.perf_counter() - start
        print_color(f"🔢 Batch size {self._samples_per_prompt}: {elapsed:.3f}s", MAGENTA)


        samples = []
        idxs_valid = []
        for idx, response in enumerate(responses):
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                # print(response) # remove print prompt too crowded the output
            if len(response) == 0:
                # if it is not outputing values we skip the eval
                pass 
            else:
                if "return " in response:
                    # if there is not the keyword return skip this
                    samples.append(response)
                    idxs_valid.append(idx)

        
        # import pdb; pdb.set_trace()
        # we only log not empty programs
        self._log_all_responses_at_once(prompt, samples, self.prompt_count, idxs_valid)

        self.prompt_count += 1
        end_time = time.time()
        logging.info(f'[Not empty {len(samples)/len(responses):.2f}] {CYAN}[GPU {self.model.device}] Samples drawn in {end_time - start} seconds...{RESET}')

        return samples

    def _log_all_responses_at_once(self, prompt: str, all_responses: list, index_prompt: int, idxs_valid: list):
        """
        Log the prompt and response to a file.
        TODO: this is not efficient. The log should be written in a separate process.
        It is fast for now. It is not a bottleneck.
        """
        assert len(all_responses) == len(idxs_valid), "They should have the same length"

        start_time = time.time()
        if self.log_path is not None:
            with open(self.log_path / f"prompt_p_{index_prompt}.log", "a") as f: # saves the prompt in file
                f.write(prompt)

            for index_response, response in zip(idxs_valid, all_responses):
                with open(self.log_path / f"response_p_{index_prompt}_r_{index_response}.log", "a") as f:
                        f.write(str(response))

        end_time = time.time()


    def _log_prompt(self, prompt: str, index_prompt: int):
        # start_time = time.time()
        if self.log_path is not None:
            # Overwrite the prompt
            with open(self.log_path / f"prompt_p_{index_prompt}.log", "w") as f: # saves the prompt in file
                f.write(prompt)

    def _log_response(self, response: str, index_prompt: int, index_response: int):
        """
        Log the prompt and response to a file.
        TODO: this is not efficient. The log should be written in a separate process.
        It is fast for now. It is not a bottleneck.
        """
        # start_time = time.time()
        if self.log_path is not None:
            # Overwrite the prompt
            # with open(self.log_path / f"prompt_p_{index_prompt}.log", "a") as f: # saves the prompt in file
            #     f.write(prompt)
            with open(self.log_path / f"response_p_{index_prompt}_r_{index_response}.log", "a") as f:
                f.write(str(response))
        # end_time = time.time()
        # logging.info(f'Log written in {end_time - start_time} seconds...')

# Example usage
if __name__ == "__main__":


    llm = CustomLLM(samples_per_prompt=100, seed=42)
    import pdb; pdb.set_trace()

    prompt = "def fibonacci(n):\n"

    output = llm.draw_samples(prompt)
