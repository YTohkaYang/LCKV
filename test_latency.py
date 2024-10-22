import os
from collections.abc import Iterable

from models import LCKVLlamaConfig
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    logging,
)
from transformers.trainer_pt_utils import get_model_param_count
from accelerate import PartialState
from accelerate.utils import set_seed
import torch
import time
from tqdm import tqdm
set_seed(42)

logging.get_logger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.get_logger("transformers.generation.utils").setLevel(logging.ERROR)
logging.get_logger("transformers.models.llama.modeling_llama").setLevel(logging.ERROR)

prompt_text_2048 = """The meaning of life is the solution to life's problems.'

It was difficult to know whether the speech was a lie or not. At first Lily was determined to accept that her brother was deeply ill and needed my help in trying to help him. However, this was hardly the right way to treat him. He was a healthy young man and I didn't want to upset his sisters by making them feel as though they had been abandoned. So Lily couldn't see the point of my helping.

'Is it true that you only said you were ill because you were selfish?'

'Well, yes, and I'm sorry. And, in a way, yes. Because it's true that I care about you. But first and foremost, I care about my brother.'

'OK,' Lily said, clearly annoyed at the interference in her own life. 'I do see the point of you.'

'So why don't you, forgive me?'

'Why don't you forgive me for what?'

'The part I played in making you miserable.'

'I don't forgive myself. I'm simply uncomfortable to talk to people like you.'

'OK, well, maybe it's not fair of me to start talking to people about myself. My mum is not going to come on the same terms as me,' I said defensively.

'I guess not,' Lily said.

'Yes, she is coming round,' I said hopefully. 'Better late than never.'

'So why do you come to see me?' Lily asked me. 'I mean, I was surprised. I've been neglecting you. You don't like me,' she said, somewhat alarmed at the harshness of her tone.

'Don't ask me to explain,' I told her, 'I don't really know.'

'But I thought you wanted to help,' Lily said slowly. 'What were you expecting me to do? When she became ill, you just seemed so upset about it.'

'I'm not in a good place right now,' I admitted. 'I've always hated end-of-life situations. I just don't know how to deal with it, Lily. It makes me feel bad.'

'I don't think I understand,' Lily replied. 'Have you tried to talk to her?'

'OK, I'll try. She used to live with me, but then she left.'

'Do you want to find out why she's given you up?' Lily asked.

'She wouldn't talk to me about it,' I said, embarrassed by the pressure I was under. 'She's tried talking to me, but I'm not open to anything.'

'You don't look open to anything,' Lily said with a note of contempt. 'She didn't put it that way. She was in love with you. I thought she wanted you to marry her.'

'She didn't say anything like that,' I said softly, biting my tongue. 'She just... didn't talk to me.'

Lily's tone changed from brusque to disapproving. 'I don't care if you are in love with her. She's leaving you.'

'Lily, I want to know why she's leaving you.'

'Because she's giving you up,' Lily declared. 'That's why.'

'I don't want to know.' I glanced over my shoulder. The door was open a crack. I wondered how it was that my brother could have found his way out of it. I wondered if my brother had had help.

'You don't want to know the answer?' Lily persisted. 'Get me a drink. Maybe you can find out what you're looking for.'

I decided to go for it. Lily had the bottle of whisky that I bought for her on her birthday a few months earlier. It was a lovely bottle with a small picture of our father on it.

'Well, I'm not leaving you,' Lily told me. 'Go and sit with my mum. I'm going to check on Peter. You need to concentrate on getting through this morning. You can give your sister some solace later.'

'Lily, I don't want to see any of you,' I told her. 'You told me you weren't going to make it to the pub but I'm already here.'

'OK, don't make me answer that,' Lily said in a firm tone. 'Look, I'm not convinced that any of you are going to make it. I've been left alone for far too long. If I stay for a while, I might be able to take the edge off your stress, but I'm not going to make it my business to see how you're doing. Goodbye, Lily.'

'OK,' I told her. 'Bye.'

'Where are you going?' Lily asked me from the door.

'To see my mum,' I told her. 'I'm taking the day off.'

'OK,' Lily said with a slight smile, 'good luck with that.'

She then went downstairs to order something to eat from the freezer. I stayed with my mum for a few hours, because she wouldn't leave. She told me she was now staying with the sisters, and that she had just been to see Peter. I told her about Peter's wife and how she had made her decision.

'Don't you dare leave me,' she told me. 'I'm not going to let you have this bit between us any more.'

'You know you have to work some more,' I told her. 'I'm getting up, so will you have a nice day?'

She said she would. I was touched by the way she told me she didn't hate me. 'I don't want to hear about anything, just know that I'm thinking of you. No matter what happens, I'm there.'

Then, I packed up my things and made my way to Lily's house. I had to make an effort to get her to talk about her brother.

'OK,' I told her, 'I guess I don't have to talk about Peter. I don't want to ask any questions. But, you don't mind if I do, do you?'

'Not if you do,' Lily told me. 'But I'd like to know why you're upset.'

'You tell me when you want to.'

'OK,' Lily said, then shook her head. 'I don't think I'll be telling you. I'll make it my business to see Peter.'

'Good,' I said, somewhat disappointed.

'Goodbye.'

'Goodbye,' I said and walked away. I couldn't help but feel that the night was too soon for Lily to suddenly get so upset about something like this.

So, what was I going to do? I didn't like getting up at nine in the morning, not just to be alone but to get up. I felt as if I couldn't handle being alone for long periods of time. Maybe I would go for a walk. Or, if I wasn't feeling up to that, I would make some sandwiches and I would go for a stroll. I could go to the pub, and talk about something that didn't involve anything very serious. If I went to the pub and wasn't forced to say anything, then I could relax and get on with my day.

I went for a stroll down the hill and ended up going into London. It was a lovely early spring morning. The trees were still very early, so I stopped at a shop to buy some fresh produce for lunch. As I was buying it, a cyclist approached me and asked me how I was. I didn't know what to say. He was really nice, though, so I explained about the situations of my brother and sister.

'I'm not sure how you're supposed to take that,' he told me, not sure what to say, 'but, hopefully, everything will work out for the best. I'm going off to work today, so I'll see you when I get back to Tottenham.'

'Ok,' I told him.

'Goodbye,' he added.

I must have looked relieved when I said goodbye. He did head to work and I went home to find Lily sitting in her chair on her sofa.

'So how are you? How is the Wiltshire weather?'

Lily told me she had been back on a walk in that great, new National Trust park. 'It was lovely,' she said. 'The sheep ran wild.'

'Did you manage to see who was out hunting?' I asked. 'Lily, are you watching it on television?'

Lily nodded.

'OK,' I said. 'Do you want me to watch?'

'You'"""
prompt_text_512 = """The meaning of life is the solution to life's problems.'

It was difficult to know whether the speech was a lie or not. At first Lily was determined to accept that her brother was deeply ill and needed my help in trying to help him. However, this was hardly the right way to treat him. He was a healthy young man and I didn't want to upset his sisters by making them feel as though they had been abandoned. So Lily couldn't see the point of my helping.

'Is it true that you only said you were ill because you were selfish?'

'Well, yes, and I'm sorry. And, in a way, yes. Because it's true that I care about you. But first and foremost, I care about my brother.'

'OK,' Lily said, clearly annoyed at the interference in her own life. 'I do see the point of you.'

'So why don't you, forgive me?'

'Why don't you forgive me for what?'

'The part I played in making you miserable.'

'I don't forgive myself. I'm simply uncomfortable to talk to people like you.'

'OK, well, maybe it's not fair of me to start talking to people about myself. My mum is not going to come on the same terms as me,' I said defensively.

'I guess not,' Lily said.

'Yes, she is coming round,' I said hopefully. 'Better late than never.'

'So why do you come to see me?' Lily asked me. 'I mean, I was surprised. I've been neglecting you. You don't like me,' she said, somewhat alarmed at the harshness of her tone.

'Don't ask me to explain,' I told her, 'I don't really know.'

'But I thought you wanted to help,' Lily said slowly. 'What were you expecting me to do? When she became ill, you just seemed so upset about it.'

'I'm not in a good place right now,' I admitted. 'I've always hated end-of-life situations. I just don't know how to deal with it, Lily. It makes me"""
prompt_text_5 = "The meaning of life is"
prompt_text_4096 = prompt_text_2048*2


def empty_cache():
    for _ in range(10):
        torch.cuda.empty_cache()

class BinarySearch:
    """
    Binary Search w/o maximum limit, search the upper bound by doubling the index.
    """
    def __init__(self, min: int = 0):
        self.low = min
        self.high = None
        self.nxt = min + 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.high is None:
            return self.nxt
        elif self.low == self.high:
            raise StopIteration
        else:
            return self.nxt
    
    def report(self, idx: int, result: bool):
        if self.high is None:
            if idx <= self.low:
                if not result:
                    raise ValueError(f"We've proven that {self.low} is feasible, but {idx} is not.")
            else:
                if not result:
                    self.high = idx - 1
                    self.nxt = (self.low + self.high) // 2
                else:
                    self.low = idx
                    self.nxt = idx * 2
        else:
            if idx < self.low or idx > self.high:
                raise ValueError(f"Index {idx} is out of range [{self.low}, {self.high}]")
            else:
                if not result:
                    self.high = idx - 1
                else:
                    self.low = idx
                self.nxt = (self.low + self.high) // 2
        if self.nxt == self.low:
            self.nxt = self.high

class Streamer:
    def __init__(self, pbar):
        self.pbar = pbar
    
    def put(self, x):
        self.pbar.update(1)

    def end(self, *args, **kwargs):
        self.pbar.close()

def get_ds_model(model_name, config, dtype, cpu_offload, disk_offload, offload_dir, num_gpus = 1):
    import deepspeed
    import torch.distributed as dist
    from transformers.deepspeed import HfDeepSpeedConfig

    hidden_size = config.hidden_size
    # debug: ModuleNotFoundError: No module named 'mpi4py'
    ## launch with deepspeed <filename>.py
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    pin_memory = True

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": hidden_size * hidden_size,
            "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": 1,
        "wall_clock_breakdown": False,
    }

    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory)

    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=True,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=2 * (1 << 30),
        )
        ds_config["aio"] = {
          "block_size": 1048576,
          "queue_depth": 8,
          "thread_count": 1,
          "single_submit": False,
          "overlap_events": True,
        }

    # dschf = HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        torch_dtype=dtype
    )
    model = model.eval()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model

def get_hf_model(model_name, config, dtype, cpu_offload, disk_offload, offload_dir, num_gpus):
    if num_gpus == 1 and dtype != torch.int8:
        # Here we use a custom device_map instead of device_map == "auto"
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        if cpu_offload:
            # NOTE: We must put some weights on GPU. Otherwise, huggingface reports errors.
            device_map = {
                "model.embed_tokens.weight": 0,
                "model.norm": "cpu",
                "model.layers": "cpu",
                "lm_head.weight": 0,
            }
        elif disk_offload:
            device_map = {
                "model.embed_tokens.weight": 0,
                "model.norm": "disk",
                "model.layers": "disk",
                "lm_head.weight": 0,
            }
        else:
            device_map = None
        max_memory = None
    else:
        # Here we use device_map == "auto", but set a low `max_memory` threshold
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        device_map = "auto"
        if cpu_offload:
            # `max_memory` should be larger than the embedding.
            # We use 2GB here because the embeding of opt-175b is 1.2GB.
            max_memory = {k: "2GB" for k in range(num_gpus)}
        elif disk_offload:
            max_memory = {k: "2GB" for k in range(num_gpus)}
        else:
            max_memory = {k: "14GB" for k in range(num_gpus)}
        max_memory["cpu"] = "160GB"

    if dtype == torch.int8:
        kwargs = {"load_in_8bit": True}
    else:
        kwargs = {"torch_dtype": dtype}

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config,
        device_map=device_map, max_memory=max_memory,
        offload_folder=offload_dir, **kwargs)
    if device_map is None:
        model.cuda()

    model.eval()
    return model


distributed_state = PartialState()

def prepare(model: str, size: str, cpu_offload: str = "none", warmup: int = 2):
    """
    Prepare the tokenizer and model.

    Arguments:
        model (str): The model name. Options are "llama" and "lckv-llama".
        size (str): The model size. Options are "50m", "1.1b", "7b", and "30b".
        cpu_offload (str): The CPU offload strategy. Options are "none", "hf", and "ds".
        warmup (int): The number of warmup layers for LCKV-LLAMA.
    """

    CONFIG_MAPPING = {
        "50m": "configs/llama_tiny.json",
        "1.1b": "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T",
        "7b": "huggyllama/llama-7b", # "yahma/llama-7b-hf"
        "30b": "huggyllama/llama-30b",
    }

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    # prepare config
    if model == "lckv-llama":
        config = LCKVLlamaConfig.from_pretrained(CONFIG_MAPPING[size])
        config._attn_implementation = "flash_attention_2"
        start, end = warmup // 2, config.num_hidden_layers - warmup // 2
        layer_types = [(end - 1 if i in range(start, end) else i) for i in range(config.num_hidden_layers)]
        config.layer_types = "_".join(map(str, layer_types))
    
    elif model == "llama":
        config = AutoConfig.from_pretrained(CONFIG_MAPPING[size])
        config._attn_implementation = "flash_attention_2"

    else:
        raise ValueError(f"Unknown model {model}")
    
    # prepare model
    if cpu_offload == "none":
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.to(distributed_state.device)

    elif cpu_offload == "hf":
        model = get_hf_model(
            model_name = CONFIG_MAPPING[size], 
            config = config,
            dtype = torch.bfloat16, 
            cpu_offload = True,
            disk_offload = False, 
            offload_dir = None, 
            num_gpus = 1
        )
    
    elif cpu_offload == "ds":
        model = get_ds_model(
            model_name = CONFIG_MAPPING[size], 
            config = config,
            dtype = torch.bfloat16, 
            cpu_offload = True,
            disk_offload = False, 
            offload_dir = None
        )
    
    else:
        raise ValueError(f"Unknown CPU offload strategy {cpu_offload}")

    print("# of trainable parameters:", get_model_param_count(model, True))
    model.eval()

    return tokenizer, model


def inject_callback(model, callback):
    """inject a callback into the model.
    """
    forward_func = model.forward
    
    def forward(self, *args, **kwargs):
        result = forward_func(*args, **kwargs)
        callback()
        return result
    
    model.forward = forward.__get__(model, type(model))


def experiment(tokenizer, model, prompt, max_new_tokens, iterator=None, verbose=False):

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    # move to cuda
    input_ids = input_ids.to(distributed_state.device)
    model.eval()

    # Pre-warmup the GPU
    _ = model.generate(input_ids[:1,:1], do_sample=True, max_length=50)

    def callback():
        if callback.timer is None:
            callback.timer = time.time()
    inject_callback(model, callback)

    def run(num_return_sequences = 32):

        callback.timer = None

        if verbose:
            pbar = tqdm(total=max_new_tokens, leave=False, desc="Generating")
            streamer = Streamer(pbar)
        else:
            streamer = None

        start = time.time()

        output_sequences = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            streamer=streamer,
        )

        end = time.time()

        return start, callback.timer, end
    
    if isinstance(iterator, Iterable):

        for i in iterator:
            # print the current date and time
            try:
                start, mid, end = run(i)
                length = max_new_tokens
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] bsz: {i:3}, time: {end-start:.3f} = {mid-start:.3f} + {end-mid:.3f}, throughput: {i*length/(end-start):.3f} tokens/s")
            except Exception as e:
                if not str(e).startswith("CUDA out of memory."):
                    print(e)
                    raise
                return
            empty_cache()
    
    else:

        iterator = BinarySearch(0 if iterator is None else iterator)
        for i in iterator:
            try:
                start, mid, end = run(i)
                length = max_new_tokens
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] bsz: {i:3}, time: {end-start:.3f} = {mid-start:.3f} + {end-mid:.3f}, throughput: {i*length/(end-start):.3f} tokens/s")
                iterator.report(i, True)
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] bsz: {i:3}, FAIL")
                if not str(e).startswith("CUDA out of memory."):
                    print(e)
                    raise
                iterator.report(i, False)
            empty_cache()
        return iterator.nxt


def main():

    # do not consequectively run the same model, it will cause CUDA out of memory

    print(">>> 7b 2048+2048 llama")
    tokenizer, model = prepare("llama", "7b")
    experiment(tokenizer, model, prompt_text_2048, max_new_tokens=2048)

    print(">>> 7b 2048+2048 lckv-llama")
    tokenizer, model = prepare("lckv-llama", "7b")
    experiment(tokenizer, model, prompt_text_2048, max_new_tokens=2048)

    print(">>> 7b 2048+2048 lckv-llama w=10")
    tokenizer, model = prepare("lckv-llama", "7b", warmup=10)
    experiment(tokenizer, model, prompt_text_2048, max_new_tokens=2048)


if __name__ == "__main__":
    main()