import argparse
import os
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import torch
import shutil
import random
import time
import math

from cs336_basics.utils import SimpleMapReduce
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.bpe_trainer import train_bpe_impl
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.llm import Transformer
from cs336_basics.llm_train import (
    AdamW,
    get_batch,
    cross_entropy,
    save_checkpoint,
    load_checkpoint,
    gradient_clipping,
    get_lr_cosine_schedule,
)
from cs336_basics.decode import (
    decode,
)

BASE_DIR = os.path.join(os.path.dirname(__file__))

def prepare_dataset_chunk(tokenizer:BPETokenizer, tmp_path:str, input_path:str, start:int, end:int):
    if os.path.exists(tmp_path+".data"):
        return
    with open(input_path, mode="r") as f:
        token_count = 0
        f.seek(start)
        content = f.read(end - start)
        buffer = tokenizer.encode(content)
        # write_back(buffer, tmp_path)
        arr = np.array(buffer, dtype=np.int32)
        arr.tofile(tmp_path + ".tmp")
    os.rename(tmp_path + ".tmp", tmp_path + ".data")

def star_wrapper(args):
    prepare_dataset_chunk(*args)

def prepare_dataset(
    tokenizer_name: str,
    input_path: str,
) -> str:
    dataset_dir = os.path.join(BASE_DIR, "assets", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    mangled = os.path.split(input_path)[1].split(".")[0]
    mangled = tokenizer_name + "-" + mangled
    dataset_dir = os.path.join(dataset_dir, mangled)

    vocab_path = os.path.join(BASE_DIR, "assets", tokenizer_name, "vocab.json")
    merges_path = os.path.join(BASE_DIR, "assets", tokenizer_name, "merges.txt")

    tokenizer = BPETokenizer.from_files(vocab_path, merges_path,["<|endoftext|>"])

    os.makedirs(dataset_dir, exist_ok=True)
    with open(input_path, mode='rb') as f:
        boundaries = find_chunk_boundaries(f, mp.cpu_count() * 4, b"<|endoftext|>")

    args = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        tmp = os.path.join(dataset_dir, f"{i:03d}")
        args.append((tokenizer, tmp, input_path, start, end))

    with mp.Pool(processes=16) as pool:
        for _ in tqdm(pool.imap_unordered(star_wrapper, args), total=len(args), desc="Tokenizing"):
            pass
    return dataset_dir

def get_batch_wrapper(dataset_dir: str, batch_size: int, context_length: int, device: str):
    files = [f for f in os.listdir(dataset_dir) if f.endswith(".data")]
    while len(files) == 0:
        time.sleep(10)
        files = [f for f in os.listdir(dataset_dir) if f.endswith(".data")]

    choice = random.choice(files)
    # choice = files[0]
    dataset_path = os.path.join(dataset_dir, choice)
    count = os.path.getsize(dataset_path) // np.dtype(np.int32).itemsize
    mmap = np.memmap(dataset_path, dtype=np.int32, mode='r', shape=(count,))
    print(f"get batch from {choice}")
    return get_batch(mmap, batch_size, context_length, device)

def clear_ckpt(ckpt_dir: str):
    ckpts = [int(f.split(".")[0]) for f in os.listdir(ckpt_dir)]
    if len(ckpts) > 15:
        ckpts.sort()
        for ckpt in ckpts[:-10]:
            os.remove(os.path.join(ckpt_dir, str(ckpt)+".ckpt"))


def train_model(args):
    # print("🚀 Training mode")
    # print(f"Dataset: {args.dataset}")
    # print(f"Vocab: {args.vocab}")
    # print(f"Checkpoint dir: {args.ckpt_dir}")

    # tokenize the dataset
    # dataset_path = prepare_dataset("owt_train_tokenizer-32000", args.dataset)
    tokenizer_name = "owt_train_tokenizer-32000"
    p = mp.Process(target=prepare_dataset, args = (tokenizer_name, args.dataset))
    p.start()

    dataset_dir = os.path.join(BASE_DIR, "assets", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    mangled = os.path.split(args.dataset)[1].split(".")[0]
    mangled = tokenizer_name + "-" + mangled
    dataset_dir = os.path.join(dataset_dir, mangled)

    device = "cuda:0"
    dtype = torch.bfloat16
    vocab_size = 32000
    context_length = 512
    d_model = 1024
    n_layers = 24
    n_heads = 16
    d_ff = 2688
    rope_theta = 10000
    batch_size = 16
    accum_steps=8
    total_tokens = 1000000000

    # # new-newer
    # dtype = torch.bfloat16
    # vocab_size = 32000
    # context_length = 2048
    # d_model = 512
    # n_layers = 8
    # n_heads = 16
    # d_ff = 1344
    # rope_theta = 10000
    # batch_size = 8
    # total_tokens = 1000000000

    # # new-newer2
    # dtype = torch.bfloat16
    # vocab_size = 32000
    # context_length = 2048
    # d_model = 768
    # n_layers = 12
    # n_heads = 16
    # d_ff = 2048
    # rope_theta = 10000
    # batch_size = 4
    # accum_steps=8
    # total_tokens = 1000000000
    
    model = Transformer(
        vocab_size,
        context_length,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        rope_theta,
        device=device,
        dtype=dtype
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # print(model.state_dict().keys())
    checkpoint_dir = args.ckpt_dir

    optimizer = AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.01)
    max_norm = 0.1
    
    step = int(total_tokens / batch_size / context_length)
    start = 0
    print(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpts = [int(f.split(".")[0]) for f in os.listdir(checkpoint_dir)]
    if len(ckpts):
        ckpts.sort()
        print(ckpts)
        ckpt = str(ckpts[-1]) + ".ckpt"
        start = load_checkpoint(os.path.join(checkpoint_dir, ckpt), model, optimizer)
        start += 1
        for pg in optimizer.param_groups:
            pg['weight_decay'] = 0.05
            pg['betas']=(0.88,0.95)
    print(start, step)
    for iter in range(start, step):
        if 0 <= iter <= 400:
            accum_steps=1
        elif 400 < iter <= 800:
            accum_steps=2
        elif 800 < iter <= 4000:
            accum_steps=4
        else:
            accum_steps=8
        total_loss=0
        start=time.time()
        for _ in range(accum_steps):
            x, y = get_batch_wrapper(dataset_dir, batch_size, context_length, device)
            loss = cross_entropy(model(x), y)/accum_steps
            loss.backward()
            total_loss += loss.detach()

        gradient_clipping(model.parameters(), max_norm)
        lr = get_lr_cosine_schedule(iter, 1e-3, 1e-5, int(2000), 10000)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()
        optimizer.zero_grad()
        end = time.time()
        print(iter, f"loss {total_loss.item():.2f} perplexity {math.exp(total_loss):.2f} lr {lr} speed {batch_size * accum_steps * context_length // (end - start)} tokens/s")
        # print(iter, f"loss {loss.item():.2f} perplexity {torch.exp(loss.detach()).item():.2f} lr {lr}")
        if iter % 200 == 0 and iter > 0:
            save_checkpoint(model, optimizer, iter, os.path.join(checkpoint_dir, str(iter)+".ckpt"))
            clear_ckpt(checkpoint_dir)
            torch.cuda.empty_cache()
    p.join()

def train_wrapper(args):
    while True:
        try:
            train_model(args)
            break
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("out of memory when training")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
            
def write_eval_txt(path: str, results, ppl, max_examples: int = 10000000):
    """
    把 infer() 的 return 内容写入 txt，方便助教批改
    max_examples 参数不需要动
    """
    with open(path, "w", encoding="utf-8") as f:
        # ===== 顶部：全局信息 =====
        f.write(f"ppl\t{ppl:.4f}\n")
        f.write(f"num_examples\t{len(results)}\n\n")

        # ===== 逐样本：只写前 max_examples 个防炸文件 =====
        for i, r in enumerate(results[:max_examples]):
            f.write(f"### example {i}\n")

            # input
            input_text = (r.get("input_text", "") or "").replace("\n", " ")
            if len(input_text) > 200:
                input_text = input_text[:200] + " ...[truncated]"
            f.write(f"input\t{input_text}\n")

            # tokens - 清理换行符，确保在一行内
            tokens = r.get("tokens", [])
            # 将每个 token 中的换行符替换为空格
            cleaned_tokens = [str(t).replace("\n", " ").replace("\r", " ") for t in tokens]
            f.write("tokens\t" + " ".join(cleaned_tokens) + "\n")

            # logits 
            logits = r.get("logits", [])
            f.write("logits\t" + " ".join(f"{l:.4f}" for l in logits) + "\n")

            # optional generated_text
            if "generated_text" in r:
                gen = (r["generated_text"] or "").replace("\n", " ")
                if len(gen) > 200:
                    gen = gen[:200] + " ...[truncated]"
                f.write(f"generated_text\t{gen}\n")

            f.write("\n")  # block 之间空行

def infer(args):
    # print("🔮 Inference mode")
    # print(f"Checkpoint: {args.ckpt}")
    # print(f"Vocab: {args.vocab}")
    # print(f"Test file: {args.test}")
    # # 👉 这里写推理逻辑
    checkpoint_dir = args.ckpt
    tokenizer_name = "owt_train_tokenizer-32000"

    vocab_path = os.path.join(BASE_DIR, "assets", tokenizer_name, "vocab.json")
    merges_path = os.path.join(BASE_DIR, "assets", tokenizer_name, "merges.txt")

    tokenizer = BPETokenizer.from_files(vocab_path, merges_path,["<|endoftext|>"])

    device = "cuda:0"
    dtype = torch.bfloat16
    vocab_size = 32000
    context_length = 512
    d_model = 1024
    n_layers = 24
    n_heads = 16
    d_ff = 2688
    rope_theta = 10000
    batch_size = 16
    accum_steps=8
    total_tokens = 1000000000

    # # new-newer
    # dtype = torch.bfloat16
    # vocab_size = 32000
    # context_length = 2048
    # d_model = 512
    # n_layers = 8
    # n_heads = 16
    # d_ff = 1344
    # rope_theta = 10000
    # batch_size = 8
    # total_tokens = 1000000000

    # dtype = torch.bfloat16
    # vocab_size = 32000
    # context_length = 2048
    # d_model = 768
    # n_layers = 12
    # n_heads = 16
    # d_ff = 2048
    # rope_theta = 10000
    # batch_size = 4
    # accum_steps=8
    # total_tokens = 1000000000
    
    model = Transformer(
        vocab_size,
        context_length,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        rope_theta,
        device=device,
        dtype=dtype
    )

    # load the latest checkpoint's weight
    ckpts = [int(f.split(".")[0]) for f in os.listdir(checkpoint_dir)]
    if len(ckpts):
        ckpts.sort()
        print(ckpts)
        ckpt = str(ckpts[-1]) + ".ckpt"
        load_checkpoint(os.path.join(checkpoint_dir, ckpt), model, None)
    
    with open(args.test, mode='r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_loss = 0.0
    total_tokens = 0
    for line in lines:
        input_ids = tokenizer.encode(line)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.forward(input_tensor[:, :-1])  # 预测下一个token
            targets = input_tensor[:, 1:]
            loss = cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()


    results = []
    for line in lines:
        input_ids = tokenizer.encode(line)
        output_tokens, probs_per_token = decode(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=input_ids,
            max_new_tokens=50,
            temperature=0.5,
            top_p=1,
            device=device,
            verbose=False
        )
        results.append(
            {
                "input_text": line,
                "tokens": output_tokens,
                "logits": probs_per_token
            }
        )

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"\n>>> Test Perplexity: {ppl:.4f}")
    for result in results:
        #lzy 修改
        print(tokenizer.decode(result['tokens']), f"tokens {len(result['tokens'])}")
    write_eval_txt(args.valid_path, results, ppl)
    return results, ppl


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transformer training and inference script")

    # 创建互斥参数组（只能选一个）
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Run in training mode")
    mode_group.add_argument("--infer", action="store_true", help="Run in inference mode")

    # 通用参数
    parser.add_argument("--vocab", type=str, help="Path to vocab file")

    # 训练模式参数
    parser.add_argument("--dataset", type=str, help="Path to training dataset")
    parser.add_argument("--ckpt_dir", type=str, help="Directory to save checkpoints")

    # 推理模式参数
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint for inference")
    parser.add_argument("--test", type=str, help="Path to test dataset")
    parser.add_argument("--valid_path", type=str, help="valid path")
    args = parser.parse_args()

    if args.train:
        train_wrapper(args)
    elif args.infer:
        infer(args)