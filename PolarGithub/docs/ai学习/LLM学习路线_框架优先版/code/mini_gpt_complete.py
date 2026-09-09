"""教育用途的单文件字符级 MiniGPT。

包含完整闭环：
文本 -> Tokenizer -> Batch -> Embedding -> Transformer -> Logits
-> Cross Entropy -> Backward -> AdamW -> Checkpoint -> Generate

这个程序强调可读性与可观察性，不追求大模型训练速度。
"""

from __future__ import annotations

import argparse
##处理命令行参数
import math
import random
##随机数库
import time
##可以统计训练时间
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
##类型注解

import torch
import torch.nn as nn
import torch.nn.functional as F
##函数形式的神经网络运算


@dataclass
class GPTConfig:
    """集中保存决定模型结构的超参数。"""

    vocab_size: int
    block_size: int = 64
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.1

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        if self.block_size < 2:
            raise ValueError("block_size 至少为 2")
        if self.vocab_size < 2:
            raise ValueError("词表至少需要两个 token")


class CharTokenizer:
    """将每个 Unicode 字符视为一个 token 的教学 Tokenizer。"""

    unknown_token = "<unk>"

    def __init__(self, text: str) -> None:
        characters = sorted(set(text))
        self.tokens = [self.unknown_token] + characters
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for index, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        unknown_id = self.stoi[self.unknown_token]
        return [self.stoi.get(character, unknown_id) for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(int(index), self.unknown_token) for index in ids)

    def state_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CharTokenizer":
        tokenizer = cls.__new__(cls)
        tokenizer.tokens = list(state["tokens"])
        tokenizer.stoi = {
            token: index for index, token in enumerate(tokenizer.tokens)
        }
        tokenizer.itos = {
            index: token for index, token in enumerate(tokenizer.tokens)
        }
        return tokenizer


class CausalSelfAttention(nn.Module):
    """带因果遮罩的多头自注意力。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # 一次线性变换同时得到 Q、K、V，最后一维从 C 变为 3C。
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape

        qkv = self.qkv(x)
        query, key, value = qkv.split(self.n_embd, dim=-1)

        # [B,T,C] -> [B,T,H,D] -> [B,H,T,D]
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_head, self.head_dim
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        # [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T]
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.causal_mask[:, :, :sequence_length, :sequence_length] == 0,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # [B,H,T,T] @ [B,H,T,D] -> [B,H,T,D]
        context = weights @ value
        # [B,H,T,D] -> [B,T,H,D] -> [B,T,C]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.residual_dropout(self.output_projection(context))


class FeedForward(nn.Module):
    """Transformer Block 中逐位置运行的两层 MLP。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """Pre-Norm 结构的 Transformer Block。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_before_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_before_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接保留原信息，同时叠加子层产生的新信息。
        x = x + self.attention(self.norm_before_attention(x))
        x = x + self.mlp(self.norm_before_mlp(x))
        return x


class MiniGPT(nn.Module):
    """从 token id 到词表 logits 的完整 GPT。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        # 输入词向量表与输出分类权重共享，形状均为 [V,C]。
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"输入长度 {sequence_length} 超过 block_size "
                f"{self.config.block_size}"
            )

        positions = torch.arange(sequence_length, device=idx.device)
        token_vectors = self.token_embedding(idx)
        position_vectors = self.position_embedding(positions)
        x = self.dropout(token_vectors + position_vectors)

        if trace:
            print(f"token ids             {tuple(idx.shape)}")
            print(f"token embedding       {tuple(token_vectors.shape)}")
            print(f"position embedding    {tuple(position_vectors.shape)}")
            print(f"combined hidden state {tuple(x.shape)}")

        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if trace:
                print(f"transformer block {index:<2} {tuple(x.shape)}")

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, -1),
                targets.reshape(batch_size * sequence_length),
            )

        if trace:
            print(f"final hidden state    {tuple(hidden.shape)}")
            print(f"logits                {tuple(logits.shape)}")
            if targets is not None:
                print(f"targets               {tuple(targets.shape)}")
                print(f"loss                   scalar = {loss.item():.4f}")

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.config.block_size :]
            logits, _ = self(idx_context)
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    effective_k = min(top_k, next_logits.size(-1))
                    threshold = torch.topk(next_logits, effective_k).values[:, -1:]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
        return idx


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return torch.device(requested)


def split_token_stream(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    split_index = int(0.9 * len(data))
    return data[:split_index], data[split_index:]


def validate_data_length(
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    block_size: int,
) -> None:
    minimum = block_size + 1
    if len(train_data) < minimum or len(validation_data) < minimum:
        raise ValueError(
            "语料过短。训练集和验证集都必须至少包含 "
            f"block_size + 1 = {minimum} 个 token；"
            "请增加语料或减小 --block-size。"
        )


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for split_name, data in (
        ("train", train_data),
        ("validation", validation_data),
    ):
        losses = torch.zeros(eval_iters)
        for iteration in range(eval_iters):
            x, y = get_batch(
                data,
                batch_size,
                model.config.block_size,
                device,
            )
            _, loss = model(x, y)
            assert loss is not None
            losses[iteration] = loss.detach().cpu()
        result[split_name] = losses.mean().item()
    model.train()
    return result


def build_optimizer(
    model: MiniGPT,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    # 矩阵参数使用权重衰减；bias 与 LayerNorm 等一维参数不衰减。
    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        target = decay_parameters if parameter.dim() >= 2 else no_decay_parameters
        target.append(parameter)

    parameter_groups = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(parameter_groups, lr=learning_rate)


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    step: int,
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(model.config),
        "tokenizer": tokenizer.state_dict(),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{path}")
    return torch.load(path, map_location=device)


def parse_arguments() -> argparse.Namespace:
    default_corpus = Path(__file__).with_name("tiny_corpus.txt")
    parser = argparse.ArgumentParser(description="训练一个教育用途的字符级 MiniGPT")
    parser.add_argument("--text", type=Path, default=default_corpus)
    parser.add_argument("--output", type=Path, default=Path("mini_gpt_checkpoint.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="我爱")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)
    device = choose_device(args.device)

    if not args.text.exists():
        raise FileNotFoundError(f"语料文件不存在：{args.text}")
    text = args.text.read_text(encoding="utf-8")

    checkpoint = None
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, device)
        tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"])
        config = GPTConfig(**checkpoint["config"])
    else:
        tokenizer = CharTokenizer(text)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )

    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, validation_data = split_token_stream(encoded)
    validate_data_length(train_data, validation_data, config.block_size)

    model = MiniGPT(config).to(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    start_step = 0
    best_validation_loss = float("inf")

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        print(f"已从 {args.resume} 恢复，将从 step {start_step} 继续")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"train_tokens={len(train_data)}, validation_tokens={len(validation_data)}")
    print(f"parameters={parameter_count:,}")

    if args.trace:
        trace_x, trace_y = get_batch(
            train_data,
            min(2, args.batch_size),
            config.block_size,
            device,
        )
        model.eval()
        with torch.no_grad():
            model(trace_x, trace_y, trace=True)
        model.train()

    training_started = time.perf_counter()
    model.train()
    for step in range(start_step, args.max_steps):
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(
                model,
                train_data,
                validation_data,
                args.batch_size,
                args.eval_iters,
                device,
            )
            print(
                f"step={step:5d} "
                f"train_loss={losses['train']:.4f} "
                f"validation_loss={losses['validation']:.4f}"
            )
            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]
                save_checkpoint(
                    args.output,
                    model,
                    optimizer,
                    tokenizer,
                    step,
                    best_validation_loss,
                )
                print(f"已保存更优 Checkpoint：{args.output}")

        x, y = get_batch(
            train_data,
            args.batch_size,
            config.block_size,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.perf_counter() - training_started
            print(
                f"train step={step:5d} "
                f"batch_loss={loss.item():.4f} "
                f"grad_norm={float(gradient_norm):.3f} "
                f"elapsed={elapsed:.1f}s"
            )

    # 使用验证集上最优的参数进行最终生成。
    if args.output.exists():
        best_checkpoint = load_checkpoint(args.output, device)
        model.load_state_dict(best_checkpoint["model"])

    prompt_ids = tokenizer.encode(args.prompt)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    generated_text = tokenizer.decode(generated[0].tolist())
    print("\n===== 生成结果 =====")
    print(generated_text)


if __name__ == "__main__":
    main()
