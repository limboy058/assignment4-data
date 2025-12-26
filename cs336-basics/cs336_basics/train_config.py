
from pathlib import Path
from omegaconf import OmegaConf


from dataclasses import dataclass, field


from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class PathsConfig:
    train_bin: Path = Path('/mnt/mnt/zjdx/tyhcyq/valid_corpus_3/corpus.bin')
    valid_bin: Path = Path('/mnt/mnt/zjdx/tyhcyq/paloma_c4_100_domain.bin')
    model_output: Path = Path('/mnt/mnt/zjdx/tyhcyq/ckpt/run_1Bdata')


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512  #已修改512
    d_model: int = 768
    d_ff: int = 2048  # floor(d_model * 8/3 / 64) * 64
    num_layers: int = 12
    num_heads: int = 12
    rope_theta: float | None = 10000.0


@dataclass
class TrainingConfig:
    seed: int = 0
    dtype: str = "bfloat16"
    train_batch_size: int = 128
    eval_batch_size: int = "${training.train_batch_size}"
    train_steps: int = 100_000
    gradient_accumulation_steps: int = 1
    compile: bool = True
    eval_iterations: int = 1_000
    eval_interval: int = 2_000
    max_grad_norm: float | None = 1.0
    device: str = "cuda"
    lr: float = 1e-3
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str = 'cs336_lab4'
    wandb_entity: str = 'limboy-east-china-normal-university'
    log_interval: int = 20
    save_checkpoints: bool = True

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def register_configs():
    OmegaConf.register_new_resolver("eval", eval)
    cs = ConfigStore.instance()
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="paths", name="base_paths", node=PathsConfig)
    cs.store(name="base_config", node=Config)
