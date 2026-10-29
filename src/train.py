"""QLoRA fine-tuning of Qwen2.5-VL for asbestos detection / type identification.

Example (single GPU)::

    python -m src.train --modality microscopic --data_dir $DATA \
        --manifest $DATA/manifest_aug.jsonl --output_dir $RUNS/micro \
        --epochs 3 --batch_size 2 --grad_accum 8 --learning_rate 1e-4 \
        --tile_size 512 --tile_label_source boxes

Writes to ``output_dir``: the LoRA adapter, the processor, ``label_space.json``
(the exact class mapping used, including merges/renames) and
``train_config.json`` (so evaluation reproduces prompts and tiling).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.dataset import AsbestosVLMDataset, DataConfig, VLMCollator
from src.labels import add_label_args, label_space_from_args
from src.model import DEFAULT_MODEL, compute_dtype, load_model_for_training, load_processor

log = logging.getLogger("train")


class BalancedTrainer(Trainer):
    """Trainer with optional class-balanced (inverse frequency) sampling."""

    def __init__(self, *args, sample_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def _get_train_sampler(self, *args, **kwargs):
        if self.sample_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        return WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights), replacement=True)


def add_data_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("data")
    g.add_argument("--modality", choices=["microscopic", "smartphone", "unified"], required=True)
    g.add_argument("--data_dir", required=True, help="Root that manifest image paths are relative to.")
    g.add_argument("--manifest", default=None, help="Default: <data_dir>/manifest.jsonl")
    g.add_argument("--task", choices=["binary", "type", "both"], default="both")
    g.add_argument("--tile_size", type=int, default=512, help="Tile size for microscopic images; 0 disables.")
    g.add_argument("--tile_overlap", type=float, default=0.0)
    g.add_argument("--tile_all_modalities", action="store_true", help="Also tile smartphone images.")
    g.add_argument("--max_tiles_per_image", type=int, default=16, help="Training cap (0 = no cap).")
    g.add_argument("--tile_label_source", choices=["image", "boxes"], default="image",
                   help="'boxes': label each tile from the annotated fibres it contains.")
    g.add_argument("--empty_tile_policy", choices=["drop", "negative", "inherit"], default="drop",
                   help="Tiles of positive images that contain no annotated fibre (boxes mode). "
                        "'negative' labels them with the first non-asbestos class in the config.")
    g.add_argument("--no_modality_hint", action="store_true", help="Do not tell the model the image type.")
    g.add_argument("--online_aug", action="store_true", help="Random flips/rot90 at load time.")
    g.add_argument("--shuffle_options", action="store_true", help="Shuffle option order in type prompts.")
    g.add_argument("--exclude_augmented", action="store_true", help="Ignore offline augmented records.")
    g.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    g.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)


def data_config_from_args(args) -> DataConfig:
    return DataConfig(
        modality=args.modality,
        tasks=("binary", "type") if args.task == "both" else (args.task,),
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_modalities=("microscopic", "smartphone") if args.tile_all_modalities else ("microscopic",),
        max_tiles_per_image=args.max_tiles_per_image,
        tile_label_source=args.tile_label_source,
        empty_tile_policy=args.empty_tile_policy,
        modality_hint=not args.no_modality_hint,
        online_aug=args.online_aug,
        shuffle_options=args.shuffle_options,
        exclude_augmented=args.exclude_augmented,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    add_label_args(ap)
    g = ap.add_argument_group("model")
    g.add_argument("--model_name", default=DEFAULT_MODEL)
    g.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    g.add_argument("--lora_r", type=int, default=16)
    g.add_argument("--lora_alpha", type=int, default=32)
    g.add_argument("--lora_dropout", type=float, default=0.05)
    g.add_argument("--tune_vision", action="store_true", help="Also LoRA-adapt the vision encoder.")
    g.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    g = ap.add_argument_group("training")
    g.add_argument("--output_dir", required=True)
    g.add_argument("--epochs", type=float, default=3)
    g.add_argument("--batch_size", type=int, default=2)
    g.add_argument("--eval_batch_size", type=int, default=4)
    g.add_argument("--grad_accum", type=int, default=8)
    g.add_argument("--learning_rate", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.0)
    g.add_argument("--warmup_ratio", type=float, default=0.03)
    g.add_argument("--lr_scheduler", default="cosine")
    g.add_argument("--eval_steps", type=int, default=0, help="Evaluate/save every N steps (0 = every epoch).")
    g.add_argument("--early_stopping_patience", type=int, default=3, help="0 disables.")
    g.add_argument("--balance", action="store_true", help="Class-balanced sampling.")
    g.add_argument("--num_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1)
    g.add_argument("--save_total_limit", type=int, default=2)
    g.add_argument("--no_gradient_checkpointing", action="store_true")
    g.add_argument("--resume", default="auto", help="'auto' (latest checkpoint in output_dir), 'no', or a path.")
    g.add_argument("--report_to", default="none", help="none | tensorboard | wandb")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max_train_samples", type=int, default=0, help="Debug: truncate the training set.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    space = label_space_from_args(args, args.modality)
    log.info("\n%s", space.describe())
    cfg = data_config_from_args(args)
    manifest = args.manifest or str(Path(args.data_dir) / "manifest.jsonl")

    train_ds = AsbestosVLMDataset(manifest, args.data_dir, space, cfg, split="train", train=True, seed=args.seed)
    if args.max_train_samples:
        train_ds.items = train_ds.items[: args.max_train_samples]
    eval_cfg = DataConfig.from_dict({**cfg.to_dict(), "tile_label_source": "image", "online_aug": False})
    val_ds = AsbestosVLMDataset(manifest, args.data_dir, space, eval_cfg, split="val", train=False, seed=args.seed)
    if len(train_ds) == 0:
        raise SystemExit("Empty training set - check --modality, --manifest and the class config.")
    log.info("Train: %s", train_ds.summary())
    log.info("Val:   %s", val_ds.summary())

    space.to_json(out / "label_space.json")
    with open(out / "train_config.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "data_config": cfg.to_dict()}, fh, indent=2)

    processor = load_processor(args.model_name, args.min_pixels, args.max_pixels)
    model = load_model_for_training(
        args.model_name, args.quant, args.lora_r, args.lora_alpha, args.lora_dropout,
        args.tune_vision, not args.no_gradient_checkpointing, args.attn_implementation,
    )

    has_val = len(val_ds) > 0
    strategy = ("steps" if args.eval_steps else "epoch") if has_val else "no"
    dtype = compute_dtype()
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        optim="paged_adamw_8bit" if args.quant != "none" and torch.cuda.is_available() else "adamw_torch",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        logging_steps=10,
        eval_strategy=strategy,
        save_strategy=strategy if has_val else ("steps" if args.eval_steps else "epoch"),
        eval_steps=args.eval_steps or None,
        save_steps=args.eval_steps or 500,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,  # enabled on the model in load_model_for_training
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=max(0, args.num_workers),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        max_grad_norm=1.0,
    )
    callbacks = []
    if has_val and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if has_val else None,
        data_collator=VLMCollator(processor, mode="train"),
        callbacks=callbacks,
        sample_weights=train_ds.sample_weights() if args.balance else None,
    )

    resume = None
    if args.resume == "auto":
        resume = get_last_checkpoint(str(out))
        if resume:
            log.info("Resuming from %s", resume)
    elif args.resume not in ("no", "none", ""):
        resume = args.resume
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    if has_val:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(str(out))  # LoRA adapter (best checkpoint if validation was available)
    processor.save_pretrained(str(out))
    trainer.save_state()
    log.info("Saved adapter, processor and label space to %s", out)


if __name__ == "__main__":
    main()
