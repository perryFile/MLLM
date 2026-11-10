import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONTEXT_LENGTH, DROPOUT, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE
from dataset import (
    DATASET_NAME,
    DialogDataset,
    IGNORE_INDEX,
    load_conversation_rows,
    load_daily_dialog_rows,
    split_rows,
)
from models.model import MiniGPT
from tokenizer import Tokenizer


OVERFIT_EXAMPLES = 8


def parse_args():
    parser = argparse.ArgumentParser(description=f"Train MiniGPT on {DATASET_NAME}")
    parser.add_argument("--mode", choices=("overfit", "full", "pretrain"), default="overfit")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Start from the weights in this checkpoint, e.g. miniGPT_pretrain_best.pth",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="Retrain tokenizer.json on combined DailyDialog + conversation training splits",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop full and pretrain mode after this many epochs without a better validation loss (0 disables)",
    )
    return parser.parse_args()


def build_optimizer(model, learning_rate, weight_decay):
    decay_params = []
    no_decay_params = []
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.dim() >= 2 and "embedding" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))


def build_scheduler(optimizer, total_steps, warmup_ratio=0.1, min_lr_ratio=0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if total_steps > 1 else 0

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            valid_targets = y != IGNORE_INDEX
            token_count = valid_targets.sum().item()
            total_loss += criterion(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item()
            total_tokens += token_count
            correct += ((logits.argmax(dim=-1) == y) & valid_targets).sum().item()

    return total_loss / total_tokens, correct / total_tokens


def save_checkpoint(path, model, optimizer, epoch, args, loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "mode": args.mode,
            "model_config": {
                "vocab_size": model.vocab_size,
                "context_length": model.context_length,
                "embedding_dim": model.embedding_dim,
                "num_heads": model.blocks[0].num_heads if len(model.blocks) > 0 else NUM_HEADS,
                "num_layers": len(model.blocks),
                "tie_weights": model.tie_weights,
            },
        },
        path,
    )


def _train_combined_tokenizer(tokenizer, seed):
    conv_train, _ = split_rows(load_conversation_rows(), seed=seed)
    pretrain_train, _ = load_daily_dialog_rows()
    combined_texts = (
        row["Text"] + "<SEP>" + row["Answer"] + "<END>"
        for row in (pretrain_train + conv_train)
    )
    tokenizer.train(combined_texts, vocab_size=VOCAB_SIZE)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    epochs = args.epochs if args.epochs is not None else (100 if args.mode == "overfit" else 15)
    batch_size = args.batch_size if args.batch_size is not None else (1 if args.mode == "overfit" else 32)
    is_finetuning = args.mode == "full" and args.init_checkpoint is not None
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    elif is_finetuning:
        learning_rate = 5e-5
    else:
        learning_rate = 3e-4

    if args.weight_decay is not None:
        weight_decay = args.weight_decay
    else:
        weight_decay = 0.0 if args.mode == "overfit" else 0.1

    if args.label_smoothing is not None:
        label_smoothing = args.label_smoothing
    else:
        label_smoothing = 0.0 if args.mode == "overfit" else (0.1 if args.mode == "full" else 0.05)

    if args.dropout is not None:
        dropout = args.dropout
    else:
        dropout = 0.0 if args.mode == "overfit" else (0.3 if args.mode == "full" else DROPOUT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Mode: {args.mode}; device: {device}; epochs: {epochs}; batch size: {batch_size}; "
        f"lr: {learning_rate:g}; weight decay: {weight_decay}; dropout: {dropout}; "
        f"label smoothing: {label_smoothing}"
    )

    # Overfit mode har ingen validation
    use_validation = args.mode != "overfit"

    tokenizer = Tokenizer()
    if args.retrain_tokenizer or tokenizer.vocab_size == 0:
        print(f"Training tokenizer (vocab_size={VOCAB_SIZE})...")
        _train_combined_tokenizer(tokenizer, seed=args.seed)

    if args.mode == "pretrain":
        # Pretraining på DailyDialog, som er meget større
        train_rows, validation_rows = load_daily_dialog_rows()
    else:
        train_rows, validation_rows = split_rows(load_conversation_rows(), seed=args.seed)

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    if args.mode == "overfit":
        # Kun ét svar pr. spørgsmål. Med flere forskellige svar på det samme
        # spørgsmål kan modellen ikke lære dem alle udenad.
        first_answers = {}
        for row in train_rows:
            first_answers.setdefault(row["Text"], row)
        train_data = list(first_answers.values())[:OVERFIT_EXAMPLES]
    else:
        train_data = train_rows
    print(train_data[0])
    train_dataset = DialogDataset(train_data, tokenizer, CONTEXT_LENGTH)
    validation_dataset = DialogDataset(validation_rows, tokenizer, CONTEXT_LENGTH)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, **loader_options
    )
    total_target_tokens = sum(
        len(tokens) - prompt_len for tokens, prompt_len in train_dataset.examples
    )
    print(
        f"Training rows: {len(train_data):,}; examples: {len(train_dataset):,}; "
        f"target tokens: {total_target_tokens:,}"
    )

    checkpoint = None
    model_cfg = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "tie_weights": True,
    }
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        saved_cfg = checkpoint.get("model_config", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if saved_cfg:
            model_cfg.update(saved_cfg)
            if "tie_weights" not in saved_cfg:
                model_cfg["tie_weights"] = "outputLayer.bias" not in state_dict
        if model_cfg["vocab_size"] != tokenizer.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size ({model_cfg['vocab_size']}) does not match "
                f"tokenizer vocab_size ({tokenizer.vocab_size}). Retrain from scratch "
                f"when using a new tokenizer."
            )

    model = MiniGPT(
        vocab_size=model_cfg["vocab_size"],
        context_length=model_cfg["context_length"],
        embedding_dim=model_cfg["embedding_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        dropout=dropout,
        tie_weights=model_cfg["tie_weights"],
    ).to(device)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from {args.init_checkpoint}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    train_criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        reduction="sum",
        label_smoothing=label_smoothing,
    )
    eval_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = None if args.mode == "overfit" else build_scheduler(optimizer, total_steps)

    best_selection_loss = float("inf")
    epochs_without_improvement = 0
    best_path = args.checkpoint_dir / f"miniGPT_{args.mode}_best.pth"
    final_path = args.checkpoint_dir / f"miniGPT_{args.mode}_final.pth"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = train_criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            token_count = (y != IGNORE_INDEX).sum().item()
            (loss / token_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                raw_loss = eval_criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            total_loss += raw_loss.item()
            total_tokens += token_count

        train_loss = total_loss / total_tokens
        _, train_accuracy = evaluate(model, train_loader, eval_criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        if use_validation:
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, eval_criterion, device
            )
            selection_loss = validation_loss
            print(
                f"Epoch {epoch + 1}/{epochs} (lr={current_lr:.2e}): train loss={train_loss:.4f}, "
                f"train accuracy={train_accuracy:.2%}, validation loss={validation_loss:.4f}, "
                f"validation accuracy={validation_accuracy:.2%}"
            )
        else:
            selection_loss = train_loss
            print(
                f"Epoch {epoch + 1}/{epochs} (lr={current_lr:.2e}): train loss={train_loss:.4f}, "
                f"train accuracy={train_accuracy:.2%}"
            )

        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            save_checkpoint(best_path, model, optimizer, epoch + 1, args, selection_loss)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping: stop når validation loss ikke længere bliver bedre
        if (
            use_validation
            and args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                f"Stopping early: validation loss has not improved for "
                f"{args.patience} epochs (best={best_selection_loss:.4f})"
            )
            break

    save_checkpoint(final_path, model, optimizer, epoch + 1, args, train_loss)
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved final checkpoint to {final_path}")

    if args.mode == "overfit":

        model.eval()

        with torch.no_grad():

            # Gennemgå alle spørgsmål
            for i, row in enumerate(train_data):

                # Brug spørgsmålet som prompt
                prompt = row["Text"] + "<SEP>"

                # Det svar modellen burde generere
                expected = row["Answer"]

                tokens = tokenizer.encode(prompt)

                x = torch.tensor(
                    [tokens],
                    dtype=torch.long,
                    device=device
                )

                generated_tokens = []

                # Generer svaret
                for _ in range(100):

                    logits = model(x[:, -CONTEXT_LENGTH:])

                    next_token = torch.argmax(
                        logits[:, -1, :],
                        dim=-1,
                        keepdim=True
                    )

                    x = torch.cat([x, next_token], dim=1)

                    generated_tokens.append(next_token.item())

                    generated = tokenizer.decode(generated_tokens)

                    # Stop ved næste replik eller samtalens afslutning
                    if "<SEP>" in generated or "<END>" in generated:
                        break

                generated = generated.split("<SEP>")[0].split("<END>")[0].strip()

                print(f"\nQuestion {i + 1}: {row['Text']}")
                print(f"Expected: {expected}")
                print(f"Generated: {generated}")
                print(f"Correct: {generated == expected.strip()}")


if __name__ == "__main__":
    main()