import argparse
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset import DialogDataset, IGNORE_INDEX
from models.model import MiniGPT
from tokenizer import Tokenizer


CONTEXT_LENGTH = 128
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniGPT on DailyDialog")
    parser.add_argument("--mode", choices=("overfit", "full"), default="overfit")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
                "context_length": CONTEXT_LENGTH,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
            },
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    epochs = args.epochs if args.epochs is not None else (100 if args.mode == "overfit" else 10)
    batch_size = args.batch_size if args.batch_size is not None else (1 if args.mode == "overfit" else 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {args.mode}; device: {device}; epochs: {epochs}; batch size: {batch_size}")

    tokenizer = Tokenizer()
    dataset = load_dataset("OpenRL/daily_dialog")
    if tokenizer.vocab_size == 0:
        tokenizer.train(
            "<SEP>".join(row["dialog"]) + "<END>" for row in dataset["train"]
        )

    train_data = dataset["train"].select(range(1)) if args.mode == "overfit" else dataset["train"]
    train_dataset = DialogDataset(train_data, tokenizer, CONTEXT_LENGTH)
    validation_dataset = DialogDataset(dataset["validation"], tokenizer, CONTEXT_LENGTH)
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
    print(
        f"Training rows: {len(train_data):,}; chunks: {len(train_dataset):,}; "
        f"tokens: {len(train_dataset.tokens):,}"
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_selection_loss = float("inf")
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
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            token_count = (y != IGNORE_INDEX).sum().item()
            (loss / token_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_tokens += token_count

        train_loss = total_loss / total_tokens
        _, train_accuracy = evaluate(model, train_loader, criterion, device)
        if args.mode == "full":
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, criterion, device
            )
            selection_loss = validation_loss
            print(
                f"Epoch {epoch + 1}/{epochs}: train loss={train_loss:.4f}, "
                f"train accuracy={train_accuracy:.2%}, validation loss={validation_loss:.4f}, "
                f"validation accuracy={validation_accuracy:.2%}"
            )
        else:
            selection_loss = train_loss
            print(
                f"Epoch {epoch + 1}/{epochs}: train loss={train_loss:.4f}, "
                f"train accuracy={train_accuracy:.2%}"
            )

        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            save_checkpoint(best_path, model, optimizer, epoch + 1, args, selection_loss)

    save_checkpoint(final_path, model, optimizer, epochs, args, train_loss)
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()