import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONTEXT_LENGTH, DROPOUT, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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

    # Overfit mode har ingen validation
    use_validation = args.mode != "overfit"

    tokenizer = Tokenizer()
    if args.mode == "pretrain":
        # Pretraining på DailyDialog, som er meget større
        train_rows, validation_rows = load_daily_dialog_rows()
    else:
        train_rows, validation_rows = split_rows(load_conversation_rows(), seed=args.seed)
    if tokenizer.vocab_size == 0:
        tokenizer.train(
            row["Text"] + "<SEP>" + row["Answer"] + "<END>" for row in train_rows
        )

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
    print(
        f"Training rows: {len(train_data):,}; chunks: {len(train_dataset):,}; "
        f"tokens: {len(train_dataset):,}"
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        # Ingen dropout i overfit mode, hvor modellen skal lære data udenad
        dropout=0.0 if args.mode == "overfit" else DROPOUT,
    ).to(device)

    if args.init_checkpoint is not None:
        # Fortsæt fra en model, der allerede er trænet (f.eks. pretraining)
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from {args.init_checkpoint}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
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
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            token_count = (y != IGNORE_INDEX).sum().item()
            (loss / token_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_tokens += token_count

        train_loss = total_loss / total_tokens
        _, train_accuracy = evaluate(model, train_loader, criterion, device)
        if use_validation:
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