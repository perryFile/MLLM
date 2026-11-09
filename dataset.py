from datasets import load_dataset
from tokenizer import Tokenizer
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100

class DialogDataset(Dataset):
    def __init__(self, data, tokenizer, context_length=128):
        self.context_length = context_length

        # Tokeniser alle samtaler én gang
        tokens = []

        for row in data:
            text = self.format_dialog(row["dialog"])
            tokens.extend(tokenizer.encode(text))

        self.tokens = torch.tensor(tokens, dtype=torch.long)

        end_token = tokenizer.encode("<END>")
        if len(end_token) != 1:
            raise ValueError("<END> must encode to exactly one token")
        self.padding_token_id = end_token[0]

    def format_dialog(self, dialog):
        return "<SEP>".join(dialog) + "<END>"

    def __len__(self):
        number_of_targets = max(0, len(self.tokens) - 1)
        return (number_of_targets + self.context_length - 1) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length + 1

        chunk = self.tokens[start:end]

        x = chunk[:-1]
        y = chunk[1:]

        padding_length = self.context_length - len(x)
        if padding_length > 0:
            x = torch.cat((
                x,
                torch.full((padding_length,), self.padding_token_id, dtype=torch.long),
            ))
            y = torch.cat((
                y,
                torch.full((padding_length,), IGNORE_INDEX, dtype=torch.long),
            ))

        return x, y



if __name__ == "__main__":
    tokenizer = Tokenizer()
    train_dataset = load_dataset("OpenRL/daily_dialog", split="train")
    datasetter = DialogDataset(train_dataset, tokenizer)
    print(len(train_dataset))
    print(tokenizer.decode(datasetter.__getitem__(0)[0]))
    targets = datasetter.__getitem__(0)[1]
    print(tokenizer.decode(targets[targets != IGNORE_INDEX].tolist()))