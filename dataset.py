from datasets import load_dataset
from tokenizer import Tokenizer
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


class DialogDataset(Dataset):
    def __init__(self, data, tokenizer, context_length=128):

        self.context_length = context_length
        self.examples = []

        # Vi bruger END som padding, da den allerede
        # findes i dit vocabulary.
        # Padding-targets bliver ignoreret under træning.
        self.pad_id = tokenizer.encode("<END>")[0]

        for row in data:

            dialog = row["dialog"]

            for i in range(len(dialog) - 1):

                question = dialog[i]
                answer = dialog[i + 1]

                # Tokenize prompt og svar
                prompt_tokens = tokenizer.encode(
                    question + "<SEP>"
                )

                answer_tokens = tokenizer.encode(
                    answer + "<END>"
                )

                tokens = prompt_tokens + answer_tokens

                # Spring eksempler over, der er for lange
                if len(tokens) > context_length + 1:
                    continue

                self.examples.append(
                    (tokens, len(prompt_tokens))
                )

        if not self.examples:
            raise ValueError("No valid training examples found.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        tokens, prompt_length = self.examples[idx]

        # Next-token prediction
        x = tokens[:-1]
        y = tokens[1:]

        # Ignorer loss på spørgsmålet,
        # men behold forudsigelsen af første svartoken
        for i in range(prompt_length - 1):
            y[i] = IGNORE_INDEX

        # Padding
        padding = self.context_length - len(x)

        x = x + [self.pad_id] * padding
        y = y + [IGNORE_INDEX] * padding

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y



if __name__ == "__main__":

    tokenizer = Tokenizer()

    train_dataset = load_dataset(
        "OpenRL/daily_dialog",
        split="train"
    )

    datasetter = DialogDataset(train_dataset, tokenizer)

    x, y = datasetter[0]

    print("Number of dialogs:", len(train_dataset))
    print("Number of training examples:", len(datasetter))

    print("\nInput:")
    print(tokenizer.decode(x.tolist()))

    print("\nTarget:")
    print(tokenizer.decode(y[y != IGNORE_INDEX].tolist()))

    print("\nShapes:")
    print("X:", x.shape)
    print("Y:", y.shape)