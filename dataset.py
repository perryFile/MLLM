import random
import re

from datasets import load_dataset
from tokenizer import Tokenizer
import torch
from torch.utils.data import Dataset


DATASET_NAME = "DSWF/conversation-dataset"
PRETRAIN_DATASET_NAME = "OpenRL/daily_dialog"
IGNORE_INDEX = -100

# Datasættet indeholder UTF-8 tegn, der er blevet læst som GBK,
# f.eks. "What鈥檚" i stedet for "What’s".
MOJIBAKE = re.compile(r"[\u2e80-\u9fff]+")


def _repair_match(match):
    try:
        return match.group().encode("gbk").decode("utf-8")
    except UnicodeError:
        return match.group()


def clean_text(text):
    text = MOJIBAKE.sub(_repair_match, text or "")
    # Datasættet blander ’ og ' i de samme ord
    return text.replace("’", "'").strip()


def load_conversation_rows():
    """Hent datasættet som en liste af rensede {"Text", "Answer"} rækker."""
    rows = []
    seen = set()

    for row in load_dataset(DATASET_NAME, split="train"):

        # En ekstra kolonne betyder, at rækken er en CSV-fejl i datasættet
        if row["Unnamed: 2"] is not None:
            continue

        question = clean_text(row["Text"])
        answer = clean_text(row["Answer"])

        # Spring tomme rækker og tekst, der ikke kunne repareres, over
        if not question or not answer or MOJIBAKE.search(question + answer):
            continue

        # Spring dubletter over
        if (question, answer) in seen:
            continue
        seen.add((question, answer))

        rows.append({"Text": question, "Answer": answer})

    return rows


def split_rows(rows, validation_fraction=0.1, seed=42):
    """Del rækkerne i train og validation.

    Datasættet har kun et train split. Vi deler på spørgsmål, så det
    samme spørgsmål ikke findes i både train og validation.
    """
    questions = sorted({row["Text"] for row in rows})
    random.Random(seed).shuffle(questions)

    validation_count = max(1, int(len(questions) * validation_fraction))
    validation_questions = set(questions[:validation_count])

    train_rows = [row for row in rows if row["Text"] not in validation_questions]
    validation_rows = [row for row in rows if row["Text"] in validation_questions]

    return train_rows, validation_rows


def clean_daily_dialog_text(text):
    """Få DailyDialog til at ligne DSWF-datasættet.

    DailyDialog skriver f.eks. "That ’ s bogus ." i stedet for "That's bogus."
    """
    text = text.replace("’", "'").replace("‘", "'")
    # "That ' s" -> "That's"
    text = re.sub(r"\s+'\s+(s|t|m|re|ve|ll|d)\b", r"'\1", text)
    # "bogus ." -> "bogus."
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _daily_dialog_pairs(split):
    rows = []

    for row in split:

        dialog = [clean_daily_dialog_text(utterance) for utterance in row["dialog"]]

        # Hver replik er spørgsmål til den næste replik
        for i in range(len(dialog) - 1):
            if dialog[i] and dialog[i + 1]:
                rows.append({"Text": dialog[i], "Answer": dialog[i + 1]})

    return rows


def load_daily_dialog_rows():
    """Hent DailyDialog til pretraining som {"Text", "Answer"} rækker."""
    dataset = load_dataset(PRETRAIN_DATASET_NAME)

    return (
        _daily_dialog_pairs(dataset["train"]),
        _daily_dialog_pairs(dataset["validation"]),
    )


class DialogDataset(Dataset):
    def __init__(self, data, tokenizer, context_length=128):

        self.context_length = context_length
        self.examples = []

        # Vi bruger END som padding, da den allerede
        # findes i dit vocabulary.
        # Padding-targets bliver ignoreret under træning.
        self.pad_id = tokenizer.encode("<END>")[0]

        for row in data:

            question = row["Text"]
            answer = row["Answer"]

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

    train_dataset, validation_dataset = split_rows(load_conversation_rows())

    datasetter = DialogDataset(train_dataset, tokenizer)

    x, y = datasetter[0]

    print("Number of train rows:", len(train_dataset))
    print("Number of validation rows:", len(validation_dataset))
    print("Number of training examples:", len(datasetter))

    print("\nInput:")
    print(tokenizer.decode(x.tolist()))

    print("\nTarget:")
    print(tokenizer.decode(y[y != IGNORE_INDEX].tolist()))

    print("\nShapes:")
    print("X:", x.shape)
    print("Y:", y.shape)