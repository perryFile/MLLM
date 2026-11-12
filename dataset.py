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


def _daily_dialog_dialogs(split):
    """Byg multi-turn samtaler som lister af (user, bot) runder fra DailyDialog."""
    dialogs = []

    for row in split:
        utterances = [
            clean_daily_dialog_text(u) for u in row["dialog"] if clean_daily_dialog_text(u)
        ]
        if len(utterances) < 2:
            continue

        # Både speaker A->B (fra indeks 0) og B->A (fra indeks 1) giver naturlige
        # multi-turn samtaler, hvor modellen altid svarer på den seneste replik.
        for start in (0, 1):
            turns = []
            for i in range(start, len(utterances) - 1, 2):
                turns.append((utterances[i], utterances[i + 1]))
            if turns:
                dialogs.append({"Turns": turns})

    return dialogs


def load_daily_dialog_rows():
    """Hent DailyDialog til pretraining som multi-turn samtaler."""
    dataset = load_dataset(PRETRAIN_DATASET_NAME)

    return (
        _daily_dialog_dialogs(dataset["train"]),
        _daily_dialog_dialogs(dataset["validation"]),
    )


def format_row_for_tokenizer(row):
    if "Turns" in row:
        return "".join(q + "<SEP>" + a + "<END>" for q, a in row["Turns"])
    return row["Text"] + "<SEP>" + row["Answer"] + "<END>"


class DialogDataset(Dataset):
    def __init__(self, data, tokenizer, context_length=128):
        self.context_length = context_length
        self.examples = []
        self.total_target_chars = 0
        self.total_target_tokens = 0

        # Vi bruger END som padding, da den allerede findes i vocabulary.
        # Padding-targets bliver ignoreret under træning.
        self.pad_id = tokenizer.encode("<END>")[0]

        for row in data:
            turns = row["Turns"] if "Turns" in row else [(row["Text"], row["Answer"])]
            self._add_turns(turns, tokenizer)

        if not self.examples:
            raise ValueError("No valid training examples found.")

    def _commit_window(self, window_turns):
        tokens = []
        target_mask = []
        char_count = 0

        for prompt_tokens, answer_tokens, ans_chars in window_turns:
            tokens.extend(prompt_tokens)
            # Kun sidste token i prompten (<SEP>) skal forudsige første svartoken
            target_mask.extend([False] * (len(prompt_tokens) - 1) + [True])

            tokens.extend(answer_tokens)
            # Alle tokens i svaret (undtagen det sidste <END>, som ikke har et næste token i samme tur)
            # skal forudsige næste token; hvis der kommer en tur mere efter <END>, skal <END> IKKE
            # tvinges til at gætte brugerens næste spørgsmål.
            target_mask.extend([True] * (len(answer_tokens) - 1) + [False])
            char_count += ans_chars

        if len(tokens) < 2:
            return

        # x = tokens[:-1], y = tokens[1:], mask svarer til positioner i x/y
        mask = target_mask[:-1]
        active_tokens = sum(mask)
        if active_tokens == 0:
            return

        self.examples.append((tokens, mask, char_count))
        self.total_target_chars += char_count
        self.total_target_tokens += active_tokens

    def _add_turns(self, turns, tokenizer):
        encoded_turns = []
        for question, answer in turns:
            prompt_tokens = tokenizer.encode(question + "<SEP>")
            answer_tokens = tokenizer.encode(answer + "<END>")
            if len(prompt_tokens) + len(answer_tokens) > self.context_length + 1:
                continue
            encoded_turns.append((prompt_tokens, answer_tokens, len(answer) + 1))

        if not encoded_turns:
            return

        # Pak sammenhængende multi-turn vinduer op til context_length + 1 tokens
        start = 0
        while start < len(encoded_turns):
            window = []
            length = 0
            end = start
            while end < len(encoded_turns):
                turn_len = len(encoded_turns[end][0]) + len(encoded_turns[end][1])
                if length + turn_len > self.context_length + 1:
                    break
                window.append(encoded_turns[end])
                length += turn_len
                end += 1

            if window:
                self._commit_window(window)

            if end >= len(encoded_turns):
                break
            # Glid vinduet frem med overlap på sidste tur for kontekstkontinuitet
            start = max(start + 1, end - 1)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, mask, _ = self.examples[idx]

        x = tokens[:-1]
        y = [tok if keep else IGNORE_INDEX for tok, keep in zip(tokens[1:], mask)]

        padding = self.context_length - len(x)
        x = x + [self.pad_id] * padding
        y = y + [IGNORE_INDEX] * padding

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )



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