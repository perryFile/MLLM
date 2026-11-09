from pathlib import Path

from tokenizers import Tokenizer as BpeTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class Tokenizer:
    def __init__(self, tokenizer_file="tokenizer.json"):
        self.tokenizer_file = Path(tokenizer_file)
        self._tokenizer = None

        if self.tokenizer_file.exists():
            self.load()

    def train(self, texts, vocab_size=2000):
        """Train a BPE vocabulary from an iterable of strings."""
        self._tokenizer = BpeTokenizer(BPE(unk_token="<UNK>"))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<UNK>", "<SEP>", "<END>"],
        )
        self._tokenizer.train_from_iterator(texts, trainer=trainer)
        self.save()

    def encode(self, text):
        """Convert text into BPE token IDs."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer has not been initialized.")
        return self._tokenizer.encode(text).ids

    def decode(self, tokens):
        """Convert BPE token IDs back into text."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer has not been initialized.")
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    def save(self):
        """Save the BPE vocabulary to JSON."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer has not been initialized.")
        self.tokenizer_file.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(self.tokenizer_file))

    def load(self):
        """Load the saved BPE vocabulary."""
        self._tokenizer = BpeTokenizer.from_file(str(self.tokenizer_file))

    @property
    def vocab_size(self):
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()
