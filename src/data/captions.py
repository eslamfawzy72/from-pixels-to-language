"""Caption preprocessing: split, clean, filter and encode the CUB `description` field.

Every rule below comes from the caption EDA (last section of notebooks/eda.ipynb).

    from src.data.captions import Vocabulary, preprocess_description

    vocab = Vocabulary.load()                                  # built from training captions only
    captions = preprocess_description(record["description"])   # list of token lists
    ids = vocab.encode(captions[0])                            # <start> ... <end> <pad> ..., length 32
    vocab.decode(ids)                                          # "this bird is brown with a lighter brown crest"

Build (or rebuild) the vocabulary file and print the cleaning statistics for the report:

    python -m src.data.captions
"""
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MIN_WORDS = 3            # shorter training captions are dropped ("this")
MAX_WORDS = 30           # longer training captions are dropped (about 1.4%); 98.6% fit
MAX_LENGTH = MAX_WORDS + 2  # sequence length with <start> and <end>
MIN_FREQ = 5             # rarer words (mostly typos) become <unk>
VOCAB_PATH = PROJECT_ROOT / "src" / "data" / f"vocab_min{MIN_FREQ}.json"

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIAL_TOKENS = [PAD, START, END, UNK]  # ids 0, 1, 2, 3

# One spelling per word: the more frequent one in the training captions
SPELLING = {
    "gray": "grey", "grays": "greys", "grayish": "greyish", "graying": "greying",
    "coloured": "colored",
}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n):
    """0-99 as words ("45" -> "forty five"). Larger numbers return "" (the only one in the data is junk)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return ""


def _decimal_to_words(match):
    whole, fraction = match.group(1), match.group(2)
    return f" {number_to_words(int(whole))} point {' '.join(_ONES[int(d)] for d in fraction)} "


def clean_caption(text):
    """Return one caption as a clean string: lowercase letters and single spaces only.

    Order matters:
    1. lowercase
    2. corrupted character U+FFFD -> space (it stands for a space: "this��bird")
    3. delete apostrophes, so "bird's" -> "birds" instead of "bird s"
    4. numbers -> words: "2.5" -> "two point five", "2white" -> "two white"
    5. every other non-letter -> space, so "reddish-brown" -> "reddish brown"
    6. one spelling per word ("gray" -> "grey")
    """
    text = text.lower().replace("�", " ")
    text = re.sub(r"['`‘’]", "", text)
    text = re.sub(r"(\d+)\.(\d+)", _decimal_to_words, text)
    text = re.sub(r"\d+", lambda m: f" {number_to_words(int(m.group()))} ", text)
    words = re.sub(r"[^a-z]+", " ", text).split()
    return " ".join(SPELLING.get(word, word) for word in words)


def tokenize(text):
    """Clean a caption and split it into words."""
    return clean_caption(text).split()


def split_captions(description):
    """Split the description field (ten captions joined by newlines) into non-empty raw captions."""
    return [caption.strip() for caption in description.split("\n") if caption.strip()]


def preprocess_description(description, train=True, stats=None):
    """Return the cleaned captions of one image as a list of token lists.

    train=True (captions the model learns from): also drops captions with fewer than MIN_WORDS
    or more than MAX_WORDS words, and exact duplicates within the same image.
    train=False (reference captions for validation / test metrics): keeps every non-empty caption,
    so references are never changed by training choices.

    `stats`, if given, is a Counter that receives how many captions each step removed.
    """
    stats = Counter() if stats is None else stats
    kept, seen = [], set()
    for raw in split_captions(description):
        stats["raw"] += 1
        tokens = tokenize(raw)
        if not tokens:
            stats["removed_empty"] += 1
            continue
        if train:
            if len(tokens) < MIN_WORDS:
                stats["removed_too_short"] += 1
                continue
            if len(tokens) > MAX_WORDS:
                stats["removed_too_long"] += 1
                continue
            key = " ".join(tokens)
            if key in seen:
                stats["removed_duplicate"] += 1
                continue
            seen.add(key)
        kept.append(tokens)
        stats["kept"] += 1
    return kept


class Vocabulary:
    """Word <-> id mapping. Ids 0-3 are <pad>, <start>, <end>, <unk>; the rest are sorted by frequency."""

    def __init__(self, words):
        self.itos = SPECIAL_TOKENS + [w for w in words if w not in SPECIAL_TOKENS]
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.pad_id, self.start_id, self.end_id, self.unk_id = (self.stoi[t] for t in SPECIAL_TOKENS)

    @classmethod
    def build(cls, token_lists, min_freq=MIN_FREQ):
        """Keep words that appear at least `min_freq` times. Use training captions only."""
        counts = Counter(word for tokens in token_lists for word in tokens)
        words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls(words)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_length=MAX_LENGTH, pad=True):
        """Token list -> ids: <start> words <end>, cut to `max_length` and padded with <pad>."""
        ids = [self.start_id] + [self.stoi.get(w, self.unk_id) for w in tokens][:max_length - 2] + [self.end_id]
        if pad:
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids):
        """Ids -> sentence. Stops at <end>; skips <start> and <pad>."""
        words = []
        for i in ids:
            i = int(i)
            if i == self.end_id:
                break
            if i not in (self.start_id, self.pad_id):
                words.append(self.itos[i])
        return " ".join(words)

    def save(self, path=VOCAB_PATH, **info):
        Path(path).write_text(json.dumps({**info, "itos": self.itos}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path=VOCAB_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Build it with: python -m src.data.captions")
        return cls(json.loads(path.read_text(encoding="utf-8"))["itos"])


def unknown_rate(token_lists, vocab):
    """Fraction of words that would become <unk>."""
    total = sum(len(tokens) for tokens in token_lists)
    unknown = sum(w not in vocab.stoi for tokens in token_lists for w in tokens)
    return unknown / total if total else 0.0


def main():
    """Build the vocabulary from the training split and print the cleaning statistics."""
    from src.data.data_loader import load_cub_dataset
    from src.data.splits import load_split

    train_dataset, test_dataset = load_cub_dataset()
    split = load_split()
    train_descriptions = train_dataset["description"]  # reads only the text column (fast)

    def captions_of(descriptions, indices, train):
        stats = Counter()
        captions = [c for i in indices for c in preprocess_description(descriptions[i], train, stats)]
        return captions, stats

    train_caps, train_stats = captions_of(train_descriptions, split["train"], train=True)
    val_caps, _ = captions_of(train_descriptions, split["val"], train=False)
    test_caps, _ = captions_of(test_dataset["description"], range(len(test_dataset)), train=False)

    vocab = Vocabulary.build(train_caps)
    counts = Counter(w for tokens in train_caps for w in tokens)
    coverage = sum(counts[w] for w in vocab.itos if w in counts) / sum(counts.values())
    vocab.save(min_freq=MIN_FREQ, max_words=MAX_WORDS, max_length=MAX_LENGTH,
               built_from=f"training split ({len(split['train'])} images, {len(train_caps)} captions)")

    print("Training captions (split['train'] images):")
    for key in ("raw", "removed_empty", "removed_too_short", "removed_too_long", "removed_duplicate", "kept"):
        print(f"  {key:<20} {train_stats[key]:>7,}")
    print(f"Validation reference captions: {len(val_caps):,}   Test reference captions: {len(test_caps):,}")
    print(f"Unique training words: {len(counts):,}")
    print(f"Vocabulary (min_freq={MIN_FREQ}): {len(vocab):,} tokens incl. 4 special, "
          f"covering {coverage:.2%} of training words")
    print(f"<unk> rate: validation {unknown_rate(val_caps, vocab):.2%}, test {unknown_rate(test_caps, vocab):.2%}")
    print(f"Saved {VOCAB_PATH}")


if __name__ == "__main__":
    main()
