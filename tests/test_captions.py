"""Caption preprocessing: cleaning rules, filtering and the vocabulary (no dataset needed)."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.data.captions import (MAX_LENGTH, MAX_WORDS, Vocabulary, clean_caption, number_to_words,
                               preprocess_description, split_captions, unknown_rate)


class CleanCaptionTest(unittest.TestCase):
    def test_lowercase_and_punctuation_become_spaces(self):
        self.assertEqual(clean_caption("This bird is Reddish-Brown, with a short/stubby bill."),
                         "this bird is reddish brown with a short stubby bill")

    def test_corrupted_character_is_a_space_not_deleted(self):
        self.assertEqual(clean_caption("this��bird��has"), "this bird has")

    def test_apostrophes_are_deleted(self):
        self.assertEqual(clean_caption("the bird's beak, it's `s tail"), "the birds beak its s tail")

    def test_numbers_become_words(self):
        self.assertEqual(clean_caption("2white wing bars, 3-5 inches"), "two white wing bars three five inches")
        self.assertEqual(clean_caption("about 2.5 times"), "about two point five times")
        self.assertEqual(clean_caption("a 45 degree angle"), "a forty five degree angle")
        self.assertEqual(clean_caption("id 81043892 end"), "id end")

    def test_number_to_words(self):
        self.assertEqual([number_to_words(n) for n in (0, 13, 20, 99, 100)],
                         ["zero", "thirteen", "twenty", "ninety nine", ""])

    def test_one_spelling_per_word(self):
        self.assertEqual(clean_caption("gray grayish coloured grays graying"), "grey greyish colored greys greying")


class PreprocessDescriptionTest(unittest.TestCase):
    long_caption = " ".join(["word"] * (MAX_WORDS + 1))
    description = f"A small bird.\nthis\n\na small bird\n{long_caption}\nthe bird has a red crown.\n"

    def test_split_drops_empty_lines(self):
        self.assertEqual(len(split_captions(self.description)), 5)

    def test_training_filters(self):
        stats = Counter()
        captions = preprocess_description(self.description, train=True, stats=stats)
        self.assertEqual(captions, [["a", "small", "bird"], ["the", "bird", "has", "a", "red", "crown"]])
        self.assertEqual((stats["removed_too_short"], stats["removed_too_long"], stats["removed_duplicate"]),
                         (1, 1, 1))

    def test_references_keep_every_caption(self):
        self.assertEqual(len(preprocess_description(self.description, train=False)), 5)


class VocabularyTest(unittest.TestCase):
    def setUp(self):
        captions = [["a", "red", "bird"]] * 5 + [["a", "blue", "bird"]] * 2
        self.vocab = Vocabulary.build(captions, min_freq=5)

    def test_rare_words_are_left_out(self):
        self.assertEqual(self.vocab.itos, ["<pad>", "<start>", "<end>", "<unk>", "a", "bird", "red"])

    def test_encode_decode_round_trip(self):
        ids = self.vocab.encode(["a", "blue", "bird"])
        self.assertEqual(len(ids), MAX_LENGTH)
        self.assertEqual(ids[:5], [1, 4, 3, 5, 2])
        self.assertEqual(self.vocab.decode(ids), "a <unk> bird")

    def test_long_captions_are_cut_but_keep_end(self):
        ids = self.vocab.encode(["a"] * 50, max_length=8)
        self.assertEqual(ids, [1, 4, 4, 4, 4, 4, 4, 2])

    def test_unknown_rate(self):
        self.assertAlmostEqual(unknown_rate([["a", "blue", "bird", "green"]], self.vocab), 0.5)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            self.vocab.save(path, min_freq=5)
            self.assertEqual(Vocabulary.load(path).itos, self.vocab.itos)


if __name__ == "__main__":
    unittest.main()
