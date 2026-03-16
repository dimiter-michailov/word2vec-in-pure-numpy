import os
import tempfile
import unittest
from pathlib import Path

from src.text_processing import TextProcessing


class TestTextProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessing.__new__(TextProcessing)
        self.processor.word_frequency = []

    def test_tokenize_lowercases_and_removes_punctuation(self):
        text = "Hello, WORLD!!!"
        tokens = self.processor.tokenize(text)

        self.assertEqual(tokens, ["hello", "world"])

    def test_tokenize_removes_numbers_and_symbols(self):
        text = "Apple 123 banana!!! @orange"
        tokens = self.processor.tokenize(text)

        self.assertEqual(tokens, ["apple", "banana", "orange"])

    def test_tokenize_handles_extra_spaces_and_newlines(self):
        text = "One   two\nthree\tfour"
        tokens = self.processor.tokenize(text)

        self.assertEqual(tokens, ["one", "two", "three", "four"])


class TestTextProcessingWithFiles(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)

        Path("datasets").mkdir()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

        token_ids_file = Path("token_ids.dat")
        if token_ids_file.exists():
            token_ids_file.unlink()

    def write_dataset_file(self, file_name, content):
        path = Path("datasets") / file_name
        with open(path, "w") as file:
            file.write(content)

    def test_build_vocab_creates_correct_vocab_and_vocab_size(self):
        self.write_dataset_file("tiny.txt", "apple banana apple orange")

        processor = TextProcessing("tiny.txt")

        self.assertEqual(processor.vocab, {
            "apple": 0,
            "banana": 1,
            "orange": 2
        })
        self.assertEqual(processor.V_size, 3)

    def test_build_vocab_counts_word_frequency_correctly(self):
        self.write_dataset_file("tiny.txt", "apple banana apple orange banana apple")

        processor = TextProcessing("tiny.txt")

        self.assertEqual(processor.word_frequency, [3, 2, 1])

    def test_encode_token_ids_returns_correct_ids(self):
        self.write_dataset_file("tiny.txt", "apple banana apple orange")

        processor = TextProcessing("tiny.txt")

        self.assertEqual(list(processor.token_ids), [0, 1, 0, 2])

    def test_build_vocab_and_encode_token_ids_work_across_multiple_lines(self):
        self.write_dataset_file("tiny.txt", "apple banana\napple orange")

        processor = TextProcessing("tiny.txt")

        self.assertEqual(processor.vocab, {
            "apple": 0,
            "banana": 1,
            "orange": 2
        })
        self.assertEqual(processor.word_frequency, [2, 1, 1])
        self.assertEqual(list(processor.token_ids), [0, 1, 0, 2])


if __name__ == "__main__":
    unittest.main()