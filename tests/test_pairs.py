import unittest
from src.cbow.cbow import CBOW
from src.skipgram.skipgram import Skipgram


class TestSkipgramPairs(unittest.TestCase):
    def setUp(self):
        self.skipgram = Skipgram.__new__(Skipgram)

    def test_make_skipgram_training_pairs_small_example(self):
        token_ids = [10, 20, 30, 40, 50]

        pairs = self.skipgram.make_skipgram_training_pairs(token_ids, 4)

        self.assertEqual(pairs, [
            (10, [20, 30]),
            (20, [10, 30, 40]),
            (30, [10, 20, 40, 50]),
            (40, [20, 30, 50]),
            (50, [30, 40])
        ])

    def test_make_skipgram_training_pairs_keeps_context_in_left_then_right_order(self):
        token_ids = [1, 2, 3, 4, 5]

        pairs = self.skipgram.make_skipgram_training_pairs(token_ids, 2)

        self.assertEqual(pairs, [
            (1, [2]),
            (2, [1, 3]),
            (3, [2, 4]),
            (4, [3, 5]),
            (5, [4])
        ])

    def test_make_skipgram_training_pairs_empty_token_ids(self):
        token_ids = []

        pairs = self.skipgram.make_skipgram_training_pairs(token_ids, 4)

        self.assertEqual(pairs, [])

    def test_make_skipgram_training_pairs_one_token_gives_no_pairs(self):
        token_ids = [7]

        pairs = self.skipgram.make_skipgram_training_pairs(token_ids, 4)

        self.assertEqual(pairs, [])

    def test_make_skipgram_training_pairs_zero_context_size_gives_no_pairs(self):
        token_ids = [1, 2, 3]

        pairs = self.skipgram.make_skipgram_training_pairs(token_ids, 0)

        self.assertEqual(pairs, [])

    def test_make_skipgram_training_pairs_odd_context_size_uses_floor_division(self):
        token_ids = [1, 2, 3, 4, 5]

        pairs = self.skipgram.make_skipgram_training_pairs(token_ids, 3)

        self.assertEqual(pairs, [
            (1, [2]),
            (2, [1, 3]),
            (3, [2, 4]),
            (4, [3, 5]),
            (5, [4])
        ])


class TestCBOWPairs(unittest.TestCase):
    def setUp(self):
        self.cbow = CBOW.__new__(CBOW)

    def test_make_cbow_training_pair_first_word(self):
        self.cbow.token_ids = [10, 20, 30, 40, 50]
        self.cbow.token_count = len(self.cbow.token_ids)
        self.cbow.context_size = 4

        pair = self.cbow.make_cbow_training_pair(0)

        self.assertEqual(pair, ([20, 30], 10))

    def test_make_cbow_training_pair_middle_word(self):
        self.cbow.token_ids = [10, 20, 30, 40, 50]
        self.cbow.token_count = len(self.cbow.token_ids)
        self.cbow.context_size = 4

        pair = self.cbow.make_cbow_training_pair(2)

        self.assertEqual(pair, ([10, 20, 40, 50], 30))

    def test_make_cbow_training_pair_last_word(self):
        self.cbow.token_ids = [10, 20, 30, 40, 50]
        self.cbow.token_count = len(self.cbow.token_ids)
        self.cbow.context_size = 4

        pair = self.cbow.make_cbow_training_pair(4)

        self.assertEqual(pair, ([30, 40], 50))

    def test_make_cbow_training_pair_keeps_context_in_left_then_right_order(self):
        self.cbow.token_ids = [1, 2, 3, 4, 5]
        self.cbow.token_count = len(self.cbow.token_ids)
        self.cbow.context_size = 2

        pair = self.cbow.make_cbow_training_pair(2)

        self.assertEqual(pair, ([2, 4], 3))

    def test_make_cbow_training_pair_one_token_gives_none(self):
        self.cbow.token_ids = [7]
        self.cbow.token_count = len(self.cbow.token_ids)
        self.cbow.context_size = 4

        pair = self.cbow.make_cbow_training_pair(0)

        self.assertEqual(pair, (None, None))

    def test_make_cbow_training_pair_zero_context_size_gives_none(self):
        self.cbow.token_ids = [1, 2, 3]
        self.cbow.token_count = len(self.cbow.token_ids)
        self.cbow.context_size = 0

        pair = self.cbow.make_cbow_training_pair(1)

        self.assertEqual(pair, (None, None))

    def test_make_cbow_training_pair_odd_context_size_uses_floor_division(self):
        self.cbow.token_ids = [1, 2, 3, 4, 5]
        self.cbow.token_count = len(self.cbow.token_ids)
        self.cbow.context_size = 3

        pair = self.cbow.make_cbow_training_pair(2)

        self.assertEqual(pair, ([2, 4], 3))