import unittest
from src.huffman_tree import HuffmanTree


class TestHuffmanTree(unittest.TestCase):
    def test_empty_word_frequency(self):
        tree = HuffmanTree([])

        self.assertEqual(tree.V_size, 0)
        self.assertEqual(tree.word_codes, [])
        self.assertEqual(tree.word_paths, [])
        self.assertEqual(tree.count, [])

    def test_one_word_gives_empty_code_and_path(self):
        tree = HuffmanTree([5])

        self.assertEqual(tree.V_size, 1)
        self.assertEqual(tree.word_codes, [[]])
        self.assertEqual(tree.word_paths, [[]])

    def test_sorted_word_ids_by_frequency_is_correct(self):
        tree = HuffmanTree([10, 3, 7])

        self.assertEqual(tree.sorted_word_ids_by_frequency, [1, 2, 0])

    def test_tree_builds_correct_number_of_nodes(self):
        tree = HuffmanTree([5, 7, 10, 15])

        self.assertEqual(len(tree.count), 2 * tree.V_size - 1)

    def test_root_count_is_sum_of_word_frequencies(self):
        tree = HuffmanTree([5, 7, 10, 15])

        self.assertEqual(tree.count[-1], 37)

    def test_two_words_get_correct_codes(self):
        tree = HuffmanTree([5, 7])

        self.assertEqual(tree.word_codes, [[0], [1]])

    def test_two_words_get_correct_paths(self):
        tree = HuffmanTree([5, 7])

        self.assertEqual(tree.word_paths, [[0], [0]])

    def test_more_frequent_word_gets_shorter_or_equal_code(self):
        tree = HuffmanTree([10, 1, 1])

        self.assertLessEqual(len(tree.word_codes[0]), len(tree.word_codes[1]))
        self.assertLessEqual(len(tree.word_codes[0]), len(tree.word_codes[2]))

    def test_each_word_code_and_path_have_same_length(self):
        tree = HuffmanTree([5, 7, 10, 15])

        for i in range(tree.V_size):
            self.assertEqual(len(tree.word_codes[i]), len(tree.word_paths[i]))

    def test_small_example_tree_builds_correct_codes_and_paths(self):
        tree = HuffmanTree([1, 2, 3, 4])

        self.assertEqual(tree.word_codes, [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0],
            [0]
        ])

        self.assertEqual(tree.word_paths, [
            [2, 1, 0],
            [2, 1, 0],
            [2, 1],
            [2]
        ])

if __name__ == "__main__":
    unittest.main()