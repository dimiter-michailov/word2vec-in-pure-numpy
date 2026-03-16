import unittest
from unittest.mock import patch
import numpy as np

from src.cbow.negative_cbow import NegativeCBOW
from src.skipgram.negative_skipgram import NegativeSkipgram


class TestNegativeCBOW(unittest.TestCase):
    def make_negative_cbow(self):
        model = NegativeCBOW.__new__(NegativeCBOW)

        model.V_size = 4
        model.embedding_dim = 2
        model.l_rate = 0.1
        model.negative_sample_count = 1

        model.input_hidden_matrix = np.array([
            [1.0, 0.0],   # word 0
            [0.0, 1.0],   # word 1
            [1.0, 1.0],   # word 2
            [2.0, 0.0]    # word 3
        ], dtype=np.float32)

        model.hidden_output_matrix = np.array([
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, -1.0]
        ], dtype=np.float32)

        model.noise_table_size = 5
        model.noise_table = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        return model

    def test_negative_sampling_does_not_return_context_or_target(self):
        model = self.make_negative_cbow()
        model.negative_sample_count = 2

        with patch("numpy.random.randint", side_effect=[
            np.array([0, 1]),
            3,
            4
        ]):
            negative_samples = model.negative_sampling(context=[0], target=1)

        self.assertEqual(list(negative_samples), [3, 4])

    def test_loss_function_matches_manual_value(self):
        model = self.make_negative_cbow()

        positive_score = 2.0
        negative_scores = np.array([0.5, -1.0], dtype=np.float32)

        positive_prob = 1.0 / (1.0 + np.exp(-positive_score))
        negative_prob = 1.0 / (1.0 + np.exp(negative_scores))

        expected_loss = -np.log(positive_prob + 1e-10)
        expected_loss += -np.sum(np.log(negative_prob + 1e-10))

        loss = model.loss_function(positive_score, negative_scores)

        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_feedforward_returns_correct_hidden_scores_and_loss(self):
        model = self.make_negative_cbow()

        model.negative_sampling = lambda context, target: np.array([2], dtype=np.int32)

        context = [0, 3]
        target = 1

        hidden, target_out, negative_samples, positive_score, negative_scores, loss = model.feedforward(context, target)

        expected_hidden = np.array([1.5, 0.0], dtype=np.float32)
        expected_positive_score = 3.0
        expected_negative_scores = np.array([0.0], dtype=np.float32)

        positive_prob = 1.0 / (1.0 + np.exp(-expected_positive_score))
        negative_prob = 1.0 / (1.0 + np.exp(-expected_negative_scores))
        expected_loss = -np.log(positive_prob + 1e-10)
        expected_loss += -np.sum(np.log(negative_prob + 1e-10))

        self.assertTrue(np.allclose(hidden, expected_hidden))
        self.assertEqual(target_out, target)
        self.assertEqual(list(negative_samples), [2])
        self.assertAlmostEqual(positive_score, expected_positive_score, places=6)
        self.assertTrue(np.allclose(negative_scores, expected_negative_scores))
        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_backpropagate_updates_only_context_rows_target_column_and_negative_column(self):
        model = self.make_negative_cbow()

        context = [0, 3]
        target = 1
        negative_samples = np.array([2], dtype=np.int32)

        hidden = np.array([1.5, 0.0], dtype=np.float32)
        positive_score = 3.0
        negative_scores = np.array([0.0], dtype=np.float32)

        old_input_hidden_matrix = model.input_hidden_matrix.copy()
        old_hidden_output_matrix = model.hidden_output_matrix.copy()

        model.backpropagate(hidden, target, negative_samples, positive_score, negative_scores, context)

        self.assertTrue(np.allclose(model.input_hidden_matrix[1], old_input_hidden_matrix[1]))
        self.assertTrue(np.allclose(model.input_hidden_matrix[2], old_input_hidden_matrix[2]))
        self.assertFalse(np.allclose(model.input_hidden_matrix[0], old_input_hidden_matrix[0]))
        self.assertFalse(np.allclose(model.input_hidden_matrix[3], old_input_hidden_matrix[3]))

        update_word_0 = model.input_hidden_matrix[0] - old_input_hidden_matrix[0]
        update_word_3 = model.input_hidden_matrix[3] - old_input_hidden_matrix[3]
        self.assertTrue(np.allclose(update_word_0, update_word_3))

        self.assertTrue(np.allclose(model.hidden_output_matrix[:, 0], old_hidden_output_matrix[:, 0]))
        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 1], old_hidden_output_matrix[:, 1]))
        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 2], old_hidden_output_matrix[:, 2]))
        self.assertTrue(np.allclose(model.hidden_output_matrix[:, 3], old_hidden_output_matrix[:, 3]))


class TestNegativeSkipgram(unittest.TestCase):
    def make_negative_skipgram(self):
        model = NegativeSkipgram.__new__(NegativeSkipgram)

        model.V_size = 4
        model.embedding_dim = 2
        model.l_rate = 0.1
        model.negative_sample_count_per_target = 1

        model.input_hidden_matrix = np.array([
            [1.0, 0.0],   # word 0
            [0.0, 1.0],   # word 1
            [1.0, 1.0],   # word 2
            [2.0, 0.0]    # word 3
        ], dtype=np.float32)

        model.hidden_output_matrix = np.array([
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, -1.0]
        ], dtype=np.float32)

        model.noise_table_size = 4
        model.noise_table = np.array([0, 1, 2, 3], dtype=np.int32)

        return model

    def test_negative_sampling_does_not_return_center_or_context(self):
        model = self.make_negative_skipgram()

        with patch("numpy.random.randint", side_effect=[
            np.array([1, 2]),
            3,
            3
        ]):
            negative_samples = model.negative_sampling(center=1, context=[0, 2])

        self.assertEqual(list(negative_samples), [3, 3])

    def test_loss_function_matches_manual_value(self):
        model = self.make_negative_skipgram()

        positive_score = 1.0
        negative_scores = np.array([-1.0], dtype=np.float32)

        positive_prob = 1.0 / (1.0 + np.exp(-positive_score))
        negative_prob = 1.0 / (1.0 + np.exp(negative_scores))

        expected_loss = -np.log(positive_prob + 1e-10)
        expected_loss += -np.sum(np.log(negative_prob + 1e-10))

        loss = model.loss_function(positive_score, negative_scores)

        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_feedforward_returns_correct_hidden_sampled_data_and_loss(self):
        model = self.make_negative_skipgram()

        model.negative_sampling = lambda center, context: np.array([3, 3], dtype=np.int32)

        center = 1
        context = [0, 2]

        center_out, hidden, sampled_data, loss = model.feedforward(center, context)

        expected_hidden = np.array([0.0, 1.0], dtype=np.float32)

        score_word_0 = 0.0
        negative_score_word_0 = np.array([-1.0], dtype=np.float32)
        positive_prob_0 = 1.0 / (1.0 + np.exp(-score_word_0))
        negative_prob_0 = 1.0 / (1.0 + np.exp(negative_score_word_0))
        loss_word_0 = -np.log(positive_prob_0 + 1e-10) - np.sum(np.log(negative_prob_0 + 1e-10))

        score_word_2 = 1.0
        negative_score_word_2 = np.array([-1.0], dtype=np.float32)
        positive_prob_2 = 1.0 / (1.0 + np.exp(-score_word_2))
        negative_prob_2 = 1.0 / (1.0 + np.exp(negative_score_word_2))
        loss_word_2 = -np.log(positive_prob_2 + 1e-10) - np.sum(np.log(negative_prob_2 + 1e-10))

        expected_loss = loss_word_0 + loss_word_2

        self.assertEqual(center_out, center)
        self.assertTrue(np.allclose(hidden, expected_hidden))
        self.assertEqual(len(sampled_data), 2)

        self.assertEqual(sampled_data[0][0], 0)
        self.assertEqual(list(sampled_data[0][1]), [3])
        self.assertAlmostEqual(sampled_data[0][2], 0.0, places=6)
        self.assertTrue(np.allclose(sampled_data[0][3], np.array([-1.0], dtype=np.float32)))

        self.assertEqual(sampled_data[1][0], 2)
        self.assertEqual(list(sampled_data[1][1]), [3])
        self.assertAlmostEqual(sampled_data[1][2], 1.0, places=6)
        self.assertTrue(np.allclose(sampled_data[1][3], np.array([-1.0], dtype=np.float32)))

        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_backpropagate_updates_only_center_row_and_used_output_columns(self):
        model = self.make_negative_skipgram()

        center = 1
        hidden = np.array([0.0, 1.0], dtype=np.float32)

        sampled_data = [
            (0, np.array([3], dtype=np.int32), 0.0, np.array([-1.0], dtype=np.float32)),
            (2, np.array([3], dtype=np.int32), 1.0, np.array([-1.0], dtype=np.float32))
        ]

        old_input_hidden_matrix = model.input_hidden_matrix.copy()
        old_hidden_output_matrix = model.hidden_output_matrix.copy()

        model.backpropagate(sampled_data, hidden, center)

        self.assertTrue(np.allclose(model.input_hidden_matrix[0], old_input_hidden_matrix[0]))
        self.assertFalse(np.allclose(model.input_hidden_matrix[1], old_input_hidden_matrix[1]))
        self.assertTrue(np.allclose(model.input_hidden_matrix[2], old_input_hidden_matrix[2]))
        self.assertTrue(np.allclose(model.input_hidden_matrix[3], old_input_hidden_matrix[3]))

        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 0], old_hidden_output_matrix[:, 0]))
        self.assertTrue(np.allclose(model.hidden_output_matrix[:, 1], old_hidden_output_matrix[:, 1]))
        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 2], old_hidden_output_matrix[:, 2]))
        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 3], old_hidden_output_matrix[:, 3]))


if __name__ == "__main__":
    unittest.main()