import unittest
import numpy as np

from src.cbow.hierarchical_cbow import HierarchicalCBOW
from src.skipgram.hierarchical_skipgram import HierarchicalSkipgram


class TestHierarchicalCBOW(unittest.TestCase):
    def make_hierarchical_cbow(self):
        model = HierarchicalCBOW.__new__(HierarchicalCBOW)

        model.V_size = 3
        model.embedding_dim = 2
        model.l_rate = 0.1

        model.input_hidden_matrix = np.array([
            [1.0, 0.0],   # word 0
            [0.0, 1.0],   # word 1
            [1.0, 1.0]    # word 2
        ], dtype=np.float32)

        model.hidden_output_matrix = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float32)

        model.word_codes = [
            [0],
            [1],
            [1, 0]
        ]

        model.word_paths = [
            [0],
            [1],
            [0, 2]
        ]

        return model

    def test_sigmoid_returns_correct_values(self):
        model = self.make_hierarchical_cbow()

        scores = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        probs = model.sigmoid(scores)

        expected = np.array([
            1.0 / (1.0 + np.exp(1.0)),
            0.5,
            1.0 / (1.0 + np.exp(-1.0))
        ], dtype=np.float32)

        self.assertTrue(np.allclose(probs, expected))

    def test_loss_one_path_matches_manual_value(self):
        model = self.make_hierarchical_cbow()

        scores = np.array([1.0, 0.5], dtype=np.float32)
        code = np.array([1.0, 0.0], dtype=np.float32)

        probs = 1.0 / (1.0 + np.exp(-scores))
        expected_loss = -(
            code[0] * np.log(probs[0] + 1e-10) +
            (1.0 - code[0]) * np.log(1.0 - probs[0] + 1e-10) +
            code[1] * np.log(probs[1] + 1e-10) +
            (1.0 - code[1]) * np.log(1.0 - probs[1] + 1e-10)
        )

        loss = model.loss_one_path(scores, code)

        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_feedforward_returns_correct_hidden_score_and_loss(self):
        model = self.make_hierarchical_cbow()

        context = [0, 2]
        target = 2

        hidden, score, loss = model.feedforward(context, target)

        expected_hidden = np.array([1.0, 0.5], dtype=np.float32)
        expected_score = np.array([1.0, 0.5], dtype=np.float32)

        probs = 1.0 / (1.0 + np.exp(-expected_score))
        expected_loss = -(
            np.log(probs[0] + 1e-10) +
            np.log(1.0 - probs[1] + 1e-10)
        )

        self.assertTrue(np.allclose(hidden, expected_hidden))
        self.assertTrue(np.allclose(score, expected_score))
        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_backpropagate_updates_only_path_columns_and_context_rows(self):
        model = self.make_hierarchical_cbow()

        context = [0, 2]
        target = 2

        hidden, score, _ = model.feedforward(context, target)

        old_input_hidden_matrix = model.input_hidden_matrix.copy()
        old_hidden_output_matrix = model.hidden_output_matrix.copy()

        model.backpropagate(hidden, target, score, context)

        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 0], old_hidden_output_matrix[:, 0]))
        self.assertTrue(np.allclose(model.hidden_output_matrix[:, 1], old_hidden_output_matrix[:, 1]))
        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 2], old_hidden_output_matrix[:, 2]))

        self.assertFalse(np.allclose(model.input_hidden_matrix[0], old_input_hidden_matrix[0]))
        self.assertTrue(np.allclose(model.input_hidden_matrix[1], old_input_hidden_matrix[1]))
        self.assertFalse(np.allclose(model.input_hidden_matrix[2], old_input_hidden_matrix[2]))

        update_word_0 = model.input_hidden_matrix[0] - old_input_hidden_matrix[0]
        update_word_2 = model.input_hidden_matrix[2] - old_input_hidden_matrix[2]
        self.assertTrue(np.allclose(update_word_0, update_word_2))


class TestHierarchicalSkipgram(unittest.TestCase):
    def make_hierarchical_skipgram(self):
        model = HierarchicalSkipgram.__new__(HierarchicalSkipgram)

        model.V_size = 3
        model.embedding_dim = 2
        model.l_rate = 0.1

        model.input_hidden_matrix = np.array([
            [1.0, 0.0],   # word 0
            [0.0, 1.0],   # word 1
            [1.0, 1.0]    # word 2
        ], dtype=np.float32)

        model.hidden_output_matrix = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float32)

        model.word_codes = [
            [0],
            [1],
            [1, 0]
        ]

        model.word_paths = [
            [0],
            [1],
            [0, 2]
        ]

        return model

    def test_sigmoid_returns_correct_values(self):
        model = self.make_hierarchical_skipgram()

        scores = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        probs = model.sigmoid(scores)

        expected = np.array([
            1.0 / (1.0 + np.exp(1.0)),
            0.5,
            1.0 / (1.0 + np.exp(-1.0))
        ], dtype=np.float32)

        self.assertTrue(np.allclose(probs, expected))

    def test_loss_one_path_matches_manual_value(self):
        model = self.make_hierarchical_skipgram()

        scores = np.array([0.0, 1.0], dtype=np.float32)
        code = np.array([1.0, 0.0], dtype=np.float32)

        probs = 1.0 / (1.0 + np.exp(-scores))
        expected_loss = -(
            code[0] * np.log(probs[0] + 1e-10) +
            (1.0 - code[0]) * np.log(1.0 - probs[0] + 1e-10) +
            code[1] * np.log(probs[1] + 1e-10) +
            (1.0 - code[1]) * np.log(1.0 - probs[1] + 1e-10)
        )

        loss = model.loss_one_path(scores, code)

        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_feedforward_returns_correct_hidden_and_loss(self):
        model = self.make_hierarchical_skipgram()

        center = 1
        context = [0, 2]

        center_out, hidden, loss = model.feedforward(center, context)

        expected_hidden = np.array([0.0, 1.0], dtype=np.float32)

        score_word_0 = np.array([0.0], dtype=np.float32)
        probs_word_0 = 1.0 / (1.0 + np.exp(-score_word_0))
        loss_word_0 = -np.log(1.0 - probs_word_0[0] + 1e-10)

        score_word_2 = np.array([0.0, 1.0], dtype=np.float32)
        probs_word_2 = 1.0 / (1.0 + np.exp(-score_word_2))
        loss_word_2 = -(
            np.log(probs_word_2[0] + 1e-10) +
            np.log(1.0 - probs_word_2[1] + 1e-10)
        )

        expected_loss = loss_word_0 + loss_word_2

        self.assertEqual(center_out, center)
        self.assertTrue(np.allclose(hidden, expected_hidden))
        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_backpropagate_updates_only_center_row_and_used_path_columns(self):
        model = self.make_hierarchical_skipgram()

        center = 1
        context = [0, 2]

        _, hidden, _ = model.feedforward(center, context)

        old_input_hidden_matrix = model.input_hidden_matrix.copy()
        old_hidden_output_matrix = model.hidden_output_matrix.copy()

        model.backpropagate(hidden, context, center)

        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 0], old_hidden_output_matrix[:, 0]))
        self.assertTrue(np.allclose(model.hidden_output_matrix[:, 1], old_hidden_output_matrix[:, 1]))
        self.assertFalse(np.allclose(model.hidden_output_matrix[:, 2], old_hidden_output_matrix[:, 2]))

        self.assertTrue(np.allclose(model.input_hidden_matrix[0], old_input_hidden_matrix[0]))
        self.assertFalse(np.allclose(model.input_hidden_matrix[1], old_input_hidden_matrix[1]))
        self.assertTrue(np.allclose(model.input_hidden_matrix[2], old_input_hidden_matrix[2]))


if __name__ == "__main__":
    unittest.main()