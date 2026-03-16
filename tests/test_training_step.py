import unittest
import numpy as np
from src.cbow.cbow import CBOW
from src.skipgram.skipgram import Skipgram

class TestCBOWTrainingStep(unittest.TestCase):

    def make_small_cbow(self):
        cbow = CBOW.__new__(CBOW)

        cbow.V_size = 3
        cbow.embedding_dim = 2
        cbow.l_rate = 0.1

        cbow.input_hidden_matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], dtype=np.float32)

        cbow.hidden_output_matrix = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float32)

        return cbow


    def test_softmax_outputs_valid_distribution(self):
        cbow = self.make_small_cbow()

        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = cbow.softmax(logits)

        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        self.assertTrue(np.all(probs > 0))


    def test_feedforward_hidden_is_mean_of_context(self):
        cbow = self.make_small_cbow()

        context = [0, 2]
        target = 1

        hidden, probs, loss = cbow.feedforward(context, target)

        expected_hidden = np.array([1.0, 0.5], dtype=np.float32)

        self.assertTrue(np.allclose(hidden, expected_hidden))


    def test_feedforward_logits_are_correct(self):
        cbow = self.make_small_cbow()

        context = [0, 2]
        target = 1

        hidden, probs, loss = cbow.feedforward(context, target)

        expected_hidden = np.array([1.0, 0.5])
        expected_logits = expected_hidden @ cbow.hidden_output_matrix

        computed_logits = hidden @ cbow.hidden_output_matrix

        self.assertTrue(np.allclose(expected_logits, computed_logits))


    def test_backpropagate_changes_parameters(self):
        cbow = self.make_small_cbow()

        context = [0, 2]
        target = 1

        hidden, probs, loss = cbow.feedforward(context, target)

        old_input = cbow.input_hidden_matrix.copy()
        old_output = cbow.hidden_output_matrix.copy()

        cbow.backpropagate(probs, hidden, target, context)

        self.assertFalse(np.allclose(old_output, cbow.hidden_output_matrix))
        self.assertFalse(np.allclose(old_input, cbow.input_hidden_matrix))


    def test_backpropagate_updates_only_context_words(self):
        cbow = self.make_small_cbow()

        context = [0, 2]
        target = 1

        hidden, probs, loss = cbow.feedforward(context, target)

        old_input = cbow.input_hidden_matrix.copy()

        cbow.backpropagate(probs, hidden, target, context)

        self.assertFalse(np.allclose(old_input[0], cbow.input_hidden_matrix[0]))
        self.assertTrue(np.allclose(old_input[1], cbow.input_hidden_matrix[1]))
        self.assertFalse(np.allclose(old_input[2], cbow.input_hidden_matrix[2]))


class TestSkipgramTrainingStep(unittest.TestCase):

    def make_small_skipgram(self):
        skipgram = Skipgram.__new__(Skipgram)

        skipgram.V_size = 3
        skipgram.embedding_dim = 2
        skipgram.l_rate = 0.1

        skipgram.input_hidden_matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], dtype=np.float32)

        skipgram.hidden_output_matrix = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0]
        ], dtype=np.float32)

        return skipgram


    def test_feedforward_hidden_is_center_vector(self):
        skipgram = self.make_small_skipgram()

        center = 1
        context = [0, 2]

        center_out, hidden, probs, loss = skipgram.feedforward(center, context)

        expected_hidden = np.array([0.0, 1.0])

        self.assertTrue(np.allclose(hidden, expected_hidden))


    def test_skipgram_loss_matches_manual_calculation(self):
        skipgram = self.make_small_skipgram()

        logits = np.array([0.0, 1.0, 1.0])
        context = [0, 2]

        shifted = logits - np.max(logits)
        sum_exp = np.sum(np.exp(shifted))

        expected_loss = 0
        for w in context:
            expected_loss += -shifted[w] + np.log(sum_exp)

        loss = skipgram.loss_function(logits, context)

        self.assertAlmostEqual(loss, expected_loss, places=6)


    def test_backpropagate_updates_only_center_word(self):
        skipgram = self.make_small_skipgram()

        center = 1
        context = [0, 2]

        _, hidden, probs, loss = skipgram.feedforward(center, context)

        old_input = skipgram.input_hidden_matrix.copy()

        skipgram.backpropagate(probs, hidden, context, center)

        self.assertTrue(np.allclose(old_input[0], skipgram.input_hidden_matrix[0]))
        self.assertFalse(np.allclose(old_input[1], skipgram.input_hidden_matrix[1]))
        self.assertTrue(np.allclose(old_input[2], skipgram.input_hidden_matrix[2]))


if __name__ == "__main__":
    unittest.main()