import time
import numpy as np
from huffman_tree import HuffmanTree

class HierarchicalCBOW:
    def __init__(self, token_ids, word_frequency, V_size, context_size, embedding_dim, l_rate=0.025):
        self.V_size = V_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.l_rate = l_rate
        self.pairs = self.make_cbow_training_pairs(token_ids, self.context_size)

        self.tree = HuffmanTree(word_frequency)
        print("num tree nodes:", len(self.tree.count))
        print("word 0 code:", self.tree.word_codes[0])
        print("word 0 path:", self.tree.word_paths[0])
        print("word 1 code:", self.tree.word_codes[1])
        print("word 1 path:", self.tree.word_paths[1])

        self.word_codes = self.tree.word_codes
        self.word_paths = self.tree.word_paths
        self.num_internal_nodes = self.V_size - 1

        input_bound = np.sqrt(3.0 / self.embedding_dim)
        output_bound = np.sqrt(3.0 / self.embedding_dim)
        # input_hidden_matrix: (V_size, embedding_dim)
        self.input_hidden_matrix = np.random.uniform(-input_bound, input_bound,(self.V_size, self.embedding_dim)).astype(np.float32)
        # hidden_output_matrix: (embedding_dim, V_size)
        self.hidden_output_matrix = np.random.uniform(-output_bound, output_bound,(self.embedding_dim, self.V_size)).astype(np.float32)

    def make_cbow_training_pairs(self, token_ids, total_context_size: int):
        """
        Builds training example used for training cbow
        Each example looks like:
            (context_id, target_id)

        Example:
            context_ids = [12, 7, 31, 14]
            target_id = 20
        Build pairs:
            ([12, 7, 31, 14], 20)
        """
        pairs = []
        context_words_on_each_side = total_context_size // 2
        context_ids = []

        for target_idx in range(len(token_ids)):
            target_id = token_ids[target_idx]

            context_ids = []

            left_start = target_idx - context_words_on_each_side
            while left_start < target_idx:
                if left_start >= 0:
                    context_ids.append(token_ids[left_start])
                left_start += 1
            
            right_start = target_idx + 1
            while right_start <= target_idx + context_words_on_each_side:
                if right_start < len(token_ids):
                    context_ids.append(token_ids[right_start])
                right_start += 1

            if len(context_ids) > 0:
                pair = (context_ids, target_id)
                pairs.append(pair)

        print("number of training pairs:", len(pairs))
        return pairs

    def sigmoid(self, score):
        score = np.asarray(score, dtype=np.float32)
        prob = np.empty_like(score, dtype=np.float32)

        positive = score >= 0
        prob[positive] = 1.0 / (1.0 + np.exp(-score[positive]))

        exp_score = np.exp(score[~positive])
        prob[~positive] = exp_score / (1.0 + exp_score)

        return prob

    def loss_one_path(self, score, code):
        probs = self.sigmoid(score)

        E = -(code * np.log(probs + 1e-10) + (1.0 - code) * np.log(1.0 - probs + 1e-10))
        return np.sum(E)
    
    def feedforward(self, context, target):
        # to keep vector sum of all the words used in the context
        hidden = np.mean(self.input_hidden_matrix[context], axis=0)

        path = np.asarray(self.word_paths[target], dtype=np.int32)
        code = np.asarray(self.word_codes[target], dtype=np.float32)

        node_vectors = self.hidden_output_matrix[:, path]
        score = hidden @ node_vectors
        E = self.loss_one_path(score, code)

        return hidden, score, E

    def backpropagate(self, hidden, target, score, context):
        """
        Applies one backpropagation step to both to the matricies and the hierarchical softmax for the given training pair.
        """
        dE_dh = np.zeros(self.embedding_dim, dtype=np.float32)

        path = np.asarray(self.word_paths[target], dtype=np.int32)
        code = np.asarray(self.word_codes[target], dtype=np.float32)

        node_vectors = self.hidden_output_matrix[:, path]
        # sig(v_prime_j T hidden)
        probs = self.sigmoid(score)

        # sig(v_prime_j T hidden) - t_j
        dE_du_j = probs - code

        # old v_prime_j
        old_node_vector = node_vectors.copy()

        # v_prime_j = v_prime_j - l_rate * (sig(v_prime_j T hidden) - t_j) T hidden
        self.hidden_output_matrix[:, path] -= self.l_rate * (hidden[:, None] * dE_du_j[None, :])

        # accumulate dE_dh over all path nodes
        dE_dh += old_node_vector @ dE_du_j

        # dE/dw_ki = dE/dh_i * x_k
        # w_ki = w_ki - learning_rate * (1/C) * dE/dw_ki
        context_size = len(context)
        update = self.l_rate * (1.0 / context_size) * dE_dh

        for word in context:
            self.input_hidden_matrix[word] -= update

    def train(self, epochs=1, start_lr=0.025, end_lr=0.0001, power=10.0):
        self.l_rate = start_lr

        for epoch in range(epochs):
            print("\nstarting epoch", epoch + 1)
            print("learning rate:", self.l_rate)
            total_loss = 0.0
            epoch_start = time.time()
        
            for i, p in enumerate(self.pairs):
                context = p[0]
                target = p[1]

                hidden, score, E = self.feedforward(context, target)
                self.backpropagate(hidden, target, score, context)

                total_loss = total_loss + E
                if i % 50000 == 0:
                    elapsed = time.time() - epoch_start
                    print("epoch", epoch + 1, "pair", i, "out of", len(self.pairs), "time:", round(elapsed, 2), "sec")
    
            average_loss = total_loss / len(self.pairs)
            print("epoch:", epoch + 1, "average_loss:", average_loss)

            progress = (epoch + 1) / epochs
            self.l_rate = end_lr + (start_lr - end_lr) * (1.0 - progress**power)