import numpy as np
import time
from huffman_tree import HuffmanTree

class HierarchicalSkipgram:
    def __init__(self, token_ids, word_frequency, V_size, context_size, embedding_dim, l_rate=0.025):
        """
        Initializes the hierarchical Skipgram model with random weights and given hyperparameters.
        Builds the Huffman tree based on the word frequencies to get the codes and paths for each word.
        """
        self.V_size = V_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.l_rate = l_rate
        self.token_ids = token_ids
        self.token_count = len(token_ids)

        self.tree = HuffmanTree(word_frequency)
        print("internal tree nodes:", len(self.tree.count) - self.V_size)
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
        # hidden_output_matrix: (embedding_dim, num_internal_nodes)
        self.hidden_output_matrix = np.random.uniform(-output_bound, output_bound,(self.embedding_dim, self.num_internal_nodes)).astype(np.float32)

    def make_skipgram_training_pair(self, center_idx):
        """
        Builds one training example used for training skipgram
        Each example looks like:
            (center_id, context_id)

        Example:
            center_id = 20
            context_ids = [12, 7, 31, 14]
        Produced pair:
            (20, [12, 7, 31, 14])
        """
        context_words_on_each_side = self.context_size // 2
        context_ids = []

        center_id = int(self.token_ids[center_idx])

        left_start = center_idx - context_words_on_each_side
        while left_start < center_idx:
            if left_start >= 0:
                context_ids.append(int(self.token_ids[left_start]))
            left_start += 1
        
        right_start = center_idx + 1
        while right_start <= center_idx + context_words_on_each_side:
            if right_start < self.token_count:
                context_ids.append(int(self.token_ids[right_start]))
            right_start += 1

        if len(context_ids) > 0:
            return center_id, context_ids

        return None, None

    def sigmoid(self, logits):
        """ 
        Sigmoid function that is numerically stable for large positive or negative logits.
        """
        logits = np.asarray(logits, dtype=np.float32)
        prob = np.empty_like(logits, dtype=np.float32)

        positive = logits >= 0
        prob[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))

        exp_logits = np.exp(logits[~positive])
        prob[~positive] = exp_logits / (1.0 + exp_logits)

        return prob
    
    def loss_one_path(self, logits, code):
        """
        Computes the loss for one path in the hierarchical softmax.
        """
        probs = self.sigmoid(logits)

        E = -(code * np.log(probs + 1e-10) + (1.0 - code) * np.log(1.0 - probs + 1e-10))
        return np.sum(E)

    def feedforward(self, center, context):
        """
        Performs one feedforward pass for the given training pair.
        """
        # hidden layer is just the input vector for the center word
        hidden = self.input_hidden_matrix[center].copy()
        
        E = 0.0

        # compute loss for each context word using its path and code in the hierarchical softmax
        for target_word in context:
            path = np.asarray(self.word_paths[target_word], dtype=np.int32)
            code = np.asarray(self.word_codes[target_word], dtype=np.float32)

            node_vectors = self.hidden_output_matrix[:, path]
            logits = hidden @ node_vectors
            E += self.loss_one_path(logits, code)

        return center, hidden, E
    
    def backpropagate(self, hidden, context, center):
        """
        Applies one backpropagation step to both to the matricies and the hierarchical softmax for the given training pair.
        """
        dE_dh = np.zeros(self.embedding_dim, dtype=np.float32)

        for target_word in context:
            path = np.asarray(self.word_paths[target_word], dtype=np.int32)
            code = np.asarray(self.word_codes[target_word], dtype=np.float32)

            # vectors for the internal nodes along the path
            node_vectors = self.hidden_output_matrix[:, path]
            logits = hidden @ node_vectors
            # sig(v_prime_j T hidden)
            probs = self.sigmoid(logits)

            # sig(v_prime_j T hidden) - t_j
            dE_du_j = probs - code

            # accumulate dE_dh over all path nodes
            dE_dh += node_vectors @ dE_du_j

            # v_prime_j = v_prime_j - l_rate * (sig(v_prime_j T hidden) - t_j) T hidden
            self.hidden_output_matrix[:, path] -= self.l_rate * (hidden[:, None] * dE_du_j[None, :])

        # dE/dw_ki = dE/dh_i * x_k
        # dE_d_input_hidden_matrix = np.outer(center_vector, dE_dh)

        # w_ki = w_ki - learning_rate * dE/dw_ki
        self.input_hidden_matrix[center] -= self.l_rate * dE_dh

    def train(self, epochs=1, start_lr=0.025, end_lr=0.0001, power=10.0):
        """
        Trains the model for the given number of epochs.
        Applies learning rate decay from start_lr to end_lr over the epochs using a power function.
        """
        self.l_rate = start_lr

        for epoch in range(epochs):
            print("\nstarting epoch", epoch + 1)
            print("learning rate:", self.l_rate)
            total_loss = 0.0
            pair_count = 0
            epoch_start = time.time()
        
            for i in range(self.token_count):
                center, context = self.make_skipgram_training_pair(i)

                if center is None:
                    continue

                center, hidden, E = self.feedforward(center, context)
                self.backpropagate(hidden, context, center)

                total_loss = total_loss + E
                pair_count += 1

                if pair_count % 50000 == 0:
                    elapsed = time.time() - epoch_start
                    print("epoch", epoch + 1, "pair", pair_count, "time:", round(elapsed, 2), "sec")
        
            average_loss = total_loss / pair_count
            print("epoch:", epoch + 1, "average_loss:", average_loss)

            # Update learning rate with decay
            progress = (epoch + 1) / epochs
            self.l_rate = end_lr + (start_lr - end_lr) * (1.0 - progress**power)