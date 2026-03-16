import time
import numpy as np

class Skipgram:
    def __init__(self, token_ids, V_size, context_size, embedding_dim, l_rate=0.025):
        """
        Initializes the Skipgram model with random weights and given hyperparameters.
        """
        self.V_size = V_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.l_rate = l_rate
        self.token_ids = token_ids
        self.token_count = len(token_ids)

        input_bound = np.sqrt(3.0 / self.embedding_dim)
        output_bound = np.sqrt(3.0 / self.embedding_dim)
        # input_hidden_matrix: (V_size, embedding_dim)
        self.input_hidden_matrix = np.random.uniform(-input_bound, input_bound,(self.V_size, self.embedding_dim)).astype(np.float32)
        # hidden_output_matrix: (embedding_dim, V_size)
        self.hidden_output_matrix = np.random.uniform(-output_bound, output_bound,(self.embedding_dim, self.V_size)).astype(np.float32)

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

    def softmax(self, output_vector):
        """
        Takes a vector of pre-softmax logits
        and returns the softmax probability vector.
        """
        max_value = np.max(output_vector)
        shifted_output = output_vector - max_value

        exp_values = np.exp(shifted_output)
        sum_exp = np.sum(exp_values)

        post_softmax = exp_values / sum_exp
        return post_softmax
    
    def loss_function(self, pre_softmax, context):
            """
            Computes the loss from the pre-softmax logits.
            pre_softmax : vector of raw logits u_j
            context     : true context words id
            returns     : scalar loss E
            """
            max_value = np.max(pre_softmax)
            shifted_output = pre_softmax - max_value

            sum_exp = np.sum(np.exp(shifted_output))

            E = 0.0
            for word in context:
                word_score = shifted_output[word]
                E += -word_score + np.log(sum_exp)

            return E

    def feedforward(self, center, context):
        """
        Performs one feedforward pass for the given training pair.
        """
        # hidden layer is just the input vector for the center word
        hidden = self.input_hidden_matrix[center].copy()
        pre_softmax = hidden @ self.hidden_output_matrix

        # softmax predicted probabilities for all words in the vocabulary
        post_softmax = self.softmax(pre_softmax)

        E = self.loss_function(pre_softmax, context)

        return center, hidden, post_softmax, E
    
    def backpropagate(self, post_softmax, hidden, context, center):
        """
        Applies one backpropagation step to the input_hidden_matrix and hidden_output_matrix for the given training pair.
        """
        # dE/du_c,j = y_c,j - t_c,j
        dE_du = len(context) * post_softmax.copy()
        np.add.at(dE_du, context, -1.0)

        # dE/dw'ij = dE/du_j * h_i
        # for the whole hidden_output_matrix
        dE_d_hidden_output_matrix = np.outer(hidden, dE_du)

        # dE/dh_i = SUM_j ( dE/du_j * w'ij )
        dE_dh = self.hidden_output_matrix @ dE_du

        # w'ij = w'ij - learning_rate * dE/dw'ij
        self.hidden_output_matrix -= self.l_rate * dE_d_hidden_output_matrix

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

                center, hidden, post_softmax, E = self.feedforward(center, context)
                self.backpropagate(post_softmax, hidden, context, center)

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