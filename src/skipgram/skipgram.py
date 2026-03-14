import time
import numpy as np

class Skipgram:
    def __init__(self, token_ids, V_size, context_size, embedding_dim, l_rate=0.025):
            self.V_size = V_size
            self.context_size = context_size
            self.embedding_dim = embedding_dim
            self.l_rate = l_rate
            self.pairs = self.make_skipgram_training_pairs(token_ids, self.context_size)

            input_bound = np.sqrt(3.0 / self.embedding_dim)
            output_bound = np.sqrt(3.0 / self.embedding_dim)
            # input_hidden_matrix: (V_size, embedding_dim)
            self.input_hidden_matrix = np.random.uniform(-input_bound, input_bound,(self.V_size, self.embedding_dim)).astype(np.float32)
            # hidden_output_matrix: (embedding_dim, V_size)
            self.hidden_output_matrix = np.random.uniform(-output_bound, output_bound,(self.embedding_dim, self.V_size)).astype(np.float32)

    def make_skipgram_training_pairs(self, token_ids, total_context_size: int):
        """
        Builds training example used for training skipgram
        Each example looks like:
            (center_id, context_id)

        Example:
            center_id = 20
            context_ids = [12, 7, 31, 14]
        Build pairs:
            (20, [12, 7, 31, 14])
        """
        pairs = []
        context_words_on_each_side = total_context_size // 2
        context_ids = []

        for center_idx in range(len(token_ids)):
            center_id = token_ids[center_idx]

            context_ids = []

            left_start = center_idx - context_words_on_each_side
            while left_start < center_idx:
                if left_start >= 0:
                    context_ids.append(token_ids[left_start])
                left_start += 1
            
            right_start = center_idx + 1
            while right_start <= center_idx + context_words_on_each_side:
                if right_start < len(token_ids):
                    context_ids.append(token_ids[right_start])
                right_start += 1

            if len(context_ids) > 0:
                pair = (center_id, context_ids)
                pairs.append(pair)

        return pairs

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
            context      : true context words id
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
        # to keep vector sum of all the words used in the context
        hidden = self.input_hidden_matrix[center].copy()
        pre_softmax = hidden @ self.hidden_output_matrix
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

        # save OLD hidden_output_matrix before updating
        old_hidden_output_matrix = self.hidden_output_matrix.copy()

        # w'ij = w'ij - learning_rate * dE/dw'ij
        self.hidden_output_matrix -= self.l_rate * dE_d_hidden_output_matrix

        # dE/dh_i = SUM_j ( dE/du_j * w'ij )
        dE_dh = old_hidden_output_matrix @ dE_du

        # dE/dw_ki = dE/dh_i * x_k
        # dE_d_input_hidden_matrix = np.outer(center_vector, dE_dh)

        # w_ki = w_ki - learning_rate * dE/dw_ki
        self.input_hidden_matrix[center] -= self.l_rate * dE_dh

    def train(self, epochs=1, start_lr=0.025, end_lr=0.0001, power=10.0):
        self.l_rate = start_lr

        for epoch in range(epochs):
            print("\nstarting epoch", epoch + 1)
            print("learning rate:", self.l_rate)
            total_loss = 0.0
            epoch_start = time.time()
        
            for i, p in enumerate(self.pairs):
                center = p[0]
                context = p[1]

                center, hidden, post_softmax, E = self.feedforward(center, context)
                self.backpropagate(post_softmax, hidden, context, center)

                total_loss = total_loss + E
                if i % 50000 == 0:
                    elapsed = time.time() - epoch_start
                    print("epoch", epoch + 1, "pair", i, "out of", len(self.pairs), "time:", round(elapsed, 2), "sec")
    
            average_loss = total_loss / len(self.pairs)
            print("epoch:", epoch + 1, "average_loss:", average_loss)

            progress = (epoch + 1) / epochs
            self.l_rate = end_lr + (start_lr - end_lr) * (1.0 - progress**power)