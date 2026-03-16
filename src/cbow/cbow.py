import time
import numpy as np

class CBOW:
    def __init__(self, token_ids, V_size, context_size, embedding_dim, l_rate=0.025):
        """ 
        Initializes the CBOW model with random weights and given hyperparameters.
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

    def make_cbow_training_pair(self, target_idx):
        """
        Builds one training example used for training cbow
        Each example looks like:
            (context_id, target_id)

        Example:
            context_ids = [12, 7, 31, 14]
            target_id = 20
        Produced pair:
            ([12, 7, 31, 14], 20)
        """
        context_words_on_each_side = self.context_size // 2
        context_ids = []

        target_id = int(self.token_ids[target_idx])

        left_start = target_idx - context_words_on_each_side
        while left_start < target_idx:
            if left_start >= 0:
                context_ids.append(int(self.token_ids[left_start]))
            left_start += 1

        right_start = target_idx + 1
        while right_start <= target_idx + context_words_on_each_side:
            if right_start < self.token_count:
                context_ids.append(int(self.token_ids[right_start]))
            right_start += 1

        if len(context_ids) > 0:
            return context_ids, target_id

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

    def loss_function(self, pre_softmax, target):
        """
        Computes the loss from the pre-softmax logits.
        pre_softmax : vector of raw logits u_j
        target      : true label word id
        returns     : scalar loss E
        """
        max_value = np.max(pre_softmax)
        shifted_output = pre_softmax - max_value

        target_score = shifted_output[target]
        sum_exp = np.sum(np.exp(shifted_output))

        E = -target_score + np.log(sum_exp)
        return E
    
    def feedforward(self, context, target):
        """
        Performs one feedforward pass for the given training pair.
        """
        # hidden layer is the average of the context word vectors
        hidden = np.mean(self.input_hidden_matrix[context], axis=0)
        pre_softmax = hidden @ self.hidden_output_matrix

        # post_softmax is the predicted probability distribution over all classes
        post_softmax = self.softmax(pre_softmax)

        E = self.loss_function(pre_softmax, target)

        return hidden, post_softmax, E

    def backpropagate(self, post_softmax, hidden, target, context):
        """
        Applies one backpropagation step to the input_hidden_matrix and hidden_output_matrix for the given training pair.
        post_softmax : final class probabilities
        hidden : values in the hidden layer
        target : true label word id
        context_sum : sum of all context word vectors
        """
        # dE/du_j = y_j - t_j
        dE_du = post_softmax.copy()
        dE_du[target] -= 1.0

        # dE/dw'ij = dE/du_j * h_i 
        # for the whole hidden_output_matrix
        dE_d_hidden_output_matrix = np.outer(hidden, dE_du)
        
        # dE/dh_i = SUM_j ( dE/du_j * w'ij )
        dE_dh = self.hidden_output_matrix @ dE_du

        # w'ij = w'ij - learning_rate * dE/dw'ij
        self.hidden_output_matrix -= self.l_rate * dE_d_hidden_output_matrix
        
        # w_ki = w_ki - learning_rate * (1/C) * dE/dw_ki
        context_size = len(context)
        for word in context:
            self.input_hidden_matrix[word] -= self.l_rate * (1.0 / context_size) * dE_dh

    def train(self, epochs=1, start_lr=0.025, end_lr=0.0001, power=10.0):
        """
        Trains the CBOW model for the given number of epochs.
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
                context, target = self.make_cbow_training_pair(i)

                if context is None:
                    continue

                hidden, post_softmax, E = self.feedforward(context, target)
                self.backpropagate(post_softmax, hidden, target, context)

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