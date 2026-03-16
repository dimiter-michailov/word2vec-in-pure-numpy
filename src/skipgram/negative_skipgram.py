import time
import numpy as np

class NegativeSkipgram:
    def __init__(self, token_ids, word_frequency, V_size, context_size, embedding_dim, l_rate=0.025):
        """
        Initializes the negative sampling Skipgram model with random weights and given hyperparameters.
        Builds the noise distribution based on the word frequencies for negative sampling."""
        # HARDCODED negative sample count for simplicity
        self.negative_sample_count_per_target = 20
        self.word_frequency = word_frequency

        self.V_size = V_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.l_rate = l_rate
        self.token_ids = token_ids
        self.token_count = len(token_ids)

        # Used for unigram distribution raised to 3/4
        self.noise_distribution = np.asarray(self.word_frequency, dtype=np.float64) ** 0.75
        self.noise_distribution /= np.sum(self.noise_distribution)

        self.noise_table_size = 1000000
        self.noise_table = np.zeros(self.noise_table_size, dtype=np.int32)

        cumulative_distribution = np.cumsum(self.noise_distribution)
        cumulative_distribution[-1] = 1.0

        word_id = 0
        for i in range(self.noise_table_size):
            ratio = (i + 1) / self.noise_table_size
            while ratio > cumulative_distribution[word_id]:
                word_id += 1
            self.noise_table[i] = word_id

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
    
    def loss_function(self, positive_logit, negative_logits):
        """
        Computes the negative-sampling loss for one positive word and its negative samples.
        positive_logit : scalar raw logit for the true output word
        negative_logits: vector of raw logits for sampled negative words
        returns        : scalar loss E
        """
        positive_prob = self.sigmoid(positive_logit)
        negative_prob = self.sigmoid(-negative_logits)

        E = -np.log(positive_prob + 1e-10)
        E += -np.sum(np.log(negative_prob + 1e-10))

        return E

    def negative_sampling(self, center, context):
        """
        Generates negative samples for the given center and context words.
        """
        total_negative_count = self.negative_sample_count_per_target * len(context)

        blocked_words = set(context)
        blocked_words.add(center)

        random_idx = np.random.randint(0, self.noise_table_size, size=total_negative_count)

        # sample based on the unigram distribution raised to 3/4, using the pre-built noise table
        samples = self.noise_table[random_idx].astype(np.int32)

        # negative samples do not include any of the context words or the center word
        for i in range(total_negative_count):
            while samples[i] in blocked_words:
                new_idx = np.random.randint(0, self.noise_table_size)
                samples[i] = self.noise_table[new_idx]

            blocked_words.add(int(samples[i]))

        return samples

    def feedforward(self, center, context):
        """
        Performs one feedforward pass for the given training pair.
        """
        # project the weights of the input word
        hidden = self.input_hidden_matrix[center].copy()

        # get all negative samples
        all_negative_samples = self.negative_sampling(center, context)

        sampled_data = []
        E = 0.0
        
        start = 0

        # compute loss for each context word using its positive logit and negative logits
        for word in context:
            end = start + self.negative_sample_count_per_target
            negative_samples = all_negative_samples[start:end]
            start = end

            positive_logit = hidden @ self.hidden_output_matrix[:, word]

            negative_vectors = self.hidden_output_matrix[:, negative_samples]
            negative_logits = hidden @ negative_vectors

            # pass the computed values for backpropagation
            sampled_data.append((word, negative_samples, positive_logit, negative_logits))

            # accumulate loss
            E += self.loss_function(positive_logit, negative_logits)

        return center, hidden, sampled_data, E
    
    def backpropagate(self, sampled_data, hidden, center):
        """
        Applies one backpropagation step to the input_hidden_matrix and hidden_output_matrix for the given training pair.
        """
        dE_dh = np.zeros(self.embedding_dim, dtype=np.float32)

        for word, negative_samples, positive_logit, negative_logits in sampled_data:
            positive_prob = self.sigmoid(positive_logit)

            #dE/du_out = sig(v'_w_out T hidden) - t_j
            dE_du_positive = positive_prob - 1.0

            #old positive vector for desired context word
            dE_dh += dE_du_positive * self.hidden_output_matrix[:, word]

            #v'w_out = v'w_out - l_rate * dE/du_out * hidden
            self.hidden_output_matrix[:, word] -= self.l_rate * dE_du_positive * hidden

            # negative word gradients (dE/du_neg)
            negative_probs = self.sigmoid(negative_logits)
            dE_du_negative = negative_probs

            #old positive vector for desired negative samples
            dE_dh += self.hidden_output_matrix[:, negative_samples] @ dE_du_negative

            #v'w_neg = v'w_neg - l_rate * dE/du_neg * hidden
            self.hidden_output_matrix[:, negative_samples] -= self.l_rate * (hidden[:, None] * dE_du_negative[None, :])

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

                center, hidden, sampled_data, E = self.feedforward(center, context)
                self.backpropagate(sampled_data, hidden, center)

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