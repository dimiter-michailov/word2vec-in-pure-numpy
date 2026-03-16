import time
import numpy as np

class NegativeCBOW:
    def __init__(self, token_ids, word_frequency, V_size, context_size, embedding_dim, l_rate=0.025):
        """
        Initializes the negative sampling CBOW model with random weights and given hyperparameters.
        Builds the noise distribution based on the word frequencies for negative sampling.
        """
        # HARDCODED negative sample count for simplicity
        self.negative_sample_count = 20
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

    def sigmoid(self, logit):
        """
        Computes the sigmoid function for the given logit(s).
        Handles both scalar and vector inputs."""
        logit = np.asarray(logit, dtype=np.float32)
        prob = np.empty_like(logit, dtype=np.float32)

        positive = logit >= 0
        prob[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))

        exp_score = np.exp(logit[~positive])
        prob[~positive] = exp_score / (1.0 + exp_score)

        return prob
    
    def negative_sampling(self, context, target):
        """
        Generates negative samples for the given context and target word.
        """
        blocked_words = set(context)
        blocked_words.add(target)

        random_idx = np.random.randint(0, self.noise_table_size, size=self.negative_sample_count)
        samples = self.noise_table[random_idx].astype(np.int32)

        for i in range(self.negative_sample_count):
            # negative samples do not include any of the context words or the target word
            while samples[i] in blocked_words:
                new_idx = np.random.randint(0, self.noise_table_size)
                samples[i] = self.noise_table[new_idx]

            blocked_words.add(int(samples[i]))

        return samples

    def loss_function(self, positive_logit, negative_logits):
        """
        Computes the loss from the pre-softmax logits.
        pre_softmax : vector of raw logits u_j
        target      : true target word id
        returns     : scalar loss E
        """
        positive_prob = self.sigmoid(positive_logit)
        negative_prob = self.sigmoid(-negative_logits)

        E = -np.log(positive_prob + 1e-10)
        E += -np.sum(np.log(negative_prob + 1e-10))

        return E
    
    def feedforward(self, context, target):
        """
        Performs one feedforward pass for the given training pair.
        """
        # mean of context word vectors
        hidden = np.mean(self.input_hidden_matrix[context], axis=0)
        # get negative samples
        negative_samples = self.negative_sampling(context, target)
        # positive logit for the target word
        positive_logit = hidden @ self.hidden_output_matrix[:, target]

        negative_vectors = self.hidden_output_matrix[:, negative_samples]
        # negative logits for the negative samples
        negative_logits = hidden @ negative_vectors

        E = self.loss_function(positive_logit, negative_logits)
        
        return hidden, target, negative_samples, positive_logit, negative_logits, E

    def backpropagate(self, hidden, target, negative_samples, positive_logit, negative_logits, context):
        """
        Applies one backpropagation step to the input_hidden_matrix and hidden_output_matrix for the given training pair.
        """
        dE_dh = np.zeros(self.embedding_dim, dtype=np.float32)

        positive_prob = self.sigmoid(positive_logit)

        # dE/du_out = sig(v'_w_out T hidden) - t_j
        dE_du_positive = positive_prob - 1.0

        # accumulate dE/dh
        dE_dh += dE_du_positive * self.hidden_output_matrix[:, target]

        # v'w_out = v'w_out - l_rate * dE/du_out * hidden
        self.hidden_output_matrix[:, target] -= self.l_rate * dE_du_positive * hidden

        # negative word gradients (dE/du_neg)
        negative_probs = self.sigmoid(negative_logits)
        dE_du_negative = negative_probs

        dE_dh += self.hidden_output_matrix[:, negative_samples] @ dE_du_negative

        # v'w_neg = v'w_neg - l_rate * dE/du_neg * hidden
        self.hidden_output_matrix[:, negative_samples] -= self.l_rate * (hidden[:, None] * dE_du_negative[None, :])

        # w_ki = w_ki - learning_rate * (1/C) * dE/dw_ki
        context_size = len(context)
        for word in context:
            self.input_hidden_matrix[word] -= self.l_rate * (1.0 / context_size) * dE_dh

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
                context, target = self.make_cbow_training_pair(i)

                if context is None:
                    continue

                hidden, target, negative_samples, positive_logit, negative_logits, E = self.feedforward(context, target)
                self.backpropagate(hidden, target, negative_samples, positive_logit, negative_logits, context)

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