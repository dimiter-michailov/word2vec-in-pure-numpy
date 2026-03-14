import time
import numpy as np

class NegativeSkipgram:
    def __init__(self, token_ids, word_frequency, V_size, context_size, embedding_dim, l_rate=0.025):
        # HARDCODED
        self.negative_sample_count_per_target = 20
        self.word_frequency = word_frequency

        self.token_ids = token_ids
        self.V_size = V_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.l_rate = l_rate
        self.pairs = self.make_skipgram_training_pairs(token_ids, self.context_size)

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

    def make_skipgram_training_pairs(self, token_ids, total_context_size: int):
        """
        Builds training examples used for training skipgram
        Each example looks like:
            (center_id, context_id)
        Example:
            center_id = 20
            context_ids = [12, 7, 31, 14]
        Resulting pair:
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

    def sigmoid(self, score):
        score = np.asarray(score, dtype=np.float32)
        prob = np.empty_like(score, dtype=np.float32)

        positive = score >= 0
        prob[positive] = 1.0 / (1.0 + np.exp(-score[positive]))

        exp_score = np.exp(score[~positive])
        prob[~positive] = exp_score / (1.0 + exp_score)

        return prob
    
    def loss_function(self, positive_score, negative_scores):
        """
        Computes the negative-sampling loss for one positive word and its negative samples.
        positive_score : scalar raw score for the true output word
        negative_scores: vector of raw scores for sampled negative words
        returns        : scalar loss E
        """
        positive_prob = self.sigmoid(positive_score)
        negative_prob = self.sigmoid(-negative_scores)

        E = -np.log(positive_prob + 1e-10)
        E += -np.sum(np.log(negative_prob + 1e-10))

        return E

    def negative_sampling(self, center, context):
        total_negative_count = self.negative_sample_count_per_target * len(context)

        blocked_words = set(context)
        blocked_words.add(center)

        random_idx = np.random.randint(0, self.noise_table_size, size=total_negative_count)
        samples = self.noise_table[random_idx].astype(np.int32)

        for i in range(total_negative_count):
            while samples[i] in blocked_words:
                new_idx = np.random.randint(0, self.noise_table_size)
                samples[i] = self.noise_table[new_idx]

        return samples

    def feedforward(self, center, context):
        # project the weights of the input word
        hidden = self.input_hidden_matrix[center].copy()

        all_negative_samples = self.negative_sampling(center, context)

        sampled_data = []
        E = 0.0
        
        start = 0

        for word in context:
            end = start + self.negative_sample_count_per_target
            negative_samples = all_negative_samples[start:end]
            start = end

            positive_score = hidden @ self.hidden_output_matrix[:, word]

            negative_vectors = self.hidden_output_matrix[:, negative_samples]
            negative_scores = hidden @ negative_vectors

            sampled_data.append((word, negative_samples, positive_score, negative_scores))

            E += self.loss_function(positive_score, negative_scores)

        return center, hidden, sampled_data, E
    
    def backpropagate(self, sampled_data, hidden, center):
        """
        Applies one backpropagation step to the input_hidden_matrix and hidden_output_matrix for the given training pair.
        """
        dE_dh = np.zeros(self.embedding_dim, dtype=np.float32)

        for word, negative_samples, positive_score, negative_scores in sampled_data:
            positive_prob = self.sigmoid(positive_score)

            #dE/du_out = sig(v'_w_out T hidden) - t_j
            dE_du_positive = positive_prob - 1.0

            #old positive vector for desired context word
            old_positive_vector = self.hidden_output_matrix[:, word].copy()

            #v'w_out = v'w_out - l_rate * dE/du_out * hidden
            self.hidden_output_matrix[:, word] -= self.l_rate * dE_du_positive * hidden

            # accumulate dE/dh
            dE_dh += dE_du_positive * old_positive_vector

            # negative word gradients (dE/du_neg)
            negative_probs = self.sigmoid(negative_scores)
            dE_du_negative = negative_probs

            #old positive vector for desired negative samples
            old_negative_vectors = self.hidden_output_matrix[:, negative_samples].copy()

            #v'w_neg = v'w_neg - l_rate * dE/du_neg * hidden
            self.hidden_output_matrix[:, negative_samples] -= self.l_rate * (hidden[:, None] * dE_du_negative[None, :])

            dE_dh += old_negative_vectors @ dE_du_negative

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

                center, hidden, sampled_data, E = self.feedforward(center, context)
                self.backpropagate(sampled_data, hidden, center)

                total_loss = total_loss + E
                if i % 50000 == 0:
                    elapsed = time.time() - epoch_start
                    print("epoch", epoch + 1, "pair", i, "out of", len(self.pairs), "time:", round(elapsed, 2), "sec")
    
            average_loss = total_loss / len(self.pairs)
            print("epoch:", epoch + 1, "average_loss:", average_loss)

            progress = (epoch + 1) / epochs
            self.l_rate = end_lr + (start_lr - end_lr) * (1.0 - progress**power)