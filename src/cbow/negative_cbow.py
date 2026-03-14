import time
import numpy as np

class NegativeCBOW:
    def __init__(self, token_ids, word_frequency, V_size, context_size, embedding_dim, l_rate=0.025):
        # HARDCODED
        self.negative_sample_count = 20
        self.word_frequency = word_frequency

        self.token_ids = token_ids
        self.V_size = V_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.l_rate = l_rate
        self.pairs = self.make_cbow_training_pairs(token_ids, self.context_size)

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
    
    def negative_sampling(self, context, target):
        blocked_words = set(context)
        blocked_words.add(target)

        random_idx = np.random.randint(0, self.noise_table_size, size=self.negative_sample_count)
        samples = self.noise_table[random_idx].astype(np.int32)

        for i in range(self.negative_sample_count):
            while samples[i] in blocked_words:
                new_idx = np.random.randint(0, self.noise_table_size)
                samples[i] = self.noise_table[new_idx]
        return samples

    def loss_function(self, positive_score, negative_scores):
        """
        Computes the loss from the pre-softmax logits.
        pre_softmax : vector of raw logits u_j
        target      : true target word id
        returns     : scalar loss E
        """
        positive_prob = self.sigmoid(positive_score)
        negative_prob = self.sigmoid(-negative_scores)

        E = -np.log(positive_prob + 1e-10)
        E += -np.sum(np.log(negative_prob + 1e-10))

        return E
    
    def feedforward(self, context, target):
        # to keep vector sum of all the words used in the context
        hidden = np.mean(self.input_hidden_matrix[context], axis=0)
        
        negative_samples = self.negative_sampling(context, target)
        positive_score = hidden @ self.hidden_output_matrix[:, target]

        negative_vectors = self.hidden_output_matrix[:, negative_samples]
        negative_scores = hidden @ negative_vectors

        E = self.loss_function(positive_score, negative_scores)
        
        return hidden, target, negative_samples, positive_score, negative_scores, E

    def backpropagate(self, hidden, target, negative_samples, positive_score, negative_scores, context):
        """
        Applies one backpropagation step to the input_hidden_matrix and hidden_output_matrix for the given training pair.
        """
        dE_dh = np.zeros(self.embedding_dim, dtype=np.float32)

        positive_prob = self.sigmoid(positive_score)

        # dE/du_out = sig(v'_w_out T hidden) - t_j
        dE_du_positive = positive_prob - 1.0

        # old positive vector for desired target word
        old_positive_vector = self.hidden_output_matrix[:, target].copy()

        # v'w_out = v'w_out - l_rate * dE/du_out * hidden
        self.hidden_output_matrix[:, target] -= self.l_rate * dE_du_positive * hidden

        # accumulate dE/dh
        dE_dh += dE_du_positive * old_positive_vector

        # negative word gradients (dE/du_neg)
        negative_probs = self.sigmoid(negative_scores)
        dE_du_negative = negative_probs

        # old vectors for desired negative samples
        old_negative_vectors = self.hidden_output_matrix[:, negative_samples].copy()

        # v'w_neg = v'w_neg - l_rate * dE/du_neg * hidden
        self.hidden_output_matrix[:, negative_samples] -= self.l_rate * (hidden[:, None] * dE_du_negative[None, :])

        dE_dh += old_negative_vectors @ dE_du_negative

        # w_ki = w_ki - learning_rate * (1/C) * dE/dw_ki
        context_size = len(context)
        for word in context:
            self.input_hidden_matrix[word] -= self.l_rate * (1.0 / context_size) * dE_dh

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

                hidden, target, negative_samples, positive_score, negative_scores, E = self.feedforward(context, target)
                self.backpropagate(hidden, target, negative_samples, positive_score, negative_scores, context)

                total_loss = total_loss + E
                if i % 50000 == 0:
                    elapsed = time.time() - epoch_start
                    print("epoch", epoch + 1, "pair", i, "out of", len(self.pairs), "time:", round(elapsed, 2), "sec")
    
            average_loss = total_loss / len(self.pairs)
            print("epoch:", epoch + 1, "average_loss:", average_loss)

            progress = (epoch + 1) / epochs
            self.l_rate = end_lr + (start_lr - end_lr) * (1.0 - progress**power)