import time
import re
import numpy as np

class CBOW:
    def __init__(self, text, context_size, embedding_dim, l_rate=0.01):
        self.l_rate = l_rate
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        self.tokens = self.tokenize(text)
        self.vocab = self.build_vocab(self.tokens)
        self.V_size = len(self.vocab)
        self.id_to_word = [None] * self.V_size
        for word, idx in self.vocab.items():
            self.id_to_word[idx] = word

        self.token_ids = self.encode_ids_in_tokens(self.tokens, self.vocab)
        self.pairs = self.make_cbow_training_pairs(self.token_ids, self.context_size)

        # input_hidden_matrix: (V_size, embedding_dim)
        self.input_hidden_matrix = ( np.random.randn(self.V_size, self.embedding_dim).astype(np.float32) * 0.01 )
        # hidden_output_matrix: (embedding_dim, V_size)
        self.hidden_output_matrix = ( np.random.randn(self.embedding_dim, self.V_size).astype(np.float32) * 0.01 )
        print("number of tokens:", len(self.tokens))
        print("vocab size:", self.V_size)
        print("number of training pairs:", len(self.pairs))

    def softmax(self, output_vector):
        """
        Takes a vector of pre-softmax scores
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
        Computes the loss from the pre-softmax scores.
        pre_softmax : vector of raw scores u_j
        target      : true target word id
        returns     : scalar loss E
        """
        max_value = np.max(pre_softmax)
        shifted_output = pre_softmax - max_value

        target_score = shifted_output[target]
        sum_exp = np.sum(np.exp(shifted_output))

        E = -target_score + np.log(sum_exp)
        return E
    
    def feedforward(self, context, target):
        # to keep vector sum of all the words used in the context
        context_sum = np.zeros(self.V_size, dtype=np.float32)

        # sum the one-hot vectors of all context words
        for word in context:
            oh = self.one_hot(word, self.V_size)
            context_sum = context_sum + oh
        
        hidden = context_sum @ self.input_hidden_matrix
        pre_softmax = hidden @ self.hidden_output_matrix
        post_softmax = self.softmax(pre_softmax)

        E = self.loss_function(pre_softmax, target)

        return context_sum, hidden, post_softmax, E
    

    def train(self, epochs=1):
        for epoch in range(epochs):
            print("\nstarting epoch", epoch + 1)
            total_loss = 0.0
            epoch_start = time.time()
        
            for i, p in enumerate(self.pairs):
                context = p[0]
                target = p[1]

                context_sum, hidden, post_softmax, E = self.feedforward(context, target)
                self.backpropagate(post_softmax, target, hidden, context_sum)

                total_loss = total_loss + E
                if i % 50000 == 0:
                    elapsed = time.time() - epoch_start
                    print("epoch", epoch + 1, "pair", i, "out of", len(self.pairs), "time:", round(elapsed, 2), "sec")
    
            average_loss = total_loss / len(self.pairs)
            print("epoch:", epoch + 1, "average_loss:", average_loss)


    def backpropagate(self, post_softmax, target, hidden, context_sum):
        """
        Applies one backpropagation step to the input_hidden_matrix and hidden_output_matrix per given training example
        """
        # dE/du_j = y_j - t_j
        target_vector = np.zeros(len(post_softmax), dtype=np.float32)
        target_vector[target] = 1.0
        dE_du = post_softmax - target_vector

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
        dE_d_input_hidden_matrix = np.outer(context_sum, dE_dh)
        
        # w_ki = w_ki - learning_rate * dE/dw_ki
        self.input_hidden_matrix -= self.l_rate * dE_d_input_hidden_matrix


    def tokenize(self, text):
        """
        Splits the full text provided for learning
        into a list of all the word occurences
        """
        text = text.lower()
        text = text.replace("\n", " ")

        text = re.sub(r"[^a-z\s]", " ", text)

        tokens = text.split()
        return tokens
    

    def build_vocab(self, tokens):
        """
        Creates a dictionary for all the words:
        {"apple": 0, "orange": 1}
        """
        vocab = {}

        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab
    

    def encode_ids_in_tokens(self, tokens, vocab):
        """
        Replaces words (strings) in the tokens (list)
        with int ids as defined in the vocab
        """
        token_ids = []

        for token in tokens:
            token_id = vocab[token]
            token_ids.append(token_id)

        return token_ids
    

    def one_hot(self, word_id, vocab_size):
        """
        Builds a one-hot encoded vector for a given word id
        """
        vector = np.zeros(vocab_size, dtype=np.float32)
        vector[word_id] = 1.0

        return vector
    
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

        return pairs


    def cosine_similarity(self, a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)

            if a_norm == 0.0 or b_norm == 0.0:
                return 0.0

            return np.dot(a, b) / (a_norm * b_norm)

    def nearest_neighbors(self, word, top_k=5):
        """
        Returns the top_k nearest words to the given word
        using cosine similarity.
        """
        if word not in self.vocab:
            return []

        word_id = self.vocab[word]
        word_embedding = self.input_hidden_matrix[word_id]

        scores = []

        for idx in range(self.V_size):
            if idx == word_id:
                continue

            other_vector = self.input_hidden_matrix[idx]
            similarity = self.cosine_similarity(word_embedding, other_vector)

            other_word = self.id_to_word[idx]
            scores.append((other_word, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
