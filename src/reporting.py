import numpy as np

class Reporting:
    def __init__(self, vocab, V_size, input_hidden_matrix):
        self.vocab = vocab
        self.V_size = V_size
        self.input_hidden_matrix = input_hidden_matrix

        self.id_to_word = [None] * V_size
        for word, idx in self.vocab.items():
            self.id_to_word[idx] = word

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
        
    def print_example_neighbors(self):
        example_words = []
        for i in range(min(10, self.V_size)):
            example_words.append(self.id_to_word[i])

        for word in example_words:
            neighbors = self.nearest_neighbors(word, top_k=5)

            print("\nWord:", word)
            for neighbor_word, score in neighbors:
                print(" ", neighbor_word, "->", round(float(score), 4))