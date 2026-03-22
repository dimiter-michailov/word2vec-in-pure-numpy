import numpy as np

class Reporting:
    def __init__(self, vocab, V_size, input_hidden_matrix, results_path="results.md"):
        self.vocab = vocab
        self.V_size = V_size
        self.input_hidden_matrix = input_hidden_matrix
        self.results_path = results_path

        self.id_to_word = [None] * V_size
        for word, idx in self.vocab.items():
            self.id_to_word[idx] = word
        
        # normalize embeddings for easier cosine similarity
        norms = np.linalg.norm(self.input_hidden_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.normalized_embeddings = self.input_hidden_matrix / norms

    @staticmethod
    def load_word2vec_txt(path):
        """
        Loads embeddings from a standard word2vec text-format file.

        Supported format:
        first line: <vocab_size> <embedding_dim>
        next lines: word val1 val2 ... valN
        """
        words = []
        vectors = []

        with open(path, "r") as file:
            first_line = file.readline()
            if not first_line:
                raise ValueError(f"Empty embeddings file: {path}")

            first_parts = first_line.strip().split()

            has_header = (
                len(first_parts) == 2
                and first_parts[0].isdigit()
                and first_parts[1].isdigit()
            )

            expected_vocab_size = None
            expected_dim = None

            if has_header:
                expected_vocab_size = int(first_parts[0])
                expected_dim = int(first_parts[1])
            else:
                if len(first_parts) < 2:
                    raise ValueError(f"Invalid first line in embeddings file: {path}")

                word = first_parts[0]
                vector = [float(value) for value in first_parts[1:]]
                expected_dim = len(vector)

                words.append(word)
                vectors.append(vector)

            for line_number, line in enumerate(file, start=2):
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid embedding row at line {line_number} in {path}"
                    )

                word = parts[0]
                vector = [float(value) for value in parts[1:]]

                if len(vector) != expected_dim:
                    raise ValueError(
                        f"Inconsistent embedding size at line {line_number} in {path}. "
                        f"Expected {expected_dim}, got {len(vector)}."
                    )

                words.append(word)
                vectors.append(vector)

        if expected_vocab_size is not None and len(words) != expected_vocab_size:
            raise ValueError(
                f"Header says vocab size {expected_vocab_size}, "
                f"but loaded {len(words)} rows from {path}."
            )

        if not words:
            raise ValueError(f"No embeddings found in file: {path}")

        vocab = {word: idx for idx, word in enumerate(words)}
        input_hidden_matrix = np.array(vectors, dtype=np.float64)

        return vocab, input_hidden_matrix

    def save_word2vec_txt(self, path):
        """
        Saves embeddings in standard word2vec text format.

        Format:
        first line: <vocab_size> <embedding_dim>
        next lines: word val1 val2 ... valN
        """
        embedding_dim = self.input_hidden_matrix.shape[1]

        with open(path, "w", encoding="utf-8") as file:
            file.write(f"{self.V_size} {embedding_dim}\n")

            for idx in range(self.V_size):
                word = self.id_to_word[idx]
                vector = self.input_hidden_matrix[idx]
                vector_text = " ".join(f"{float(value):.10f}" for value in vector)
                file.write(f"{word} {vector_text}\n")

    def nearest_neighbors(self, word, top_k=5):
        """
        Returns the top_k nearest words to the given word
        using cosine similarity.
        """
        if word not in self.vocab:
            return []

        word_id = self.vocab[word]
        word_embedding = self.normalized_embeddings[word_id]

        scores = []

        for idx in range(self.V_size):
            if idx == word_id:
                continue
            
            # cosine similarity is just dot product since embeddings are normalized
            similarity = float(np.dot(word_embedding, self.normalized_embeddings[idx]))
            other_word = self.id_to_word[idx]
            if len(other_word) == 1:
                continue
            scores.append((other_word, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
        
    def print_example_neighbors(self):
        """
        Prints the nearest neighbors for a few example words.
        """
        example_words = []
        for i in range(self.V_size):
            word = self.id_to_word[i]
            if len(word) > 1:
                example_words.append(word)
            if len(example_words) == min(10, self.V_size):
                break
        
        # 5 nearest neighbors for each example word
        for word in example_words:
            neighbors = self.nearest_neighbors(word, top_k=5)

            print("\nWord:", word)
            for neighbor_word, score in neighbors:
                print(" ", neighbor_word, "->", round(float(score), 4))

    def report_neighbors(self, num_words=10, top_k=5):
        """
        Prepares a markdown-formatted string with the nearest neighbors for a few example words for the report.
        """
        lines = []

        example_words = []
        for i in range(self.V_size):
            word = self.id_to_word[i]
            if len(word) > 1:
                example_words.append(word)
            if len(example_words) == min(num_words, self.V_size):
                break

        for word in example_words:
            neighbors = self.nearest_neighbors(word, top_k=top_k)

            lines.append(f"### {word}")
            for neighbor_word, score in neighbors:
                lines.append(f"- {neighbor_word} ({score:.4f})")
            lines.append("")

        return "\n".join(lines)

    def analogy(self, a, b, c, top_k=5):
        """
        Solves for "?": a is to b like c is to ?
        using vec(b) - vec(a) + vec(c)
        """
        a_id = self.vocab[a]
        b_id = self.vocab[b]
        c_id = self.vocab[c]

        # compute the analogy query vector
        query = self.input_hidden_matrix[b_id] - self.input_hidden_matrix[a_id] + self.input_hidden_matrix[c_id]
        
        # normalize the query vector
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return []

        query = query / query_norm
        
        # cosine similarity is just dot product since embeddings are normalized
        scores = self.normalized_embeddings @ query

        # exclude a, b, c from the candidates
        scores[a_id] = -np.inf
        scores[b_id] = -np.inf
        scores[c_id] = -np.inf

        top_k = max(1, min(top_k, self.V_size - 3))

        # get the top_k highest scoring words
        top_ids = np.argpartition(scores, -top_k)[-top_k:]
        top_ids = top_ids[np.argsort(scores[top_ids])[::-1]]

        results = []
        for idx in top_ids:
            results.append((self.id_to_word[idx], float(scores[idx])))

        return results
    
    def read_analogy_file(self, path):
        """
        Reads files that define word analogy test.
        Format:
        : category_name
        a b c d
        a b c d
        """
        categories = {}
        current_category = None

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().lower()

                if not line:
                    continue

                if line.startswith(":"):
                    current_category = line[1:].strip()
                    categories[current_category] = []
                    continue

                parts = line.split()
                if len(parts) != 4 or current_category is None:
                    continue

                a, b, c, d = parts
                categories[current_category].append((a, b, c, d))

        return categories
    
    def evaluate_analogies_with_summary(self, categories, questions_file, top_k=1):
        """
        Each category name is a list of analogy questions.
        Each question is a tuple:
        (a, b, c, d)
        This method provides a summary of the results per category and analogy type.
        Returns both the markdown report text and a structured summary dictionary.
        """
        lines = []

        semantic_categories = {"capital-common-countries", "capital-world", "currency", "city-in-state", "family"}

        semantic_questions_in_file = 0
        semantic_questions_asked = 0
        semantic_correct = 0
        semantic_skipped = 0

        syntactic_questions_in_file = 0
        syntactic_questions_asked = 0
        syntactic_correct = 0
        syntactic_skipped = 0

        lines.append(f"### {questions_file}")
        lines.append("")

        # keep track of correct/skipped counts for each question 
        for category_name, questions in categories.items():
            questions_in_file = len(questions)
            questions_asked = 0
            correct = 0
            skipped = 0

            for a, b, c, d in questions:
                if any(word not in self.vocab for word in (a, b, c, d)):
                    skipped += 1
                    continue

                preds = self.analogy(a, b, c, top_k=top_k)
                questions_asked += 1

                if top_k == 1:
                    if preds and preds[0][0] == d:
                        correct += 1
                else:
                    if any(word == d for word, _ in preds):
                        correct += 1

            accuracy = correct / questions_asked if questions_asked > 0 else 0.0

            lines.append(f"#### {category_name}")
            lines.append(f"- questions in category: **{questions_in_file}**")
            lines.append(f"- questions asked: **{questions_asked}**")
            lines.append(f"- skipped: **{skipped}**")
            lines.append(f"- correct: **{correct}**")
            lines.append(f"- accuracy: **{accuracy:.4f}**")
            lines.append("")

            if category_name in semantic_categories:
                semantic_questions_in_file += questions_in_file
                semantic_questions_asked += questions_asked
                semantic_correct += correct
                semantic_skipped += skipped
            else:
                syntactic_questions_in_file += questions_in_file
                syntactic_questions_asked += questions_asked
                syntactic_correct += correct
                syntactic_skipped += skipped

        semantic_accuracy = (
            semantic_correct / semantic_questions_asked
            if semantic_questions_asked > 0 else 0.0
        )

        syntactic_accuracy = (
            syntactic_correct / syntactic_questions_asked
            if syntactic_questions_asked > 0 else 0.0
        )

        overall_questions_in_file = semantic_questions_in_file + syntactic_questions_in_file
        overall_questions_asked = semantic_questions_asked + syntactic_questions_asked
        overall_correct = semantic_correct + syntactic_correct
        overall_skipped = semantic_skipped + syntactic_skipped
        overall_accuracy = (
            overall_correct / overall_questions_asked
            if overall_questions_asked > 0 else 0.0
        )

        lines.append("### Summary by analogy type")
        lines.append("")

        lines.append("**Semantic**")
        lines.append(f"- questions in file: **{semantic_questions_in_file}**")
        lines.append(f"- questions asked: **{semantic_questions_asked}**")
        lines.append(f"- skipped: **{semantic_skipped}**")
        lines.append(f"- correct: **{semantic_correct}**")
        lines.append(f"- accuracy: **{semantic_accuracy:.4f}**")
        lines.append("")

        lines.append("**Syntactic**")
        lines.append(f"- questions in file: **{syntactic_questions_in_file}**")
        lines.append(f"- questions asked: **{syntactic_questions_asked}**")
        lines.append(f"- skipped: **{syntactic_skipped}**")
        lines.append(f"- correct: **{syntactic_correct}**")
        lines.append(f"- accuracy: **{syntactic_accuracy:.4f}**")
        lines.append("")

        lines.append("**Overall**")
        lines.append(f"- questions in file: **{overall_questions_in_file}**")
        lines.append(f"- questions asked: **{overall_questions_asked}**")
        lines.append(f"- skipped: **{overall_skipped}**")
        lines.append(f"- correct: **{overall_correct}**")
        lines.append(f"- accuracy: **{overall_accuracy:.4f}**")
        lines.append("")

        summary = {
            "questions_file": questions_file,
            "overall": {
                "questions_in_file": overall_questions_in_file,
                "questions_asked": overall_questions_asked,
                "skipped": overall_skipped,
                "correct": overall_correct,
                "accuracy": overall_accuracy,
            },
            "semantic": {
                "questions_in_file": semantic_questions_in_file,
                "questions_asked": semantic_questions_asked,
                "skipped": semantic_skipped,
                "correct": semantic_correct,
                "accuracy": semantic_accuracy,
            },
            "syntactic": {
                "questions_in_file": syntactic_questions_in_file,
                "questions_asked": syntactic_questions_asked,
                "skipped": syntactic_skipped,
                "correct": syntactic_correct,
                "accuracy": syntactic_accuracy,
            },
        }

        return "\n".join(lines), summary

    def evaluate_analogies(self, categories, questions_file, top_k=1):
        """
        Backward-compatible wrapper that returns only the markdown text.
        """
        report_text, _ = self.evaluate_analogies_with_summary(
            categories, questions_file, top_k=top_k
        )
        return report_text

    def populate_results_template(self, dataset_name, token_count, parameters_text, neighbors_text, custom_results_text, google_results_text):
        """
        Fills the results template with the passed information and saves it to results.md
        """
        with open("results_template.md", "r", encoding="utf-8") as file:
            template = file.read()

        filled = (
            template
            .replace("{{dataset_name}}", dataset_name)
            .replace("{{vocab_size}}", str(self.V_size))
            .replace("{{token_count}}", str(token_count))
            .replace("{{parameters_text}}", parameters_text)
            .replace("{{neighbors_text}}", neighbors_text)
            .replace("{{custom_results_text}}", custom_results_text)
            .replace("{{google_results_text}}", google_results_text)
        )

        with open(self.results_path, "w", encoding="utf-8") as file:
            file.write(filled)