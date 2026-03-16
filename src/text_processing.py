from pathlib import Path
import re
import numpy as np

class TextProcessing:
    def __init__(self, file_names):
        self.word_frequency = []
        self.text = None
        self.tokens = None

        self.file_names = self.read_dataset(file_names)
        self.vocab = self.build_vocab(self.file_names)
        self.V_size = len(self.vocab)
        self.token_ids = self.encode_token_ids(self.file_names, self.vocab)

    def read_dataset(self, file_names):
        """
        file_names can be:
            "tiny.txt": str
            or 
            ["tiny.txt", "another.txt"]: list
        Reads from the 'datasets/' folder
        Returns the text as one big string
        """
        if isinstance(file_names, str):
            file_names = [file_names]

        cleaned_file_names = []
        
        for file_name in file_names:
            file_name = file_name.strip()
            if not file_name.endswith(".txt"):
                file_name += ".txt"

            path = Path("datasets") / file_name
            if not path.exists():
                raise FileNotFoundError(f"Could not find file: {path}")

            cleaned_file_names.append(file_name)

        return cleaned_file_names

    def tokenize(self, text):
        """
        Splits the full provided text
        into a list of all word occurences.
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)

        tokens = text.split()
        return tokens

    def build_vocab(self, file_names):
        """
        Creates a dictionary for all the words:
        {"apple": 0, "orange": 1}
        
        And also records their frequency in word_frequency[]
        """
        vocab = {}
        self.word_frequency = []
        self.token_count = 0

        for file_name in file_names:
            path = Path("datasets") / file_name

            with open(path, "r") as file:
                for line in file:
                    tokens = self.tokenize(line)

                    for token in tokens:
                        self.token_count += 1

                        if token not in vocab:
                            vocab[token] = len(vocab)
                            self.word_frequency.append(1)
                        else:
                            self.word_frequency[vocab[token]] += 1

        return vocab
    
    def encode_token_ids(self, file_names, vocab):
        """
        Replaces words (strings) in the dataset with int ids.
        Each word is mapped to its id in the vocab dictionary.

        The token ids are stored in a file on disk.
        """
        token_ids_path = Path("token_ids.dat")

        token_ids = np.memmap(
            token_ids_path,
            dtype=np.int32,
            mode="w+",
            shape=(self.token_count,)
        )

        index = 0

        for file_name in file_names:
            path = Path("datasets") / file_name

            with open(path, "r") as file:
                for line in file:
                    tokens = self.tokenize(line)

                    for token in tokens:
                        token_id = vocab[token]
                        token_ids[index] = token_id
                        index += 1

        token_ids.flush()
        return token_ids