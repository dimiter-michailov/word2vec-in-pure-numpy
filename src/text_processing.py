from pathlib import Path
import re
import numpy as np

class TextProcessing:
    def __init__(self, file_names):
        self.word_frequency = []
        self.text = self.read_dataset(file_names)
        self.tokens = self.tokenize(self.text)
        self.vocab = self.build_vocab(self.tokens)
        self.V_size = len(self.vocab)
        self.token_ids = self.encode_token_ids(self.tokens, self.vocab)

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

        full_text = ""
        
        for file_name in file_names:
            file_name = file_name.strip()
            if not file_name.endswith(".txt"):
                file_name += ".txt"

            path = Path("datasets") / file_name
            with open(path, "r") as file:
                full_text += file.read() + " "

        return full_text

    def tokenize(self, text):
        """
        Splits the full provided text
        into a list of all word occurences.
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)

        tokens = text.split()
        return tokens

    def build_vocab(self, tokens):
        """
        Creates a dictionary for all the words:
        {"apple": 0, "orange": 1}
        And also records their frequency in word_frequency[]
        """
        vocab = {}

        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
                self.word_frequency.append(1)
            else:
                self.word_frequency[vocab[token]] += 1

        return vocab
    
    def encode_token_ids(self, tokens, vocab):
        """
        Replaces words (strings) in the tokens list with int ids.
        Each word is mapped to its id in the vocab dictionary.
        """
        token_ids = []

        for token in tokens:
            token_id = vocab[token]
            token_ids.append(token_id)

        return token_ids