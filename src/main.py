import numpy as np
from pathlib import Path
from cbow import CBOW
from skipgram import Skipgram

def read_dataset(file_names):
    """
    file_names can be:
        "tiny.txt": str
        or 
        ["tiny.txt", "another.txt"]: list
    Reads from the datasets/ folder
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

def print_example_neighbors(model, example_words):
    for word in example_words:
        neighbors = model.nearest_neighbors(word, top_k=5)

        print("\nWord:", word)
        if len(neighbors) == 0:
            print("  not found in vocabulary")
        else:
            for neighbor_word, score in neighbors:
                print(" ", neighbor_word, "->", round(float(score), 4))

def main():
    file_names = input("Choose file name(s) for training, comma-separated: ").split(",")
    file_names = [f.strip() for f in file_names]
    model_type = input("Choose between cbow or skipgram implementation. Spell out choice: ").strip().lower()
    context_size = int(input("Choose the total context size (includes all words other than the target). Type int: ").strip())
    embedding_dim = int(input("Choose the word-embedding size. Type int: ").strip())
    epochs = int(input("Choose number of training epochs. Type int: ").strip())

    text = read_dataset(file_names)

    if model_type == "cbow":
        model = CBOW(text, context_size, embedding_dim)
        model.train(epochs=epochs)

        example_words = []
        for i in range(min(10, model.V_size)):
            example_words.append(model.id_to_word[i])

        print_example_neighbors(model, example_words)

    elif model_type == "skipgram":
        model = Skipgram(text, context_size, embedding_dim)
        model.train(epochs=epochs)

if __name__ == "__main__":
    main()