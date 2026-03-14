import numpy as np
from cbow.cbow import CBOW
from cbow.hierarchical_cbow import HierarchicalCBOW
from cbow.negative_cbow import NegativeCBOW
from skipgram.hierarchical_skipgram import HierarchicalSkipgram
from skipgram.negative_skipgram import NegativeSkipgram 
from skipgram.skipgram import Skipgram
from text_processing import TextProcessing
from reporting import Reporting

def main():
    file_names = input("Choose file name(s) for training, comma-separated: ").split(",")
    file_names = [f.strip() for f in file_names]
    model_type = input("Choose between cbow or skipgram implementation. Spell out choice: ").strip().lower()
    further_type = input("Choose the model variant (spell out \"basic\", \"hs\" - uses hierarchical softmax, or \"ns\" - uses negative sampling): ").strip().lower()
    context_size = int(input("Choose the total context size (includes all words other than the target). Type int: ").strip())
    embedding_dim = int(input("Choose the word embedding size. Type int: ").strip())
    epochs = int(input("Choose number of training epochs. Type int: ").strip())

    text_processor = TextProcessing(file_names)
    token_ids = text_processor.token_ids
    word_frequency = text_processor.word_frequency
    print("number of tokens:", len(text_processor.tokens))
    print("vocab size:", text_processor.V_size)

    if model_type == "cbow":
        if further_type == "basic":
            model = CBOW(token_ids, text_processor.V_size, context_size, embedding_dim)
            model.train(epochs=epochs)
        elif further_type == "hs":
            model = HierarchicalCBOW(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)
            model.train(epochs=epochs)
        elif further_type == "ns":
            model = NegativeCBOW(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)
            model.train(epochs=epochs)

        reporting = Reporting(text_processor.vocab, text_processor.V_size, model.input_hidden_matrix)
        reporting.print_example_neighbors()

    elif model_type == "skipgram":
        if further_type == "basic":
            model = Skipgram(token_ids, text_processor.V_size, context_size, embedding_dim)
            model.train(epochs=epochs)
        elif further_type == "hs":
            model = HierarchicalSkipgram(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)
            model.train(epochs=epochs)
        elif further_type == "ns":
            model = NegativeSkipgram(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)
            model.train(epochs=epochs)

        reporting = Reporting(text_processor.vocab, text_processor.V_size, model.input_hidden_matrix)
        reporting.print_example_neighbors()

if __name__ == "__main__":
    main()