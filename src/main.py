import os
import numpy as np
from markdown_pdf import MarkdownPdf, Section
from cbow.cbow import CBOW
from cbow.hierarchical_cbow import HierarchicalCBOW
from cbow.negative_cbow import NegativeCBOW
from skipgram.hierarchical_skipgram import HierarchicalSkipgram
from skipgram.negative_skipgram import NegativeSkipgram 
from skipgram.skipgram import Skipgram
from text_processing import TextProcessing
from reporting import Reporting

def get_user_choices():
    """
    Interactively gets the user's choices for training and evaluation from the command line.
    """
    available_files = sorted([f for f in os.listdir("datasets") if f.endswith(".txt")])

    print("\nAvailable training texts:")
    for i, file_name in enumerate(available_files, start=1):
        print(f"{i}. {file_name}")
    while True:
        try:
            selected_numbers = input("\nChoose file number(s) for training, comma-separated: ").split(",")
            selected_numbers = [int(num.strip()) for num in selected_numbers]
            if not selected_numbers or any(num < 1 or num > len(available_files) for num in selected_numbers):
                raise ValueError
            file_names = [available_files[num - 1] for num in selected_numbers]
            break
        except ValueError:
            print("\nInvalid selection. Please try again.")

    print("\nChoose model type:")
    print("1. cbow")
    print("2. skipgram")
    while True:
        model_choice = input("Type number: ").strip()
        if model_choice == "1":
            model_type = "cbow"
            break
        elif model_choice == "2":
            model_type = "skipgram"
            break
        else:
            print("\nInvalid choice. Please try again.")

    print("\nChoose model variant:")
    print("1. standard (no optimizations)")
    print("2. hierarchical softmax")
    print("3. negative sampling (hardcoded 20 negative samples per positive)")
    while True:
        variant_choice = input("Type number: ").strip()
        if variant_choice == "1":
            further_type = "standard"
            break
        elif variant_choice == "2":
            further_type = "hierarchical softmax"
            break
        elif variant_choice == "3":
            further_type = "negative sampling"
            break
        else:
            print("\nInvalid choice. Please try again.")

    while True:
        context_size_input = input("\nChoose the context size (includes all words other than the target/center). Type even int: ").strip()
        if context_size_input.isdigit() and int(context_size_input) % 2 == 0:
            context_size = int(context_size_input)
            break
        else:
            print("\nInvalid input. Only even numbers accepted.")
    embedding_dim = int(input("\nChoose the word embedding size. Type int: ").strip())
    epochs = int(input("\nChoose number of training epochs. Type int: ").strip())

    # analogy evaluation performs vector arithmetics to solve analogy questions of the form "a is to b as c is to ?" from the learned word embeddings. 
    # It can take a long time to run on larger vocabularies. 
    print("\nPerform analogy evaluation?")
    print("1. Yes")
    print("2. No")
    analogy_choice = input("Type number: ").strip()
    perform_analogy_evaluation = analogy_choice == "1"

    return file_names, model_type, further_type, context_size, embedding_dim, epochs, perform_analogy_evaluation

def main():
    # single saved file from last run
    saved_run_file = "last_run_embeddings.npz"

    # initial user prompt to select workflow
    print("Select workflow:")
    print("1. Train a new model configuration (standard workflow)")
    print("2. Generate report from saved embeddings")
    while True:
        run_choice = input("Type number: ").strip()
        if run_choice in {"1", "2"}:
            break
        print("\nInvalid choice. Please try again.")
    
    # only in train workflow: get user choices for training and evaluation
    if run_choice == "1":
        file_names, model_type, further_type, context_size, embedding_dim, epochs, perform_analogy_evaluation = get_user_choices()
    # load the saved embeddings and parameters from the last training run
    else:
        if not os.path.exists(saved_run_file):
            raise FileNotFoundError(f"Saved embeddings file not found: {saved_run_file}")

        saved_run = np.load(saved_run_file, allow_pickle=True)

        file_names = saved_run["file_names"].tolist()
        model_type = str(saved_run["model_type"].item())
        further_type = str(saved_run["further_type"].item())
        context_size = int(saved_run["context_size"].item())
        embedding_dim = int(saved_run["embedding_dim"].item())
        epochs = int(saved_run["epochs"].item())
        input_hidden_matrix = saved_run["input_hidden_matrix"]

        print(f"\nLoaded saved embeddings from {saved_run_file}")
        print("\nPerform analogy evaluation?")
        print("1. Yes")
        print("2. No")
        analogy_choice = input("Type number: ").strip()
        perform_analogy_evaluation = analogy_choice == "1"

    # process the text
    text_processor = TextProcessing(file_names)
    token_ids = text_processor.token_ids
    word_frequency = text_processor.word_frequency
    print("\nnumber of tokens:", text_processor.token_count)
    print("vocab size:", text_processor.V_size)

    # main model selection logic
    if run_choice == "1":
        model = None
        if model_type == "cbow":
            if further_type == "standard":
                model = CBOW(token_ids, text_processor.V_size, context_size, embedding_dim)
            elif further_type == "hierarchical softmax":
                model = HierarchicalCBOW(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)
            elif further_type == "negative sampling":
                model = NegativeCBOW(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)

        elif model_type == "skipgram":
            if further_type == "standard":
                model = Skipgram(token_ids, text_processor.V_size, context_size, embedding_dim)
            elif further_type == "hierarchical softmax":
                model = HierarchicalSkipgram(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)
            elif further_type == "negative sampling":
                model = NegativeSkipgram(token_ids, word_frequency, text_processor.V_size, context_size, embedding_dim)

        model.train(epochs=epochs)
        input_hidden_matrix = model.input_hidden_matrix

        np.savez(
            saved_run_file,
            input_hidden_matrix=input_hidden_matrix,
            file_names=np.array(file_names, dtype=object),
            model_type=model_type,
            further_type=further_type,
            context_size=context_size,
            embedding_dim=embedding_dim,
            epochs=epochs,
        )

        print(f"\nSaved learned embeddings to {saved_run_file}")
    else:
        if input_hidden_matrix.shape[0] != text_processor.V_size:
            raise ValueError("\nCould not load saved embeddings.")

    # reporting generates the results and saves them in results.md
    reporting = Reporting(text_processor.vocab, text_processor.V_size, input_hidden_matrix)
    reporting.print_example_neighbors()
    neighbors_text = reporting.report_neighbors()

    custom_results_text = "Not run in this execution."
    google_results_text = "Not run in this execution."

    if perform_analogy_evaluation:
        custom_categories = reporting.read_analogy_file("analogy_questions/custom_analogies.txt")
        custom_results_text = reporting.evaluate_analogies(custom_categories, questions_file="custom_analogies.txt", top_k=1 )
        google_categories = reporting.read_analogy_file("analogy_questions/google_analogies.txt")
        google_results_text = reporting.evaluate_analogies(google_categories, questions_file="google_analogies.txt", top_k=1 )

    parameters_text = (
        f"- model: {model_type}\n"
        f"- variant: {further_type}\n"
        f"- embedding size: {embedding_dim}\n"
        f"- context size: {context_size}\n"
        f"- epochs: {epochs}"
    )

    dataset_name = ", ".join(file_names)
    reporting.populate_results_template(
        dataset_name=dataset_name,
        token_count=text_processor.token_count,
        parameters_text=parameters_text,
        neighbors_text=neighbors_text,
        custom_results_text=custom_results_text,
        google_results_text=google_results_text
    )

    # Generate also a .pdf from results.md
    pdf = MarkdownPdf()
    with open("results.md", "r") as f:
        pdf.add_section(Section(f.read()))
    pdf.save("results.pdf")

    print("\nThe full results produced by the model are available in results.md")
    print("results.pdf is also available for preview.")

if __name__ == "__main__":
    main()