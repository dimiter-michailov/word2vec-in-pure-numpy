import os
from cbow.cbow import CBOW
from cbow.hierarchical_cbow import HierarchicalCBOW
from cbow.negative_cbow import NegativeCBOW
from skipgram.hierarchical_skipgram import HierarchicalSkipgram
from skipgram.negative_skipgram import NegativeSkipgram 
from skipgram.skipgram import Skipgram
from text_processing import TextProcessing
from reporting import Reporting
from run_storage import (EMBEDDINGS_DIR, build_run_metadata, build_scoreboard_row, choose_saved_embedding_file, 
                         ensure_embeddings_dir, infer_run_metadata_from_embedding_file, upsert_scoreboard_row,)

def ask_perform_analogy_evaluation():
    """
    Interactively asks whether analogy evaluation should be performed.
    """
    # analogy evaluation performs vector arithmetics to solve analogy questions of the form "a is to b as c is to ?" from the learned word embeddings.
    # It can take a long time to run on larger vocabularies.
    print("\nPerform analogy evaluation?")
    print("1. Yes")
    print("2. No")

    while True:
        analogy_choice = input("Type number: ").strip()
        if analogy_choice == "1":
            return True
        elif analogy_choice == "2":
            return False
        else:
            print("\nInvalid choice. Please try again.")

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

    perform_analogy_evaluation = ask_perform_analogy_evaluation()

    return file_names, model_type, further_type, context_size, embedding_dim, epochs, perform_analogy_evaluation

def create_model(model_type, further_type, token_ids, word_frequency, vocab_size, context_size, embedding_dim):
    """
    Main model selection logic.
    """
    if model_type == "cbow":
        if further_type == "standard":
            return CBOW(token_ids, vocab_size, context_size, embedding_dim)
        elif further_type == "hierarchical softmax":
            return HierarchicalCBOW(token_ids, word_frequency, vocab_size, context_size, embedding_dim)
        elif further_type == "negative sampling":
            return NegativeCBOW(token_ids, word_frequency, vocab_size, context_size, embedding_dim)

    elif model_type == "skipgram":
        if further_type == "standard":
            return Skipgram(token_ids, vocab_size, context_size, embedding_dim)
        elif further_type == "hierarchical softmax":
            return HierarchicalSkipgram(token_ids, word_frequency, vocab_size, context_size, embedding_dim)
        elif further_type == "negative sampling":
            return NegativeSkipgram(token_ids, word_frequency, vocab_size, context_size, embedding_dim)

    raise ValueError("Unsupported model configuration.")

def main():
    ensure_embeddings_dir()

    # initial user prompt to select workflow
    print("Select workflow:")
    print("1. Train a new model configuration (standard workflow)")
    print("2. Generate report from saved embeddings (.txt)")
    while True:
        run_choice = input("Type number: ").strip()
        if run_choice in {"1", "2"}:
            break
        print("\nInvalid choice. Please try again.")

    custom_summary = None
    google_summary = None
    custom_results_text = "Not run in this execution."
    google_results_text = "Not run in this execution."

    # only in train workflow: get user choices for training and evaluation
    if run_choice == "1":
        file_names, model_type, further_type, context_size, embedding_dim, epochs, perform_analogy_evaluation = get_user_choices()

        # process the text
        text_processor = TextProcessing(file_names)
        token_ids = text_processor.token_ids
        word_frequency = text_processor.word_frequency
        print("\nnumber of tokens:", text_processor.token_count)
        print("vocab size:", text_processor.V_size)

        # main model selection logic
        model = create_model(
            model_type=model_type,
            further_type=further_type,
            token_ids=token_ids,
            word_frequency=word_frequency,
            vocab_size=text_processor.V_size,
            context_size=context_size,
            embedding_dim=embedding_dim,
        )

        model.train(epochs=epochs)
        input_hidden_matrix = model.input_hidden_matrix

        # reporting generates the results and saves them in results.md
        reporting = Reporting(text_processor.vocab, text_processor.V_size, input_hidden_matrix)

        # save learned embeddings in standard word2vec text format
        run_id, run_summary, embedding_path = build_run_metadata(
            file_names=file_names,
            model_type=model_type,
            further_type=further_type,
            context_size=context_size,
            embedding_dim=embedding_dim,
            epochs=epochs,
        )
        reporting.save_word2vec_txt(embedding_path)

        print(f"\nSaved learned embeddings to {embedding_path}")

        dataset_name = ", ".join(file_names)
        token_count = text_processor.token_count
        parameters_text = (
            f"- run id: {run_id}\n"
            f"- model: {model_type}\n"
            f"- variant: {further_type}\n"
            f"- embedding size: {embedding_dim}\n"
            f"- context size: {context_size}\n"
            f"- epochs: {epochs}\n"
            f"- embedding file: {os.path.basename(embedding_path)}"
        )

    # load the saved embeddings and parameters from a selected .txt embedding file
    else:
        embedding_path = choose_saved_embedding_file()
        vocab, input_hidden_matrix = Reporting.load_word2vec_txt(embedding_path)

        run_id, run_summary = infer_run_metadata_from_embedding_file(embedding_path)

        print(f"\nLoaded saved embeddings from {embedding_path}")
        print("vocab size:", len(vocab))
        print("embedding size:", input_hidden_matrix.shape[1])

        perform_analogy_evaluation = ask_perform_analogy_evaluation()

        reporting = Reporting(vocab, len(vocab), input_hidden_matrix)

        dataset_name = f"loaded from {os.path.basename(embedding_path)}"
        token_count = "N/A"
        parameters_text = (
            f"- run id: {run_id}\n"
            f"- embedding file: {os.path.basename(embedding_path)}\n"
            f"- embedding size: {input_hidden_matrix.shape[1]}"
        )

    reporting.print_example_neighbors()
    neighbors_text = reporting.report_neighbors()

    if perform_analogy_evaluation:
        custom_categories = reporting.read_analogy_file("analogy_questions/custom_analogies.txt")
        custom_results_text, custom_summary = reporting.evaluate_analogies_with_summary(
            custom_categories,
            questions_file="custom_analogies.txt",
            top_k=1,
        )

        google_categories = reporting.read_analogy_file("analogy_questions/google_analogies.txt")
        google_results_text, google_summary = reporting.evaluate_analogies_with_summary(
            google_categories,
            questions_file="google_analogies.txt",
            top_k=1,
        )

    reporting.populate_results_template(
        dataset_name=dataset_name,
        token_count=token_count,
        parameters_text=parameters_text,
        neighbors_text=neighbors_text,
        custom_results_text=custom_results_text,
        google_results_text=google_results_text
    )

    # update scoreboard only for a fresh model training run
    if run_choice == "1":
        scoreboard_row = build_scoreboard_row(
            run_id=run_id,
            run_summary=run_summary,
            embedding_file_name=os.path.basename(embedding_path),
            custom_summary=custom_summary,
            google_summary=google_summary,
        )
        upsert_scoreboard_row(scoreboard_row)
        print("\nscoreboard.csv has been updated.")

    print("\nThe full results produced by the model are available in results.md")
    print(f"Embeddings are stored in {EMBEDDINGS_DIR}/")

if __name__ == "__main__":
    main()