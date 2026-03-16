import os
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
    model_choice = input("Type number: ").strip()
    model_type = "cbow" if model_choice == "1" else "skipgram"

    print("\nChoose model variant:")
    print("1. standard (no optimizations)")
    print("2. hierarchical softmax")
    print("3. negative sampling (hardcoded 20 negative samples per positive)")
    variant_choice = input("Type number: ").strip()
    if variant_choice == "1":
        further_type = "standard"
    elif variant_choice == "2":
        further_type = "hierarchical softmax"
    else:
        further_type = "negative sampling"

    while True:
        context_size_input = input("\nChoose the context size (includes all words other than the target/center). Type even int: ").strip()
        if context_size_input.isdigit() and int(context_size_input) % 2 == 0:
            context_size = int(context_size_input)
            break
        else:
            print("\nInvalid input. Only even numbers accepted.")
    embedding_dim = int(input("\nChoose the word embedding size. Type int: ").strip())
    epochs = int(input("\nChoose number of training epochs. Type int: ").strip())

    # analogy evaluation performs vector arithmetics to solve analogy questions of the form "a is to b as c is to ?" 
    # from the learned word embeddings. 
    # It can take a long time to run on larger vocabularies. 
    print("\nPerform analogy evaluation?")
    print("1. Yes")
    print("2. No")
    analogy_choice = input("Type number: ").strip()
    perform_analogy_evaluation = analogy_choice == "1"

    return file_names, model_type, further_type, context_size, embedding_dim, epochs, perform_analogy_evaluation

def main():
    # get user choices for training and evaluation
    file_names, model_type, further_type, context_size, embedding_dim, epochs, perform_analogy_evaluation = get_user_choices()

    # process the text
    text_processor = TextProcessing(file_names)
    token_ids = text_processor.token_ids
    word_frequency = text_processor.word_frequency
    print("\nnumber of tokens:", text_processor.token_count)
    print("vocab size:", text_processor.V_size)

    # main model selection logic
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

    # reporting generates the results and saves them in results.md after training
    reporting = Reporting(text_processor.vocab, text_processor.V_size, model.input_hidden_matrix)
    reporting.print_example_neighbors()
    neighbors_text = reporting.report_neighbors()

    custom_results_text = "Not run in this execution."
    google_results_text = "Not run in this execution."

    if perform_analogy_evaluation:
        custom_categories = reporting.read_analogy_file("src/analogy_questions/custom_analogies.txt")
        custom_results_text = reporting.evaluate_analogies(custom_categories, questions_file="custom_analogies.txt", top_k=1 )
        google_categories = reporting.read_analogy_file("src/analogy_questions/google_analogies.txt")
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