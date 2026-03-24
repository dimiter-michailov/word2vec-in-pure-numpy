import csv
import os

EMBEDDINGS_DIR = "saved_embeddings"
SCOREBOARD_FILE = "scoreboard.csv"

SCOREBOARD_FIELDS = [
    "run_id",
    "run_summary",
    "embedding_file",
    "custom_correct",
    "custom_questions_asked",
    "custom_overall_accuracy",
    "custom_semantic_accuracy",
    "custom_syntactic_accuracy",
    "google_correct",
    "google_questions_asked",
    "google_overall_accuracy",
    "google_semantic_accuracy",
    "google_syntactic_accuracy",
]

def ensure_embeddings_dir():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def sanitize_for_filename(text):
    cleaned = []
    for char in text.lower():
        if char.isalnum() or char in {"_", "-"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")

def get_next_run_index():
    ensure_embeddings_dir()

    max_index = 0
    for file_name in os.listdir(EMBEDDINGS_DIR):
        if not file_name.endswith(".txt"):
            continue
        if not file_name.startswith("run_"):
            continue

        prefix = file_name.split("__", 1)[0]
        try:
            run_number = int(prefix.replace("run_", ""))
            max_index = max(max_index, run_number)
        except ValueError:
            continue

    return max_index + 1

def build_run_metadata(file_names, model_type, further_type, context_size, embedding_dim, epochs):
    run_index = get_next_run_index()
    run_id = f"run_{run_index:03d}"

    dataset_tag = "+".join(
        sanitize_for_filename(os.path.splitext(file_name)[0])
        for file_name in file_names
    )
    variant_tag = sanitize_for_filename(further_type.replace(" ", "_"))

    embedding_file_name = (
        f"{run_id}__{dataset_tag}__{model_type}__{variant_tag}"
        f"__ctx{context_size}__dim{embedding_dim}__ep{epochs}.txt"
    )

    run_summary = (
        f"{run_id} | datasets={', '.join(file_names)} | model={model_type} "
        f"| variant={further_type} | context={context_size} "
        f"| dim={embedding_dim} | epochs={epochs}"
    )

    embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_file_name)
    return run_id, run_summary, embedding_path

def choose_saved_embedding_file():
    ensure_embeddings_dir()

    available_files = sorted(
        [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".txt")]
    )

    if not available_files:
        raise FileNotFoundError(
            f"No saved embedding files found in {EMBEDDINGS_DIR}/"
        )

    print("\nAvailable saved embedding files:")
    for i, file_name in enumerate(available_files, start=1):
        print(f"{i}. {file_name}")

    while True:
        try:
            selected_number = int(input("\nChoose embedding file number: ").strip())
            if selected_number < 1 or selected_number > len(available_files):
                raise ValueError
            return os.path.join(EMBEDDINGS_DIR, available_files[selected_number - 1])
        except ValueError:
            print("\nInvalid selection. Please try again.")

def infer_run_metadata_from_embedding_file(embedding_path):
    embedding_file_name = os.path.basename(embedding_path)
    stem = os.path.splitext(embedding_file_name)[0]
    parts = stem.split("__")

    if len(parts) >= 7 and parts[0].startswith("run_"):
        run_id = parts[0]
        dataset_tag = parts[1].replace("+", ", ")
        model_type = parts[2]
        variant_tag = parts[3]
        context_size = parts[4].replace("ctx", "")
        embedding_dim = parts[5].replace("dim", "")
        epochs = parts[6].replace("ep", "")

        run_summary = (
            f"{run_id} | datasets={dataset_tag} | model={model_type} "
            f"| variant={variant_tag} | context={context_size} "
            f"| dim={embedding_dim} | epochs={epochs}"
        )
        return run_id, run_summary

    return "external", f"embedding_file={embedding_file_name}"

def scoreboard_value(summary, section, field):
    if summary is None:
        return "not_run"

    value = summary[section][field]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def build_scoreboard_row(run_id, run_summary, embedding_file_name, custom_summary, google_summary):
    return {
        "run_id": run_id,
        "run_summary": run_summary,
        "embedding_file": embedding_file_name,
        "custom_correct": scoreboard_value(custom_summary, "overall", "correct"),
        "custom_questions_asked": scoreboard_value(custom_summary, "overall", "questions_asked"),
        "custom_overall_accuracy": scoreboard_value(custom_summary, "overall", "accuracy"),
        "custom_semantic_accuracy": scoreboard_value(custom_summary, "semantic", "accuracy"),
        "custom_syntactic_accuracy": scoreboard_value(custom_summary, "syntactic", "accuracy"),
        "google_correct": scoreboard_value(google_summary, "overall", "correct"),
        "google_questions_asked": scoreboard_value(google_summary, "overall", "questions_asked"),
        "google_overall_accuracy": scoreboard_value(google_summary, "overall", "accuracy"),
        "google_semantic_accuracy": scoreboard_value(google_summary, "semantic", "accuracy"),
        "google_syntactic_accuracy": scoreboard_value(google_summary, "syntactic", "accuracy"),
    }

def is_embedding_logged(embedding_file_name):
    if not os.path.exists(SCOREBOARD_FILE):
        return False

    with open(SCOREBOARD_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for existing_row in reader:
            if existing_row.get("embedding_file") == embedding_file_name:
                return True

    return False

def upsert_scoreboard_row(row):
    rows = []

    if os.path.exists(SCOREBOARD_FILE):
        with open(SCOREBOARD_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for existing_row in reader:
                rows.append(existing_row)

    updated = False
    for index, existing_row in enumerate(rows):
        if existing_row.get("embedding_file") == row["embedding_file"]:
            rows[index] = row
            updated = True
            break

    if not updated:
        rows.append(row)

    with open(SCOREBOARD_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCOREBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)