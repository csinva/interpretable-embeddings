import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

def read_json_file(filepath):
    """Reads a JSON file and returns its content."""
    with open(filepath, 'r') as file:
        return json.load(file)

def write_json_file(data, filepath):
    """Writes data to a JSON file."""
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def cluster_and_filter_questions(questions, model_name='all-MiniLM-L6-v2', eps=0.05, min_samples=2):
    """Clusters questions, selects unique ones per cluster, and tracks removed questions."""
    model = SentenceTransformer(model_name)
    question_texts = [question['question'] for question in questions]
    embeddings = model.encode(question_texts, convert_to_tensor=False)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)

    unique_questions = []
    removed_questions = []
    added_clusters = set()

    for question_idx, cluster_label in enumerate(clustering.labels_):
        if cluster_label == -1:  # Treat outliers as unique
            unique_questions.append(questions[question_idx])
            continue
        cluster_label_str = str(cluster_label)
        if cluster_label_str not in added_clusters:
            added_clusters.add(cluster_label_str)
            unique_questions.append(questions[question_idx])
        else:
            removed_questions.append(questions[question_idx])

    return unique_questions, removed_questions

def renumber_question_ids(questions):
    """Renumbers question IDs from 1 to n, maintaining the original question content."""
    return [{'question_id': i + 1, 'question': question['question']} for i, question in enumerate(questions)]

# Specify the file paths
input_file = 'questions.json'  # Path to your input JSON file
output_file_unique = 'unique_questions.json'  # File to write unique questions
output_file_removed = 'removed_questions.json'  # File to write removed questions
output_file_renumbered = 'renumbered_questions.json'  # File to write renumbered unique questions

# Processing
questions = read_json_file(input_file)
unique_questions, removed_questions = cluster_and_filter_questions(questions)
write_json_file(unique_questions, output_file_unique)
write_json_file(removed_questions, output_file_removed)

# Renumber and write the renumbered unique questions
renumbered_questions = renumber_question_ids(unique_questions)
write_json_file(renumbered_questions, output_file_renumbered)