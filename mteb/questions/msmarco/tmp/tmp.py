import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def write_json_file(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


# def find_similar_questions(questions, model_name='all-MiniLM-L6-v2', similarity_threshold=0.95):
def find_similar_questions(questions, model_name='sentence-t5-xl', similarity_threshold=0.98):
    # Initialize the model
    model = SentenceTransformer(model_name)

    # Extract question texts
    question_texts = [question['question'] for question in questions]

    # Generate embeddings
    embeddings = model.encode(question_texts, convert_to_tensor=True)

    # Move embeddings to CPU for cosine_similarity calculation
    embeddings = embeddings.cpu()

    similar_questions = []
    
    # Calculate cosine similarity
    cos_sim = cosine_similarity(embeddings, embeddings)

    for i in range(len(questions)):
        for j in range(i+1, len(questions)):
            if cos_sim[i, j] >= similarity_threshold:
                similar_questions.append({
                    'question_id_1': questions[i]['question_id'],
                    'question_1': questions[i]['question'],
                    'question_id_2': questions[j]['question_id'],
                    'question_2': questions[j]['question'],
                    'similarity': float(cos_sim[i, j])
                })
    
    return similar_questions

input_file = 'questions.json'  # Replace with your JSON file path
questions = read_json_file(input_file)

similar_questions = find_similar_questions(questions)

output_file = 'output_similar_questions.json'  # Replace with your desired output file name
write_json_file(similar_questions, output_file)

print(f"Similar questions with nuanced scoring written to {output_file}.")