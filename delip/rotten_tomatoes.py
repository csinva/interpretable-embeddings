# 1. **Identify Style Attributes Relevant to Sentiment in Movie Reviews**
#    Use GPT-4 to brainstorm a list of potential style attributes related to sentiment using a two-stage prompting method.

from datasets import load_dataset
from openai import OpenAI
import os
import re
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Set OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set model name
model_name = "gpt-4"

# Load the Rotten Tomatoes dataset from Hugging Face
print("Loading Rotten Tomatoes dataset...")
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
reviews_data = dataset['train']
reviews = reviews_data['text']
print("Dataset loaded successfully. Number of reviews:", len(reviews))

# Define a function to generate potential style attributes for sentiment analysis using a two-stage prompting method
def generate_style_attributes():
    print("Generating style attributes using GPT-4 (Stage 1)...")
    # Stage 1: Brainstorming attributes
    prompt_stage_1 = (
        "Brainstorm a comprehensive list of stylistic features that can be used to analyze movie reviews. "
        "These should include elements of writing style, such as tone, use of emotional words, sentence complexity, "
        "use of intensifiers, and other linguistic features that may differentiate positive and negative reviews."
    )
    
    response_stage_1 = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content":prompt_stage_1}], max_tokens=150
    )
    raw_attributes = response_stage_1.choices[0].message.content.strip().split("\n")
    print("Stage 1 complete. Raw attributes generated:", raw_attributes)
    
    # Stage 2: Refining and filtering attributes
    print("Refining style attributes using GPT-4 (Stage 2)...")
    prompt_stage_2 = (
        "Here is a list of stylistic features identified for movie reviews: \n"
        f"{', '.join(raw_attributes)}\n"
        "Refine this list to include only those features that are most likely to impact sentiment analysis, "
        "specifically in distinguishing between positive and negative movie reviews."
    )
    
    response_stage_2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":prompt_stage_2,}],
        max_tokens=150
    )
    refined_attributes = response_stage_2.choices[0].message.content.strip().split("\n")
    print("Stage 2 complete. Refined attributes generated:", refined_attributes)
    
    return refined_attributes

# Generate style attributes using GPT-4 two-stage prompting
style_attributes = generate_style_attributes()
print("Style Attributes Suggested by GPT-4:")
for attribute in style_attributes:
    print(f"- {attribute}")

# 2. **Generate Binary Embeddings for Reviews Using GPT-4**
#    Use GPT-4 to score each attribute for each review to generate binary embeddings.

def gpt4_binary_score(review, attributes):
    embeddings = []
    for attribute in attributes:
        prompt = f"Given the movie review: '{review}', does it exhibit the following attribute: '{attribute}'? Answer with Yes or No."
        # print(f"Prompting GPT-4 for attribute '{attribute}' on review: '{review[:50]}...'")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content":prompt,}],
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()
        # print(f"Response for attribute '{attribute}':", answer)
        score = 1 if answer == "yes" else 0
        embeddings.append(score)
    return embeddings

# Generate binary embeddings for each review
print("Generating embeddings for sample reviews...")
embeddings = []
for idx, review in enumerate(reviews[:5]):  # Limiting to first 5 reviews for demonstration
    print(f"Processing review {idx + 1}/{5}")
    scores = gpt4_binary_score(review, style_attributes)
    embeddings.append(scores)
    print(f"Embedding for review {idx + 1}: {scores}")

# 3. **Apply the Scoring System to the Rotten Tomatoes Dataset**
# Apply GPT-4 binary scoring to generate embeddings for the entire dataset

def generate_review_embedding(review, attributes):
    # print(f"Generating embedding for review: '{review[:50]}...'")
    return gpt4_binary_score(review, attributes)

print("Generating embeddings for the entire dataset...")
rotten_tomatoes_data = pd.DataFrame(reviews_data)
rotten_tomatoes_data['Embeddings'] = rotten_tomatoes_data['text'].apply(lambda x: generate_review_embedding(x, style_attributes))
print("Embeddings generated for the entire dataset.")

print(rotten_tomatoes_data.head())

# 4. **Calculate the Clustering Score Using Embedding Metrics**
# Example: calculate clustering score for sentiment labels (positive vs negative)
print("Calculating clustering scores...")
positive_reviews = rotten_tomatoes_data[rotten_tomatoes_data['label'] == 1]['Embeddings']
negative_reviews = rotten_tomatoes_data[rotten_tomatoes_data['label'] == 0]['Embeddings']

# Compute average intra-class and inter-class distances
def calculate_average_distance(scores):
    scores_list = list(scores)
    distance_sum = 0
    count = len(scores_list)
    distances = euclidean_distances(scores_list)
    for i in range(count):
        for j in range(count):
            if i != j:
                distance_sum += distances[i][j]
    average_distance = distance_sum / (count * (count - 1))
    print("Average distance calculated:", average_distance)
    return average_distance

print("Calculating intra-class distances for positive reviews...")
intra_class_positive = calculate_average_distance(positive_reviews)
print("Calculating intra-class distances for negative reviews...")
intra_class_negative = calculate_average_distance(negative_reviews)
print("Calculating inter-class distance...")
inter_class_distance = euclidean_distances(list(positive_reviews), list(negative_reviews)).mean()
print("Inter-class distance calculated:", inter_class_distance)

clustering_score = inter_class_distance - (intra_class_positive + intra_class_negative) / 2
print("\nClustering Score (StyleGenome):", clustering_score)

# 5. **Analyze the Results**
#    Evaluate the clustering score and analyze discriminative features.
print("\nAnalysis of Results:")
for attribute in style_attributes:
    print(f"Analyzing attribute: {attribute}")

# 6. **Update Benchmark Table with StyleGenome Results**
print("Updating benchmark table with StyleGenome results...")
benchmark_table = pd.read_csv("benchmark_results.csv")

# Add StyleGenome benchmark results
table_entry = {
    "Method": "StyleGenome",
    "Clustering Score": clustering_score,
}
benchmark_table = benchmark_table.append(table_entry, ignore_index=True)

# Save the updated table
benchmark_table.to_csv("benchmark_results_updated.csv", index=False)
print("\nBenchmark Table updated with StyleGenome results.")
