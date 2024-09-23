from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Define the trivia questions and matching answers
phrases = [
    "Who was the first president of the United States?",
    "What is the capital city of France?",
    "In what year did humans first land on the moon?",
    "Which element on the periodic table has the chemical symbol O?",
    "What is the largest planet in the solar system?",
    "The first president of the United States was George Washington.",
    "The capital city of France is Paris.",
    "Humans first landed on the moon in the year 1969.",
    "The chemical symbol O represents the element Oxygen.",
    "The largest planet in the solar system is Jupiter."
]

# Generate embeddings for each phrase using OpenAI embeddings
embeddings = embedding_model.get_text_embedding_batch(phrases)

# Convert embeddings to a numpy array
embeddings_array = np.array(embeddings)

# Print the first phrase and the first several elements of its embedding
print(f"Phrase: {phrases[0]}")
print(f"First 5 elements of its embedding: {embeddings_array[0][:5]}\n")

# Compute cosine similarity between the embeddings
similarity_matrix = cosine_similarity(embeddings_array)

# Print the cosine similarity matrix
print("Cosine Similarity Matrix:")
print(np.round(similarity_matrix, 2))
print("\nDetailed Similarity Results:\n")

# Output comparison between phrases with improved readability
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        print(f"Cosine similarity between:\n  '{phrases[i]}'\n  and\n  '{phrases[j]}'\n  => {similarity_matrix[i, j]:.4f}\n")