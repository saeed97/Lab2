from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Define the new space-related questions and answers
phrases = [
    "What year did the first human land on the moon?",
    "Which planet is known as the Red Planet?",
    "What is the largest moon of Saturn?",
    "Who was the first person to travel into space?",
    "What is the name of NASA's rover that landed on Mars in 2021?",
    "The first human landed on the moon in 1969.",
    "The planet known as the Red Planet is Mars.",
    "The largest moon of Saturn is Titan.",
    "Yuri Gagarin was the first person to travel into space.",
    "NASA's rover that landed on Mars in 2021 is named Perseverance."
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