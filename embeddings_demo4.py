from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Define the question and answers (1 correct, 4 closely related wrong ones)
phrases = [
    "What spacecraft was used in the mission to carry the first humans to the moon?",  # Question
    "Apollo 11 was the spacecraft used to carry the first humans to the moon.",       # Correct Answer
    "Apollo 12 was the spacecraft used to carry the first humans to the moon.",         # Wrong Answer
    "Apollo 14 was the spacecraft used to carry astronauts on the third successful moon landing mission.", # Wrong Answer
    "Apollo 10 was the spacecraft used to carry the first humans to the moon.", # Wrong Answer
    "Apollo 16 was the spacecraft that carried astronauts to explore the lunar highlands."   # Wrong Answer
]

# Generate embeddings for the question and answers using OpenAI embeddings
embeddings = embedding_model.get_text_embedding_batch(phrases)

# Convert embeddings to a numpy array
embeddings_array = np.array(embeddings)

# Print the first phrase and the first several elements of its embedding
print(f"Phrase: {phrases[0]}")
print(f"First 5 elements of its embedding: {embeddings_array[0][:5]}\n")

# Compute cosine similarity between the embeddings
similarity_matrix = cosine_similarity(embeddings_array)

print("\nDetailed Similarity Results:\n")

# Output comparison between question and answers with improved readability
for i in range(1, len(phrases)):
    print(f"Cosine similarity between the question and:\n  '{phrases[i]}'\n  => {similarity_matrix[0, i]:.4f}\n")