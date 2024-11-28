# !pip install datasets transformers torch scikit-learn matplotlib seaborn umap-learn

# Import necessary libraries
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

# Step 1: Load the dataset and preprocess it
dataset = load_dataset("QIAIUNCC/EYE-QA-PLUS", split="test")
df = dataset.to_pandas()

# Limit to the first 100 rows
df = df.head(1000)

# Drop 'source' and 'instruction' columns; use 'input' as 'question'
df = df.drop(columns=['source', 'instruction'])
df = df.rename(columns={'input': 'question'})
df = df[['question', 'output']]

# Step 2: Load BioBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Function to get embeddings with truncation to avoid length-related errors
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.cpu().numpy()

# Step 3: Generate embeddings for each 'question' and 'output'
questions_embeddings = []
outputs_embeddings = []

print("Generating embeddings for questions and outputs...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    question_embedding = get_embeddings(row['question'])
    output_embedding = get_embeddings(row['output'])
    
    questions_embeddings.append(question_embedding)
    outputs_embeddings.append(output_embedding)

questions_embeddings = torch.cat([torch.tensor(embed) for embed in questions_embeddings])
outputs_embeddings = torch.cat([torch.tensor(embed) for embed in outputs_embeddings])

# Step 4: Calculate similarity and distance scores
print("Calculating similarity and distance scores...")

# Cosine similarity between each question and its respective output
q_to_output_cos_sim = [cosine_similarity([q], [o])[0][0] for q, o in zip(questions_embeddings, outputs_embeddings)]

# Euclidean and Manhattan distances between each question and output
q_to_output_euc_dist = [euclidean_distances([q], [o])[0][0] for q, o in zip(questions_embeddings, outputs_embeddings)]
q_to_output_man_dist = [manhattan_distances([q], [o])[0][0] for q, o in zip(questions_embeddings, outputs_embeddings)]

# Step 5: Visualizations

# 1. t-SNE visualization of question and output embeddings
print("Generating t-SNE plot...")
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(torch.cat([questions_embeddings, outputs_embeddings]))

plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:len(questions_embeddings), 0], embeddings_2d[:len(questions_embeddings), 1], c='blue', label='Questions')
plt.scatter(embeddings_2d[len(questions_embeddings):, 0], embeddings_2d[len(questions_embeddings):, 1], c='red', label='Outputs')
plt.legend()
plt.title("t-SNE Visualization of Question and Output Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# 2. Histogram of Cosine Similarity Scores
plt.figure(figsize=(10, 6))
sns.histplot(q_to_output_cos_sim, bins=20, kde=True)
plt.title("Distribution of Cosine Similarity Scores")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.show()

# 3. Display few sample embeddings
print("Sample question embeddings (first 3):", questions_embeddings[:5])
print("Sample output embeddings (first 3):", outputs_embeddings[:5])

# Step 6: Save results to DataFrame and CSV
# Create DataFrame to store similarity and distance scores
results_df = pd.DataFrame({
    "Question": df['question'],
    "Output": df['output'],
    "Cosine_Similarity": q_to_output_cos_sim,
    "Euclidean_Distance": q_to_output_euc_dist,
    "Manhattan_Distance": q_to_output_man_dist
})

# Save to CSV
results_df.to_csv("similarity_distance_scores_1500.csv", index=False)
print("Scores saved to 'similarity_distance_scores_1500.csv'")