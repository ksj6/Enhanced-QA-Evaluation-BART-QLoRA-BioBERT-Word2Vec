# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path

# Set up the directory to save results
output_dir = Path("/content/drive/MyDrive/finallll_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Load both CSV files
biobert_file_path = "/content/Final_initial_generated_cosine_csv_biobert.csv"  # Replace with your BioBERT CSV file path
word2vec_file_path = "/content/Final_initial_generated_cosine_csv_word_to_vec.csv"  # Replace with your Word2Vec CSV file path

df_biobert = pd.read_csv(biobert_file_path)
df_word2vec = pd.read_csv(word2vec_file_path)

# Function to plot histograms and t-SNE for cosine similarity scores
def plot_analysis(df, model_name):
    # Plot histograms for each cosine similarity metric
    similarity_columns = [
        "Cosine_Similarity",
        "Cosine_Similarity_Initial_Generated",
        "Cosine_Similarity_Question_Generated"
    ]
    
    for col in similarity_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"{model_name} - Distribution of {col}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.savefig(output_dir / f"{model_name}_{col}_histogram.png")
        plt.close()

    # Define combinations for t-SNE plots
    similarity_pairs = [
        ("Cosine_Similarity", "Cosine_Similarity_Initial_Generated"),
        ("Cosine_Similarity_Initial_Generated", "Cosine_Similarity_Question_Generated"),
        ("Cosine_Similarity", "Cosine_Similarity_Question_Generated")
    ]

    # Generate t-SNE plots with distinct colors for each combination
    for (sim1, sim2) in similarity_pairs:
        embeddings = np.stack([df[sim1], df[sim2]], axis=1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.arange(len(tsne_results)), cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label="Sample Index")
        plt.title(f"{model_name} - t-SNE of {sim1} vs {sim2}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)
        plt.savefig(output_dir / f"{model_name}_{sim1}_vs_{sim2}_tsne_plot.png")
        plt.close()

# Perform analysis for BioBERT and Word2Vec files
plot_analysis(df_biobert, "BioBERT")
plot_analysis(df_word2vec, "Word2Vec")

print(f"All plots saved to '{output_dir}' directory.")
