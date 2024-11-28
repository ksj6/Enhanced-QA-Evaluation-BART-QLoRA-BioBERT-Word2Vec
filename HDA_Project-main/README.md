# Eye-QA BART Fine-Tuning and Evaluation

## Project Overview

This project focuses on fine-tuning a **BART (Bidirectional and Auto-Regressive Transformers)** model on a specialized ophthalmology question-answer dataset, **EYE-QA-PLUS**. The goal is to enable the model to generate high-quality answers to clinical questions related to ophthalmology. Once fine-tuned, we assess the model's performance by comparing the cosine similarity between the model-generated answers and the initial answers provided in the dataset. **BioBERT** and **Word2Vec** embeddings are used to generate these similarity scores, which allow us to determine if the fine-tuned model produces improved, contextually relevant answers.

To get started, clone this repository:

```bash
git clone https://github.com/Dhairyashil2002/HDA_Project
cd HDA_Project
```

## Dataset Details

The dataset used in this project is [**EYE-QA-PLUS**](https://huggingface.co/datasets/QIAIUNCC/EYE-QA-PLUS), containing:
- **Inputs**: General and fine-grained ophthalmology questions.
- **Outputs**: Corresponding answers with additional instructions and source information.
- **Data Splits**:
  - **Train split**: 32,000 rows, used for fine-tuning the BART model.
  - **Test split**: 4,910 rows, reserved for evaluation.

The test split data is available in the file `initial_dataset_cosine_scores.csv` in the `data` folder. This file includes initial cosine similarity scores for the questions and answers, generated using the Python script `Files (Codes)/Initial_Data_plots_cosine_similarity_initial.py`.

## Project Structure

The project directory is structured as follows:

```plaintext
|-- Data
    |-- Final_initial_generated_cosine_csv_biobert.csv
    |-- Final_initial_generated_cosine_csv_word_to_vec.csv
    |-- Initial_dataset_cosine_scores.csv
|-- Files ( Codes )
    |-- BART-Finetuned.ipynb
    |-- Final_HDA_Project.ipynb
    |-- Initial_Data_plots_cosine_similarity_initial.py
    |-- New_answer_generation_finetuned_bart.py
    |-- Plots_Histograms_tsne.py
|-- README.md
|-- requirements.txt
|-- Results ( Outputs )
    |-- BioBERT Histograms and TSNE Plots
        |-- BioBERT_Cosine_Similarity_histogram.png
        |-- BioBERT_Cosine_Similarity_Initial_Generated_histogram.png
        |-- BioBERT_Cosine_Similarity_Initial_Generated_vs_Cosine_Similarity_Question_Generated_tsne_plot.png
        |-- BioBERT_Cosine_Similarity_Question_Generated_histogram.png
        |-- BioBERT_Cosine_Similarity_vs_Cosine_Similarity_Initial_Generated_tsne_plot.png
        |-- BioBERT_Cosine_Similarity_vs_Cosine_Similarity_Question_Generated_tsne_plot.png
    |-- BioBERT_cosine_similarity_combined_bar_plot.png
    |-- biobert_metrics.csv
    |-- Initial_data_cosine_similarity_plot.jpg
    |-- Initial_data_TSNE_plot.jpg
    |-- Word2Vec_cosine_similarity_combined_bar_plot.png
    |-- word2vec_metrics.csv
    |-- WordtoVec Histograms and TSNE Plots
        |-- Word2Vec_Cosine_Similarity_histogram.png
        |-- Word2Vec_Cosine_Similarity_Initial_Generated_histogram.png
        |-- Word2Vec_Cosine_Similarity_Initial_Generated_vs_Cosine_Similarity_Question_Generated_tsne_plot.png
        |-- Word2Vec_Cosine_Similarity_Question_Generated_histogram.png
        |-- Word2Vec_Cosine_Similarity_vs_Cosine_Similarity_Initial_Generated_tsne_plot.png
        |-- Word2Vec_Cosine_Similarity_vs_Cosine_Similarity_Question_Generated_tsne_plot.png
```
### Step 2: Evaluate Model Performance

After fine-tuning, evaluate the model's ability to generate relevant answers by running the `Final_HDA_Project.ipynb` notebook. This evaluation compares the cosine similarity scores between:

*   **Question vs. Initial Answer**
*   **Question vs. Generated Answer**
*   **Initial Answer vs. Generated Answer**

The evaluation uses both BioBERT and Word2Vec embeddings, with results visualized through histograms and t-SNE plots.


### Step 3: View and Analyze Results

After running the evaluation notebook, results are saved in the `Results (Outputs)` folder. Key files and outputs include:

*   **BioBERT Folder**: Contains histograms and t-SNE plots for cosine similarity comparisons (Question vs. Initial Answer, Question vs. Generated Answer, Initial Answer vs. Generated Answer).
*   **Word2Vec Folder**: Contains similar visualizations for Word2Vec embeddings.
*   **Metrics Files**:
    *   `biobert_metrics.csv`: Summary of average cosine similarity scores using BioBERT embeddings.
    *   `wordtovec_metrics.csv`: Summary of average cosine similarity scores using Word2Vec embeddings.

These visualizations and metrics files provide insights into how close the model-generated answers are to the initial answers and whether the model improvements are statistically significant.

## Dependencies

Before running the notebooks, install the necessary dependencies using the following commands:

```bash
pip install -r requirements.txt
```


**Command to Run**:

```bash
jupyter notebook "Files (Codes)/Final_HDA_Project.ipynb"
```

## Results Interpretation

*   **Higher Cosine Similarity**: Improved similarity between **Question vs. Generated Answer** (compared to **Question vs. Initial Answer**) suggests the model is producing answers that are more contextually relevant to the questions.
*   **Consistent Initial vs. Generated Answer Similarity**: Higher similarity between **Initial Answer vs. Generated Answer** indicates the model has learned to generate responses similar in quality to the original dataset answers.
*   **Visual Analysis**: The histograms and t-SNE plots provide visual confirmation of similarity trends and distribution patterns for BioBERT and Word2Vec embeddings.

  The below image is of the Initial datset's T-SNE Visualization plot of the Question and the output embeddings which gives us the similar embeddings result for the initial data.

  ![Initial Data t-SNE Plot](Results%20(%20Outputs%20)/Initial_data_TSNE_plot.jpg)

  The below image is the combined bar plot for the cosine similarities scores for the Question and the Initial , Question and the Generated answer and the Initial and the generated answer which effectively shows that the Bart finetuned model is performing well as we get a good cosine similarity score between the Initial and the Generated Answer.
  
  ![Initial Data t-SNE Plot](Results%20(%20Outputs%20)/BioBERT_cosine_similarity_combined_bar_plot.png)

  Below are the following more plots and visualizations independently for the Bio-BERT and the Word to Vec embedding models.
  ![Initial Data t-SNE Plot](Results%20(%20Outputs%20)/BioBERT%20Histograms%20and%20TSNE%20Plots/BioBERT_Cosine_Similarity_Initial_Generated_histogram.png)
  ![Initial Data t-SNE Plot](Results%20(%20Outputs%20)/BioBERT%20Histograms%20and%20TSNE%20Plots/BioBERT_Cosine_Similarity_vs_Cosine_Similarity_Initial_Generated_tsne_plot.png)
  ![Initial Data t-SNE Plot](Results%20(%20Outputs%20)/WordtoVec%20Histograms%20and%20TSNE%20Plots/Word2Vec_Cosine_Similarity_Initial_Generated_vs_Cosine_Similarity_Question_Generated_tsne_plot.png)

## Conclusion

This project demonstrates the effectiveness of fine-tuning the BART model for domain-specific question answering in the field of ophthalmology. Cosine similarity scores calculated using BioBERT and Word2Vec embeddings provide quantitative and visual evidence of model improvement, aiding in the evaluation of answer relevance and quality.

Each model's metrics for comparing **Initial vs. Generated Answer** and **Question vs. Generated Answer** are provided below.
## Word2Vec Model Metrics

| Metric               | Initial vs. Generated Answer | Question vs. Generated Answer |
|----------------------|------------------------------|-------------------------------|
| **Cosine Similarity** | 0.7819642357379198           | 0.7121449808776379            |
| **Euclidean Distance**| 0.7578837997764349           | 0.9375786134600639            |
| **Manhattan Distance**| 10.473982694801043           | 12.95119222612609             |

**Table 1:** Metrics for the Word2Vec model show moderate to high cosine similarity, with some distance variation in Euclidean and Manhattan measures, indicating differing levels of similarity between initial answers, questions, and generated answers.

## BioBERT Model Metrics

| Metric               | Initial vs. Generated Answer | Question vs. Generated Answer |
|----------------------|------------------------------|-------------------------------|
| **Cosine Similarity** | 0.9370405098199844           | 0.908356391787529             |
| **Euclidean Distance**| 3.4703431522810515           | 4.370629085838795             |
| **Manhattan Distance**| 76.29245510666182            | 95.70385496363345             |

**Table 2:** BioBERT model metrics indicate high cosine similarity and a larger variation in Euclidean and Manhattan distances compared to Word2Vec, suggesting stronger similarity retention in the generated answers.

**Overall** BioBERT model performs better than Word2Vec on Q&A tasks, which can be seen by the metrics above.

## Acknowledgments

*   **EYE-QA-PLUS Dataset**: Dataset hosted by Hugging Face.
*   **Transformers Library**: For the BART model and embeddings.
*   **BioBERT** and **Word2Vec**: For generating embeddings and calculating cosine similarity metrics.

## Group 10( Members )

*  21bds004  - Anubhav Gupta
*  21bds011  - Rohit Chaudhari
*  21bds013  - Chintan Chawda 
*  21bds017  - Dhairyashil Shendage
*  21bds025  - Kartik Jagtap
*  21bds026  - Kashish Lonpande

## Course Instructor
* Dr. Girish G.N.

