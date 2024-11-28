import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import time

# Load the tokenizer and model from the Hugging Face Hub
model_name = "keshish/bart-eye-qa-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the dataset from Hugging Face and select the 'input' column from the test split
dataset_name = "QIAIUNCC/EYE-QA-PLUS"
dataset = load_dataset(dataset_name, split="test")
questions = dataset["input"][:1000]  # Take the first 1000 rows of the "input" column

# List to store results
generated_answers = []

# Initialize timing
start_time = time.time()

# Generate answers for each question with progress updates
for i, question in enumerate(questions, start=1):
    # Encode the input question and generate the model's answer
    inputs = tokenizer.encode(question, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Append the question and generated answer to the list
    generated_answers.append({
        "Question": question,
        "Generated Answer": answer
    })
    
    # Calculate elapsed time and estimated time remaining
    elapsed_time = time.time() - start_time
    avg_time_per_question = elapsed_time / i
    estimated_total_time = avg_time_per_question * len(questions)
    time_remaining = estimated_total_time - elapsed_time
    
    # Print progress update
    print(f"Processed question {i}/{len(questions)}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Estimated time remaining: {time_remaining:.2f} seconds")
    print(f"Generated Answer: {answer}")
    print("-" * 40)

# Convert results to a DataFrame and save to a new CSV file
output_csv_path = "/content/sample_data/generated_answers.csv"  # Replace with the desired output path
output_df = pd.DataFrame(generated_answers)
output_df.to_csv(output_csv_path, index=False)
print(f"\nGenerated answers saved to '{output_csv_path}'")
