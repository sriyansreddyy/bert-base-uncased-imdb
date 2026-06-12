import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Baseline Evaluation on: {device} ---")
    
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Load the Full Clean Dataset
    print("Loading the full IMDB test dataset (25,000 samples)...")
    dataset = load_dataset("imdb", split="test")
    
    # 3. Load Frozen Victim Model
    model_name = "textattack/bert-base-uncased-imdb"
    print(f"Loading victim model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval() # Explicitly lock the model for inference

    # 4. Tokenize the Data
    def tokenize_data(example):
        return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_data, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 5. Create DataLoader
    # We can use a larger batch size here (e.g., 64) because pure inference uses much less VRAM than the attack loop.
    dataloader = DataLoader(tokenized_dataset, batch_size=64) 

    # 6. Evaluation Loop
    correct = 0
    total = 0

    print("Starting evaluation loop...")
    with torch.no_grad(): # Critical: disables gradient calculation to save VRAM and speed up processing
        for batch in tqdm(dataloader, desc="Evaluating Batches"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Get predictions and compare to ground truth
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # 7. Calculate and Output Metrics
    accuracy = (correct / total) * 100
    print(f"\n--- Final Baseline Metrics ---")
    print(f"Total Examples Evaluated : {total}")
    print(f"Correct Predictions      : {correct}")
    print(f"Clean Data Accuracy      : {accuracy:.2f}%")

if __name__ == "__main__":
    main()