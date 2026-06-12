import pandas as pd
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

def main():
    print("--- Initializing Adversarial Training Pipeline ---")
    
    model_name = "textattack/bert-base-uncased-imdb"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    print("Extracting adversarial examples from CSV...")
   
    df = pd.read_csv("attack_results_1000.csv")
    
    
    successful_attacks = df[df['result_type'] == 'Successful']
    
    
    adv_texts = successful_attacks['perturbed_text'].str.replace(r'\[\[|\]\]', '', regex=True).tolist()
    adv_labels = successful_attacks['ground_truth_output'].tolist()

    adv_dataset = Dataset.from_dict({"text": adv_texts, "label": adv_labels})

   
    print("Fetching original IMDB training data...")
    clean_dataset = load_dataset("imdb", split="train[:2000]") 

    
    adv_dataset = adv_dataset.cast(clean_dataset.features)
    
    print("Fusing clean and adversarial datasets...")
    mixed_dataset = concatenate_datasets([clean_dataset, adv_dataset]).shuffle(seed=42)

    def tokenize_data(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    print("Tokenizing data...")
    tokenized_dataset = mixed_dataset.map(tokenize_data, batched=True)

    training_args = TrainingArguments(
        output_dir="./hardened_bert",
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=2e-5,              
        weight_decay=0.01,
        report_to="none"                
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Initiating VRAM allocation and training loop...")
    trainer.train()

    print("Saving the hardened model to disk...")
    trainer.save_model("./hardened_bert_final")
    tokenizer.save_pretrained("./hardened_bert_final")
    print("--- Adversarial Retraining Complete ---")

if __name__ == "__main__":
    main()
