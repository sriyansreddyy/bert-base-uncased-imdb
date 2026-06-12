import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running TextFooler Attack on: {device} ---")

    model_name = "textattack/bert-base-uncased-imdb"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    print("Loading IMDB test samples...")
    dataset = HuggingFaceDataset("imdb", split="test")

    attack_args = textattack.AttackArgs(
        num_examples=1000,            
        log_to_csv="attack_results_1000.csv", 
        checkpoint_interval=50,      
        checkpoint_dir="checkpoints",
        disable_stdout=False,         
        parallel=True                
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)
    print("Starting the attack loop...")
    attacker.attack_dataset()

if __name__ == "__main__":
    main()
