import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    # 1. Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running TextFooler Attack on: {device} ---")

    # 2. Load the exact fine-tuned BERT model and tokenizer
    model_name = "textattack/bert-base-uncased-imdb"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Wrap the model for TextAttack
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # 3. Load the TextFooler attack recipe
    # TextFooler replaces important words with synonyms based on embedding similarity 
    # while ensuring part-of-speech consistency and semantic preservation.
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    # 4. Load a subset of the IMDB test dataset
    # We load 20 samples here because adversarial attacks search a large space 
    # of mutations and take significantly longer than standard evaluation runs.
    print("Loading IMDB test samples...")
    dataset = HuggingFaceDataset("imdb", split="test")

   # 5. Configure the attack arguments
    attack_args = textattack.AttackArgs(
        num_examples=1000,             # A solid benchmark run
        log_to_csv="attack_results_1000.csv", 
        checkpoint_interval=50,      
        checkpoint_dir="checkpoints",
        disable_stdout=False,         
        parallel=True                 # Let's see if the GPU handles the workers now!
    )

    # 6. Initialize the attacker and launch
    attacker = textattack.Attacker(attack, dataset, attack_args)
    print("Starting the attack loop...")
    attacker.attack_dataset()

if __name__ == "__main__":
    main()