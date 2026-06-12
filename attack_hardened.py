import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running TextFooler Attack on Hardened Model: {device} ---")

    # 1. Load the locally trained, hardened model
    model_path = "./hardened_bert_final"
    print(f"Loading hardened model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # 2. Load the TextFooler attack recipe
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    # 3. Load the IMDB test dataset
    print("Loading clean IMDB test samples...")
    dataset = HuggingFaceDataset("imdb", split="test")

    # 4. Configure the attack arguments
    attack_args = textattack.AttackArgs(
        num_examples=1000,            # The full benchmark run
        log_to_csv="hardened_attack_results_1000.csv", 
        checkpoint_interval=50,       # Increased checkpointing for safety
        checkpoint_dir="checkpoints_hardened",
        disable_stdout=False,
        parallel=True                 
    )

    # 5. Initialize and launch
    attacker = textattack.Attacker(attack, dataset, attack_args)
    print("Starting the attack loop on the vaccinated model...")
    attacker.attack_dataset()

        # 5. Initialize and launch
        #checkpoint_file = "checkpoints_hardened/1781028847195.ta.chkpt"
        #attacker = textattack.Attacker.from_checkpoint(attack, dataset, checkpoint_file)
    
        #print(f"Resuming the attack loop from {checkpoint_file}...")
        #attacker.attack_dataset()

if __name__ == "__main__":
    main()