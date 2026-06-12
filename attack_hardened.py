import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running TextFooler Attack on Hardened Model: {device} ---")

    model_path = "./hardened_bert_final"
    print(f"Loading hardened model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    print("Loading clean IMDB test samples...")
    dataset = HuggingFaceDataset("imdb", split="test")

    attack_args = textattack.AttackArgs(
        num_examples=1000,           
        log_to_csv="hardened_attack_results_1000.csv", 
        checkpoint_interval=50,      
        checkpoint_dir="checkpoints_hardened",
        disable_stdout=False,
        parallel=True                 
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)
    print("Starting the attack loop on the vaccinated model...")
    attacker.attack_dataset()

        #checkpoint_file = "checkpoints_hardened/1781028847195.ta.chkpt"
        #attacker = textattack.Attacker.from_checkpoint(attack, dataset, checkpoint_file)
    
        #print(f"Resuming the attack loop from {checkpoint_file}...")
        #attacker.attack_dataset()

if __name__ == "__main__":
    main()
