import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

class SMPCSecureModelWrapper(HuggingFaceModelWrapper):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def __call__(self, text_input_list):
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text_input_list, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(model_device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        hard_labels = torch.argmax(logits, dim=-1)
        
        secure_outputs = F.one_hot(hard_labels, num_classes=logits.shape[-1]).float()
        
        return secure_outputs.cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Defense-in-Depth Pipeline on: {device} ---")

    model_path = "./hardened_bert_final"
    print(f"Loading hardened model weights from: {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    secure_model_wrapper = SMPCSecureModelWrapper(model, tokenizer)

    attack = textattack.attack_recipes.TextFoolerJin2019.build(secure_model_wrapper)

    print("Loading clean IMDB test samples...")
    dataset = HuggingFaceDataset("imdb", split="test")

    attack_args = textattack.AttackArgs(
        num_examples=1000,            
        log_to_csv="combined_defense_results_1000.csv", 
        checkpoint_interval=50,       
        checkpoint_dir="checkpoints_combined",
        disable_stdout=False,
        parallel=False                
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)
    print("Starting training...")
    attacker.attack_dataset()

if __name__ == "__main__":
    main()