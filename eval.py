from transformers import VisionEncoderDecoderModel, BartTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import HuggingfaceLatexDataset, get_image_transform, load_latex_dataset
import re

# model initialization
model_path = "./finetuned-vit-bart"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model.eval().cuda()

image_transform = get_image_transform()

# datasets loading
full_data = load_latex_dataset("full", "test", data_path="linxy/LaTeX_OCR")
synthetic_data = load_latex_dataset("synthetic_handwrite", "test", data_path="linxy/LaTeX_OCR")
test_full_dataset = HuggingfaceLatexDataset(full_data, tokenizer, image_transform)
test_synthetic_dataset = HuggingfaceLatexDataset(synthetic_data, tokenizer, image_transform)

# choose your metric
bleu_metric = load("bleu")
exact_match_metric = load("exact_match")
# edit_distance_metric = load("edit_distance")

# dataloaders
test_full_dataloader = DataLoader(test_full_dataset, batch_size=1)
test_synthetic_dataloader = DataLoader(test_synthetic_dataset, batch_size=1)

def normalize_latex(latex_str):
    # Remove all whitespace (spaces, tabs, newlines)
    normalized = re.sub(r'\s+', '', latex_str)
    
    # Remove unnecessary curly braces around single characters in sub/superscripts
    # Example: x_{i} → x_i, y^{n} → y^n
    normalized = re.sub(r'([_^])\{([a-zA-Z0-9])\}', r'\1\2', normalized)
    
    # Remove empty braces {}
    normalized = normalized.replace('{}', '')
    
    # Optional: Normalize common variants
    # Convert \ldots, \cdots to \dots
    normalized = re.sub(r'\\[lc]dots', r'\\dots', normalized)
    
    # Remove braces around single Greek letters
    # Example: {\alpha} → \alpha
    normalized = re.sub(r'\{(\\\w+)\}', r'\1', normalized)
    
    return normalized

def evaluate(dataloader, dataset_name):
    print(f"Evaluating on {dataset_name} dataset...")
    resutls = []

    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].cuda()
        labels = batch["labels"].cuda()

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=512)

        # Decode the generated ids and labels
        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"prediction: {pred}")
        print(f"length: {len(pred)}")
        label = tokenizer.decode(labels[0][labels[0] != -100], skip_special_tokens=True)
        print(f"label: {label}")
        print(f"length: {len(label)}")   

        # Calculate metrics
        bleu_score = bleu_metric.compute(predictions=[pred], references=[[label]])["bleu"]
        exact_match_score = exact_match_metric.compute(predictions=[pred], references=[[label]])["exact_match"]
        # edit_distance_score = edit_distance_metric.compute(predictions=pred, references=label)["edit_distance"]

        resutls.append({
            "bleu": bleu_score,
            "exact_match": exact_match_score,
            # "edit_distance": edit_distance_score
        })

    print(f"{dataset_name} Evaluation Results:")
    print("BLEU score:", bleu_metric.compute())
    print("Exact Match score:", exact_match_metric.compute())
    # print("Edit Distance score:", edit_distance_metric.compute())

evaluate(test_full_dataloader, "Full Test")
evaluate(test_synthetic_dataloader, "Synthetic Handwritten Test")