from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import HuggingfaceLatexDataset, get_image_transform, load_latex_dataset
import re
import os

def get_latest_checkpoint_by_time(model_dir: str) -> str:
    checkpoint_dirs = [
        os.path.join(model_dir, d)
        for d in os.listdir(model_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, d))
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {model_dir}")

    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint

def evaluate(dataloader, dataset_name):
    print(f"Evaluating on {dataset_name} dataset...")
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].cuda()
        labels = batch["labels"].cuda()

        with torch.no_grad():
            generated_ids = model.module.generate(pixel_values, max_length=600)

        # Decode the generated ids and labels
        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(f"prediction: {pred}")
        # print(f"length: {len(pred)}")
        label = tokenizer.decode(labels[0][labels[0] != -100], skip_special_tokens=True)
        # print(f"label: {label}")
        # print(f"length: {len(label)}")   

        all_preds.append(pred)
        all_labels.append(label)
    
    bleu_metric.add_batch(predictions = all_preds, references = all_labels)
    exact_match_metric.add_batch(predictions = all_preds, references = all_labels)
    cer_metric.add_batch(predictions = all_preds, references = all_labels)
    
    print(f"{dataset_name} Evaluation Results:")
    print("BLEU score:", bleu_metric.compute())
    print("Exact Match score:", exact_match_metric.compute())
    print("CER score:", cer_metric.compute())

# model initialization
model_dir = "./finetuned-vit-bart"
model_path = get_latest_checkpoint_by_time(model_dir)
print(f"Initialized model from {model_path}")
model = VisionEncoderDecoderModel.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="latex_tokenizer.json",  # path to your tokenizer
    unk_token="<unk>",
    pad_token="<pad>",
    eos_token="</s>",
    bos_token="<s>"
)
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
cer_metric = load("cer")
# edit_distance_metric = load("edit_distance")

# dataloaders
test_full_dataloader = DataLoader(test_full_dataset, batch_size=16)
test_synthetic_dataloader = DataLoader(test_synthetic_dataset, batch_size=16)

evaluate(test_full_dataloader, "Full Test")
evaluate(test_synthetic_dataloader, "Synthetic Handwritten Test")