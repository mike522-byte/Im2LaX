from transformers import PreTrainedTokenizerFast, BartTokenizer
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from data import load_latex_dataset, get_image_transform, HuggingfaceLatexDataset
from datasets import concatenate_datasets


# Tokenizer (BPE Tokenizer)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# input images preprocessing
image_transform = get_image_transform()

# data preparation
full_data = load_latex_dataset("full", "train", data_path="linxy/LaTeX_OCR")
synthetic_data = load_latex_dataset("synthetic_handwrite", "train", data_path="linxy/LaTeX_OCR")
synthetic_half = synthetic_data.select(range(len(synthetic_data) // 2))
first_train_data = concatenate_datasets([full_data, synthetic_half])
train_dataset = HuggingfaceLatexDataset(first_train_data, tokenizer, image_transform)

# loading transformer models
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k",
    "facebook/bart-base",
    encoder_image_size=(64, 784)
)

# model configurations
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.vocab_size = len(tokenizer)
model.config.max_length = 600

# print("Model Config:")
# print(model.config.decoder_start_token_id)
# print(model.config.eos_token_id)
# print(model.config.pad_token_id)

# trainig arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-vit-bart/version-3",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,
    save_strategy="epoch",
    eval_strategy="no",
    predict_with_generate=True,
    report_to="none",
)    

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset
)

trainer.train()