from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from data import load_latex_dataset, get_image_transform, HuggingfaceLatexDataset
from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# First: Split LaTeX commands and symbols
latex_pattern = r"\\[a-zA-Z]+\*?|\\[^a-zA-Z\s]"
latex_split = Split(
    pattern = latex_pattern,
    behavior = "isolated",
)

# Second: Split whitespace (optional, if you want to keep spaces separate)
whitespace_split = WhitespaceSplit()

# Combine them in sequence
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    latex_split,
    whitespace_split
])

# training the tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=2048,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    min_frequency=1,
)

tokenizer.train(files=["equations.txt"], trainer=trainer)  # Your LaTeX dataset
tokenizer.save("latex_tokenizer.json")

custom_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="latex_tokenizer.json",
    bos_token="[BOS]",
    eos_token="[EOS]",
    pad_token="[PAD]",
    unk_token="[UNK]",
)

print(custom_tokenizer.tokenize(r"V ( f ) = \{ \mathbf { z } \in \left( \mathbf { C } ^ { \ast } \right) ^ { \mathbf { k } } \mid f ("))