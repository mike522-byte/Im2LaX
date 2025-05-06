from data import load_latex_dataset, get_image_transform, HuggingfaceLatexDataset

full_data = load_latex_dataset("full", "train", data_path="linxy/LaTeX_OCR")
with open("equations.txt", "w", encoding="utf-8") as f:
    for item in full_data:
        f.write(item["text"] + "\n")  # One equation per line