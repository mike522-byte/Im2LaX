from data import load_latex_dataset, get_image_transform, HuggingfaceLatexDataset
from transformers import ViTModel, BartForConditionalGeneration, BartTokenizer
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

data = load_latex_dataset("full", "train[:5]", data_path="linxy/LaTeX_OCR")
test_data = HuggingfaceLatexDataset(data, tokenizer, get_image_transform())

print(test_data[0]['pixel_values'].shape)  # Check the shape of the tensor
img_np = test_data[0]['pixel_values'].squeeze(0).numpy()
img = np.transpose(img_np, (1, 2, 0))  # Transpose to (H, W, C) format for visualization
plt.imshow(img)
plt.axis('off')
plt.show()



