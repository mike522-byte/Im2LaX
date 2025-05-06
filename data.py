from torchvision import transforms
from torchvision.transforms import functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class HuggingfaceLatexDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_transform, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['image'].convert("RGB")
        pix = self.image_transform(img)
        
        encoded = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = encoded.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
        attention_mask = encoded.attention_mask.squeeze(0)

        return {'pixel_values': pix, 'labels': labels, 'attention_mask': attention_mask} # pixel_values contain list of tensors


class PadToSquareThenResize:
    def __init__(self, target_size, fill=255, resample=Image.NEAREST):
        self.target_size = target_size
        self.fill = fill
        self.resample = resample

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be a PIL Image")

        w, h = img.size
        max_side = max(w, h)

        # Pad to square
        mode = img.mode  # e.g., "L" for grayscale or "RGB"
        new_img = Image.new(mode, (max_side, max_side), color=self.fill)
        paste_x = (max_side - w) // 2
        paste_y = (max_side - h) // 2
        new_img.paste(img, (paste_x, paste_y))

        # Resize
        return new_img.resize((self.target_size, self.target_size), resample=self.resample)

    def __repr__(self):
        return f"{self.__class__.__name__}(target_size={self.target_size}, fill={self.fill})"


class PadToSize:
    def __init__(self, target_width, target_height, fill=255):
        self.tw = target_width
        self.th = target_height
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        pad_left = (self.tw - w) // 2
        pad_top = (self.th - h) // 2
        pad_right = self.tw - w - pad_left
        pad_bottom = self.th - h - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

    def __repr__(self):
        return f"{self.__class__.__name__}(target_width={self.tw}, target_height={self.th})"


class Binarize:
    """
    Binarize a grayscale image using a fixed threshold.
    Should be applied before ResizeToSquarePad.
    """
    def __init__(self, threshold=200):
        self.threshold = threshold

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be a PIL Image")
        img = img.convert("L")
        img = img.point(lambda p: 0 if p < self.threshold else 255, mode='1')
        return img.convert("L")

    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold})"
    

class CropEmptyBottom:
    def __init__(self, padding=10):
        self.padding = padding #保留内容边缘的余量（防止误裁）

    def __call__(self, img):
        img_np = np.array(img)
        
        non_empty_rows = np.where(img_np.min(axis=1) == 0)[0]
        non_empty_cols = np.where(img_np.min(axis=0) == 0) [0]

        if len(non_empty_rows) == 0 & len(non_empty_cols) == 0:
            return img 
        
        top = max(0, non_empty_rows[0] - self.padding)
        bottom = min(img_np.shape[0], non_empty_rows[-1] + self.padding + 1)
        left = max(0, non_empty_cols[0] - self.padding)
        right = min(img_np.shape[1], non_empty_cols[-1] + self.padding + 1)

        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img

    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold}, padding={self.padding})"


def get_image_transform():
    return transforms.Compose([
        Binarize(threshold=200),
        CropEmptyBottom(padding=10),
        PadToSize(784, 64, fill=255),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def load_latex_dataset(name, split, data_path='linxy/LaTeX_OCR'):
    dataset = load_dataset(data_path, name=name, split=split)
    return dataset