# Project Title: Im2LaX - End-to-End LaTeX Formula Recognition via Vision-Language Modeling

## Project Overview
Im2LaX is an end-to-end formula recognition system that translates rendered mathematical formula images into LaTeX code. This project explores state-of-the-art vision-language (VL) model architectures for structured text generation, targeting precise formula transcription in academic and technical contexts.

## Key Contributions

### 1. Model Architecture Exploration
- Conducted extensive research on VL models including Pix2Seq, Donut, TrOCR.
- Analyzed and incorporated ideas from existing formula OCR systems such as LaTeX-OCR and Mathpix.
- Designed multiple encoder-decoder model combinations:
  - Vision encoders: ViT (Vision Transformer), Swin Transformer.
  - Language decoders: BART, GPT-family.

### 2. Supervised Fine-tuning and Ablation Studies
- Implemented supervised fine-tuning pipelines across models to evaluate their performance.
- Currently fine-tuning Qwen2.5-VL (3B) to assess its capacity in understanding and generating LaTeX.
- Conducted ablation studies to compare encoder-decoder configurations and understand trade-offs in accuracy and inference speed.

### 3. Evaluation Metrics
- **Textual Similarity Metrics**: BLEU, Edit Distance, Exact Match.
- **Structural Accuracy Metrics**: Subtree match rate between predicted LaTeX Abstract Syntax Tree (AST) and ground truth.
- **Baseline Comparison**: Evaluated zero-shot performance of non-fine-tuned large vision-language models as baselines.

### 4. Future Plans
- Explore reinforcement learning with DeepSeek-RL1 and implement GRPO (Generative Reinforcement Policy Optimization) to further enhance LaTeX output fidelity.
- Extend the system to noisy or handwritten formula images and evaluate robustness.

## Technical Stack
- **Frameworks**: PyTorch, Hugging Face Transformers, DeepSpeed
- **Models**: ViT, Swin, BART, GPT, Qwen2.5-VL
- **Datasets**: LaTeX-OCR, Custom Rendered Formulas
- **Tools**: Python, SLURM

## Objective
To push the boundary of structured document understanding by building a robust, modular, and accurate image-to-LaTeX system that can be applied to educational technology, scientific digitization, and math-aware language models.
