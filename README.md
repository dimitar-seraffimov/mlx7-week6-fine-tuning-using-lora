# mlx7-week6-fine-tuning-using-lora

An introduction to Parameter-Efficient Fine-Tuning using LoRA.

Week 6 is all about understanding Low-Rank Adaptation of Large Language Models (LoRA) and how to practically apply it. <br>
I have spent Monday and Tuesday researching, reading and getting a better understanding of the technology and picking a task to tackle.
References to interesting papers, GitHub repos and articles:
| Title | Link |
|----------------------------------------------------|-------------------------------------------------|
| LoRA: Low-Rank Adaptation of Large Language Models | https://arxiv.org/pdf/2106.09685 |
| High-Resolution Image Synthesis with Latent Diffusion Models | https://arxiv.org/pdf/2112.10752 |
| Toolformer: Language Models Can Teach Themselves to Use Tools | https://arxiv.org/pdf/2302.04761 |
| Internet-Augmented Dialogue Generation | https://arxiv.org/pdf/2107.07566 |
| Using LoRA for Efficient Stable Diffusion Fine-Tuning | https://huggingface.co/blog/lora |
| Diffusion Explainer | https://poloclub.github.io/diffusion-explainer/ |
| Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning | https://github.com/cloneofsimo/lora |
| Tom and Jerry Image Classification | https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification |
| Phycics of Language Models | https://physics.allen-zhu.com/home |
| RADLADS: Rapid Attention Distillation to Linear Attention De-
coders at Scale | https://arxiv.org/pdf/2505.03005 |
| The Illustrated Stable Diffusion | https://jalammar.github.io/illustrated-stable-diffusion/ |

# Tom and Jerry Universe Image Generation with LoRA

Fine-tuning a pre-trained Stable Diffusion model using **Low-Rank Adaptation (LoRA)** to generate images in a 'Tom and Jerry' universe style.

## Overview

The main goal is to:

- fine-tune a Stable Diffusion model using LoRA
- generate images that incorporate the visual style and characteristics of the Tom and Jerry cartoons

## Project Structure

```
├── data/
│   ├── train/
│   └── validation/
├── notebooks/
│   └── fine_tuning.ipynb --- working on that only for now!
├── scripts/
│   ├── setup_environment.sh
│   └── fine_tune.py
├── outputs/
│   ├── pretrained_outputs/
│   └── fine_tuned_outputs/
├── requirements.txt
└── README.md
```

## Dataset

The dataset consists of images from the Tom and Jerry cartoon, available [here](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification).

- Download and prepare the dataset:

```bash
kaggle datasets download balabaskar/tom-and-jerry-image-classification
unzip tom-and-jerry-image-classification.zip -d data/
```

## Fine-Tuning Approach

**Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)

**Method**: LoRA-based fine-tuning

- insert LoRA adapters into the cross-attention layers.
- freeze the original model weights; only update LoRA parameters during training.

## Results

All results, including generated images and evaluation metrics, will be available in the `outputs/` directory.
