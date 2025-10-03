# Project README

## How to Run in Colab(FOR q1)

1. **Open a new Colab Notebook**
2. **Install required packages**: Run the following cell to install dependencies:

```python
!pip install torch torchvision pytorch-ignite matplotlib
```

3. **Upload the Notebook**: Upload this repository's notebook to Colab.

4. **Set Runtime**: Ensure you have selected a GPU runtime, if available.

5. **Run the Cells**: Execute the notebook cells sequentially.

## Best Model Configuration

This project uses the following best configuration for the `RelViT` model on the CIFAR10 dataset:

- **Model**: RelViT
- **Dataset**: CIFAR10
- **Image Size**: 32
- **Channels**: 256
- **Head Channels**: 32
- **Number of Blocks**: 8
- **Patch Size**: 2
- **Embedding Dropout**: 0.0
- **Transformer Dropout**: 0.0
- **Head Dropout**: 0.3
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3
- **Weight Decay**: 1e-1
- **Data Augmentation**: TrivialAugment, RandomHorizontalFlip, RandomCrop, RandomErasing

## Results

Below is a summary of the overall classification test accuracy:

| Model Configuration | Overall Test Accuracy |
| ------------------- | --------------------- |
| RelViT (BEST CONFIG)| 95.9%                 |

*Note: The test accuracy is subject to change depending on the training run and seed configuration.*

