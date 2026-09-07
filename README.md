# Precog Task: Spurious Correlations in CNNs

A comprehensive project exploring how Convolutional Neural Networks (CNNs) learn spurious correlations and can be fooled by color-label biases in image classification tasks.

## 📋 Overview

This repository contains a series of experiments investigating how CNNs exploit spurious correlations—specifically color biases—when training on biased datasets. The project demonstrates that models trained on data with strong color-digit correlations fail catastrophically when tested on color-unbiased data, highlighting the importance of understanding model robustness and bias.

### Key Concept

A Colored MNIST dataset is created where:
- **Biased split (95%)**: Specific digits are consistently colored (e.g., digit "1" is always red)
- **Unbiased split (5%)**: Digits appear in random colors
- Models trained on this biased data learn to rely heavily on color as a shortcut predictor, rather than learning true digit features

## 🎯 Tasks

### Task 0: ColoredMNIST Dataset Class (`task_0.py`)
Implements the `ColoredMNIST` PyTorch Dataset class that:
- Loads standard MNIST images
- Applies color transformations based on digit labels
- Creates a color-label correlation with configurable bias strength
- Supports both biased training and unbiased testing scenarios

**Key Features:**
- 10 distinct color palettes (one per digit)
- Configurable bias probability (`p_biased`)
- Realistic background noise with color-matched coloring

### Task 1: Basic CNN Training & Evaluation (`task_1.py`)
Trains a simple 3-layer CNN on the Colored MNIST dataset and demonstrates model bias.

**Architecture:**
- 3 convolutional layers (32→64→128 filters)
- 2 fully connected layers
- Max pooling for dimensionality reduction

**Evaluation Metrics:**
- Training accuracy on biased data
- Validation accuracy on biased data
- Hard test accuracy on unbiased data

**Outputs:**
- `red_1_color_dependency.png`: Visualization showing the model's color dependency
- `confusion_matrix.png`: Confusion matrix on unbiased test data

### Task 2: Advanced Analysis (`task_2.py`)
[Additional analysis tasks TBD]

### Tasks 3-6: Extended Experiments (`task_3.py` - `task_6.py`)
[Extended experimental tasks exploring robustness and debiasing techniques]

## 📊 Dataset

### ColoredMNIST Details

The dataset extends standard MNIST with color:

```
Input: 28×28 grayscale MNIST digit
↓
Apply color based on label + bias probability
↓
Output: 28×28 RGB image with colored digit and colored background
```

**Bias Mechanism:**
- With probability `p_biased`: digit gets its assigned color
- With probability `1 - p_biased`: digit gets a random "wrong" color
- Background noise is also color-matched for visual coherence

### Expected Performance Gap

A model trained on 95% biased data typically exhibits:

| Scenario | Accuracy |
|----------|----------|
| Training (biased colors) | ~95-99% |
| Validation (biased colors) | ~90-98% |
| Testing (unbiased colors) | ~10-30% |

This dramatic drop illustrates severe overfitting to spurious correlations.

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch >= 1.9.0
torchvision >= 0.10.0
numpy
matplotlib
scikit-learn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Harshavardhan894/Precog_task.git
cd Precog_task

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn
```

### Running the Experiments

**Download dataset and generate sample visualizations:**
```bash
python download_dataset.py
python save_fig_colour_dataset.py
```

**Train basic CNN and evaluate on biased/unbiased data:**
```bash
python task_1.py
```

This will output:
- Training, validation, and hard test accuracies
- `red_1_color_dependency.png` - Color bias visualization
- `confusion_matrix.png` - Model performance breakdown

## 📁 File Structure

```
Precog_task/
├── README.md                          # This file
├── cnn_Tasks.pdf                      # Task specifications document
├── task_0.py                          # ColoredMNIST dataset implementation
├── task_1.py                          # CNN training & evaluation
├── task_2.py                          # [Extended analysis]
├── task_3.py                          # [Extended experiments]
├── task_4.py                          # [Extended experiments]
├── task_4_2.py                        # [Extended experiments variant]
├── task_5.py                          # [Extended experiments]
├── task_6.py                          # [Extended experiments]
├── download_dataset.py                # Utility to download MNIST
├── save_fig_colour_dataset.py         # Utility to visualize dataset
└── data/                              # (Generated) MNIST data directory
```

## 🔬 Key Insights

### Why This Matters

1. **Shortcut Learning**: Models find the easiest path to minimize loss, exploiting spurious correlations
2. **Distribution Shift**: Excellent performance on training data doesn't guarantee real-world robustness
3. **Interpretability**: Understanding what features models actually use is crucial for deployment
4. **Debiasing**: This project motivates techniques for training more robust, generalizable models

### Model Behavior

When trained on 95% biased Colored MNIST, the model learns:
- **Primary pattern**: Color → Digit mapping (shortcut)
- **Secondary pattern**: Actual digit features (true labels)
- **Result**: Catastrophic failure when color-label correlation breaks

## 📈 Expected Outputs

After running `task_1.py`:

```
Training
------------------------------
Training Accuracy:   98.56%
Validation Accuracy: 96.43%
Hard Test Accuracy:  15.23%
------------------------------
Saving bias proof image and confusion matrix...
```

- **red_1_color_dependency.png**: Shows a digit "1" colored in red (its biased color), demonstrating what the model actually "sees"
- **confusion_matrix.png**: Reveals that the model's errors correlate with color confusion on unbiased data

## 🎓 Learning Objectives

By exploring this project, you'll understand:
- How CNNs can be fooled by spurious correlations
- The importance of dataset quality and representative samples
- Techniques for identifying and mitigating shortcut learning
- The gap between training and real-world performance
- Visualization methods for understanding model decisions

## 🔗 Related Concepts

- **Shortcut Learning**: [Geirhos et al., 2020](https://arxiv.org/abs/1905.13549)
- **Spurious Correlations in Machine Learning**
- **Distribution Shift & Domain Generalization**
- **Interpretability in Deep Learning**

## 💡 Future Extensions

Potential experiments to extend this work:
- Implement debiasing techniques (e.g., reweighting, data augmentation)
- Explore different CNN architectures (ResNet, Vision Transformers)
- Analyze learned feature representations
- Test transfer learning from unbiased pre-training
- Implement adversarial training for robustness

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0]

## 👤 Author

Harshavardhan894

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This project is part of the Precog task series. See `cnn_Tasks.pdf` for detailed specifications.
