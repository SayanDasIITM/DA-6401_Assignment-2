
```markdown
# DA6401 Assignment 2: CNN from Scratch and Fine-Tuning with Wandb Integration

This repository contains an implementation of two deep learning approaches for the DA6401 assignment:
1. **Part A:** Training a custom Convolutional Neural Network (CNN) from scratch on the iNaturalist dataset.
2. **Part B:** Fine-tuning a pre-trained model (from torchvision) using various fine-tuning strategies.

In addition, the code includes:
- Theoretical computation of FLOPs and parameter count.
- Visualization functions for filters, test predictions, and (optionally) guided backpropagation for selected neurons.
- Integration with [Weights & Biases (wandb)](https://wandb.ai) for experiment tracking and hyperparameter sweeps.

## Table of Contents
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Code Structure](#code-structure)
- [Usage](#usage)
  - [Running Part A (Custom CNN)](#running-part-a-custom-cnn)
  - [Running Part B (Fine-Tuning)](#running-part-b-fine-tuning)
  - [Guided Backpropagation Visualization](#guided-backpropagation-visualization)
  - [Wandb Sweeps](#wandb-sweeps)
- [Visualizations](#visualizations)
- [Report and Analysis](#report-and-analysis)
- [License](#license)

## Requirements
- Python 3.7 or higher
- PyTorch
- Torchvision
- Wandb
- Matplotlib
- Numpy

## Dataset Setup
The code assumes that the iNaturalist dataset is organized in the following folder structure:
```

./inat/train/<class_name>/...
./inat/test/<class_name>/...

```

Place your dataset accordingly before running the code.

## Installation
1. **Clone the repository:**
```

git clone https://github.com/<your-username>/da6401_assignment2.git
cd da6401_assignment2

```

2. **Create and activate a virtual environment (optional but recommended):**
```

python -m venv venv
source venv/bin/activate  \# On Windows: venv\Scripts\activate

```

3. **Install required packages:**
```

pip install torch torchvision wandb matplotlib numpy

```

4. **Log in to wandb:**
```

wandb login

```

## Code Structure
- **assignment_code.py**:
Main script that implements both Part A (custom CNN) and Part B (fine-tuning) with comprehensive visualization and theoretical computation functions.

- **README.md**:
This documentation file.

The code is organized into the following sections:

- **Model Definitions**:
CustomCNN and functions to load pre-trained models.

- **Theoretical Computations**:
Functions to compute FLOPs and parameter count.

- **Data Preparation**:
Functions to load the iNaturalist dataset using the ImageFolder structure.

- **Visualizations**:
Functions to visualize filters, test predictions grid, and guided backpropagation.

- **Training & Evaluation**:
Functions to train and evaluate the model.

- **Wandb Integration & Sweeps**:
Code is integrated with wandb using wandb.config. A YAML sweep configuration template is included in the comments.

## Usage

### Running Part A (Custom CNN)
To train a custom CNN from scratch (Part A):

```

python assignment_code.py --part partA

```

You can also specify additional hyperparameters; for example:

```

python assignment_code.py --part partA --epochs 20 --batch_size 64 --lr 0.001 --num_conv_blocks 5 --filters 32 64 128 256 512 --activations relu relu relu relu relu --dense_neurons 256

```

### Running Part B (Fine-Tuning)
To fine-tune a pre-trained model (e.g., ResNet50) with a chosen fine-tuning strategy (e.g., "last_two"):

```

python assignment_code.py --part partB --model_name resnet50 --finetune_strategy last_two

```

Available pre-trained models include:
- resnet18
- resnet50
- vgg16
- efficientnet_b0
- googlenet
- inception_v3
- vit_base_patch16_224

And available fine-tuning strategies:
- last (freeze all layers except the final classifier)
- last_two (for ResNet-like models, unfreeze the last block and final classifier)
- all (fine-tune the entire network)

### Guided Backpropagation Visualization
To generate guided backpropagation visualizations for exactly 10 neurons from the last convolutional layer (CONV5) of the custom CNN, run:

```

python assignment_code.py --part partA --guided_backprop

```

This will save a visualization image (guided_backprop.png) showing gradients for 10 selected neurons.

### Wandb Sweeps
The code supports hyperparameter sweeps via wandb. A sample sweep configuration template is provided in the code comments. To run a sweep:

1. Create a YAML file (e.g., sweep_config.yaml) using the provided template.

2. Start the sweep:
```

wandb sweep sweep_config.yaml

```

3. Launch an agent with the returned sweep ID:
```

wandb agent <sweep_id>

```

The sweep will override hyperparameters using wandb.config.

## Visualizations
The code automatically saves the following visualizations:

- **Filter Visualization**:
Saves the first layer filters to filters.png.

- **Test Predictions Grid**:
Saves a 10×3 grid of test images with predictions to test_predictions.png.

- **Guided Backpropagation (Optional)**:
If enabled via --guided_backprop, saves the guided backpropagation visualization as guided_backprop.png.

## Report and Analysis
All metrics and model statistics are logged to wandb. Use the wandb dashboard to generate:
- Accuracy vs. Created plots.
- Parallel coordinates plots.
- Correlation summary tables.

Include these plots in your report along with your insights and analysis as required by the assignment.

## License
This project is provided for educational purposes. Please cite appropriately if you reuse any part of the code.
```

