# DL_DA6401_ASSIGNMENT_1

# Fashion-MNIST Neural Network with WandB Logging

## Overview
This project implements a deep learning model to classify images from the Fashion-MNIST dataset using a custom-built neural network. The implementation includes various activation functions, optimizers, and weight initialization techniques. The project also integrates **Weights & Biases (WandB)** for logging and hyperparameter tuning via **Sweeps**.

## Features
- **Fashion-MNIST Data Processing**: Loads, normalizes, and visualizes the dataset.
- **Custom Neural Network**: Implements a feedforward neural network with multiple hidden layers.
- **Activation Functions**: Supports ReLU, Tanh, Sigmoid, and Softmax.
- **Optimizers**: Includes SGD, Momentum, Nesterov, RMSProp, Adam, and NAdam.
- **Weight Initialization**: Supports Random, Xavier, and Normal initialization.
- **Training & Evaluation**: Implements forward and backward passes, batch training, and model evaluation.
- **Hyperparameter Tuning**: Uses WandB Sweeps to optimize model parameters.
- **Performance Visualization**: Plots training/validation loss and accuracy curves.
- **Confusion Matrix**: Displays classification performance.

## Installation
To run the project, ensure you have the following dependencies installed:
```bash
pip install numpy keras sklearn matplotlib seaborn wandb
```

## Running the Project
### 1. Initialize WandB
Before running the script, log into Weights & Biases:
```bash
wandb login
```

### 2. Run the Script
Execute the Python script to train the model:
```bash
python train.py
```

### 3. Using WandB Sweeps
To start hyperparameter tuning using WandB Sweeps, use wandb sweep.

## Important Commands
- **Train Model**: `python dl_ma23m007_a1.py`
- **Log into WandB**: `wandb login`
- **Run Hyperparameter Tuning**:
  by using wandb sweeps
- **View WandB Logs**: Check your Weights & Biases dashboard.

## WandB Sweep Configuration
```yaml
sweep_config:
  method: bayes
  name: fashion_mnist_sweep
  metric:
    goal: maximize
    name: Validation Accuracy
  parameters:
    epochs:
      values: [5, 10]
    hidden_layers:
      values: [3, 4, 5]
    layer_size:
      values: [32, 64, 128, 256, 512]
    optimizer:
      values: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
    batch_size:
      values: [32, 64, 128]
    activation:
      values: ['sigmoid', 'tanh', 'relu']
    lr:
      values: [1e-3, 1e-4]
    weight_init:
      values: ['random', 'xavier', 'random_normal', 'xavier_normal', 'xavier_uniform']
    weight_decay:
      values: [0.0, 0.0001, 0.001, 0.0005]
```

## Results
- **Accuracy**: The model achieves high classification accuracy.
- **Confusion Matrix**: Visualizes misclassifications.
- **Training Progress**: Loss and accuracy curves available in WandB.

## Note: 
Use the train.py for running the code in command prompt which i have done without the wandb parameters.

wandb report: [link](https://wandb.ai/ma23m007-iit-madras/fashion_emnist_gross_entropy/reports/MA23M007_DA6401-Assignment-1--VmlldzoxMTgxNzEwOQ)
