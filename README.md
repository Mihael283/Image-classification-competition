# Image Classification Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Data Augmentation](#data-augmentation)
7. [Training Process](#training-process)
8. [Logging and Monitoring](#logging-and-monitoring)
9. [Performance Metrics](#performance-metrics)
10. [Best model logs](#best-model-logs)
## Project Overview
This project implements a Convolutional Neural Network (CNN) for image classification. It includes custom data augmentation techniques, a parallel network architecture, and logging functionality.

On top of that there is more than 10 different model architectures that can be used for training. The models are optimized for 100x100 images in grayscale (1 channel). Training dataset is not publicaly available, but it is possible to use any dataset that is in the same format as the one used in this project.

Best result was achieved with the yolo_v3 model, which achieved 86% accuracy on the validation set and 81% accuracy on secret test set. Result placed me in the top 10% of the competitors(300+).

## Installation
To set up the project environment:

NOTE: Python version 3.8.10 was used.

- Setup the environment:
```python -m venv env```
- Activate the environment:
```source env/bin/activate```
- Install the required packages:
```pip install -r requirements.txt```

## Project Structure
- `architecture.py`: Main script for model training and evaluation
- `classes/parallel_net.py`: Contains the CNN model architecture
- `dataset.py`: Custom dataset class and augmentation functions
- `logger.py`: Logging utilities for training progress and results

## Usage
To train the model:

- Change the path to training data by modifiying the `train_dir` variable in `train.py`
- Run the script:
```python train.py```

## Model Architecture
The model uses a custom CNN architecture (`MyCNN`) with the following key features:
- Multiple convolutional layers with batch normalization and ReLU activation
- Parallel convolutional branches for multi-scale feature extraction
- Attention mechanism
- Global average pooling and fully connected layers for classification

## Data Augmentation
The project implements various augmentation techniques including:
- Random horizontal and vertical flips
- Random rotations
- Brightness and contrast adjustments
- Gaussian blur
- Random erasing
- Black bar augmentation

## Training Process
- Uses Adam optimizer with weight decay
- Implements learning rate scheduling with CosineAnnealingLR
- Employs early stopping to prevent overfitting
- Utilizes mixed precision training for improved performance

## Logging and Monitoring
- Comprehensive logging of training and validation metrics
- Per-class accuracy tracking
- Confusion matrix generation
- Best model checkpointing

## Performance Metrics
The model's performance is evaluated using:
- Training and validation loss
- Overall accuracy
- Per-class accuracy



