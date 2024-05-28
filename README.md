# Symbolic Integration with Neural Networks

This project aims to train a neural network to predict the original mathematical function given its derivative. The project involves generating training and validation data, training the model, evaluating its performance, and plotting the results.

## Project Overview

### Steps Involved

1. **Generating Training Data**:
   - We generate random mathematical functions and their derivatives using SymPy.
   - Functions include polynomials, trigonometric, exponential, and logarithmic functions.
   - Data is saved to a CSV file for easy access.

2. **Tokenizer Creation**:
   - A custom tokenizer is created to handle mathematical expressions.
   - The tokenizer is trained on sample mathematical functions and saved for future use.

3. **Model Training**:
   - An encoder-decoder model is trained on the generated data using the transformers library.
   - Training loss is monitored and plotted to observe the model's learning progress.

4. **Model Evaluation**:
   - The trained model is evaluated using a separate validation dataset.
   - Validation loss is calculated to assess model performance.

5. **Prediction**:
   - The model is used to predict the original function given a derivative.
   - Results are compared to the expected output to gauge accuracy.

6. **Plotting Results**:
   - Training loss over epochs is plotted to visualize the learning trend.
   - Predictions are plotted to show the model's performance.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- numpy
- pandas
- matplotlib
- sympy
- torch
- transformers

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib sympy torch transformers