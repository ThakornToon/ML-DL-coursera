# Machine Learning & Deep Learning Specialization in Java

This repository contains my personal implementations of selected labs from the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) and [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) on Coursera.

## Purpose
The original labs in these courses are taught using Python. However, I prefer working with **Java** and wanted to challenge myself by building these models and algorithms from scratch in Java. This approach helps me gain a deeper understanding of the core concepts, math, and logic without relying on high-level Python libraries (like NumPy, scikit-learn, or TensorFlow).

## Status
**Work in Progress (WIP)**  
This project is currently in active development.

## Current Implementations
Based on the current progress, the repository includes concepts such as:

- **Course 1: Supervised Machine Learning**
  - **Linear Regression** (`C1W2LinearRegression.java`)
  - **Logistic Regression** (`C1W3LogisticRegression.java`)
- **Course 2: Advanced Learning Algorithms**
  - **Neural Networks (Coffee Roasting)** (`C2W1Lab3CoffeeRoasting.java`)
- **Custom ML Tools (`ML.tools` Package)**
  - **Sequential API** (`Sequential.java`): Groups a linear stack of layers into a single model.
  - **Dense Layer** (`Dense.java`): Represents a fully connected neural network layer with forward/backward propagation.
  - **Activation Functions** (`Activation.java`): Supports `sigmoid`, `relu`, `linear`.
  - **Data Normalization** (`Normalization.java`): Z-score normalization for features.
  - **Model Weights I/O** (`ModelWeightsIO.java`): Unified system for saving and loading model weights (Linear, Logistic, and Neural Networks).

*(More algorithms and neural network models will be added in the future).*

## Data
The training data used in these implementations is based on the datasets provided in the official Coursera labs. To run and test these models with the exact same data, you can find the datasets within the respective lab materials on Coursera.

## How to Run
1. Clone this repository.
2. Open the project in your preferred Java IDE (such as **IntelliJ IDEA**, Eclipse, or VS Code).
3. Ensure you have the Java Development Kit (JDK) installed and configured for the project.
4. Run the specific `.java` class (e.g., `C1W2LinearRegression`) with `start` method for the lab you wish to try.
