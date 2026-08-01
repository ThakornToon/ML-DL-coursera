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
  - **Neural Networks for Handwritten Digit Recognition** (`C2W1Assignment.java`)
  - **Neural Networks for Multiclass Classification** (`C2W2Assignment.java`)
  - **Advice for Applying Machine Learning (Model Evaluation)** (`C2W3Assignment.java`)
- **Custom ML Tools (`ML.tools` Package)**
  - **Sequential API** (`Sequential.java`): Groups a linear stack of layers into a single model, supporting Binary and Sparse Categorical Crossentropy (with `from_logits` stability trick).
  - **Dense Layer** (`Dense.java`): Represents a fully connected neural network layer with forward/backward propagation.
  - **Activation Functions** (`Activation.java`): Supports `sigmoid`, `relu`, `softmax` (with full Jacobian derivatives), and `linear`.
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

---

# ML Tools - User Guide

This guide explains how to use the custom `ML.tools` package to build, train, and run neural networks from scratch in Java.

## 1. Creating a Model

The core of the framework is the `Sequential` class, which allows you to stack layers sequentially.

```java
import ML.tools.Sequential;
import ML.tools.Dense;

// 1. Initialize the model
Sequential model = new Sequential();
```

## 2. Adding Layers

Use the `Dense` class to create fully connected layers. You must specify the number of neurons, the activation function, and an optional name.

```java
// Syntax: new Dense(units, activation_name, layer_name)
model.add(new Dense(25, "relu", "hidden_layer_1"));
model.add(new Dense(15, "relu", "hidden_layer_2"));
```

### Supported Activations
* `"relu"`: Standard for hidden layers (prevents vanishing gradients).
* `"leaky_relu"`: Variation of ReLU that allows a small, non-zero gradient when the unit is not active (alpha = 0.01).
* `"sigmoid"`: Used for binary classification output (0 or 1).
* `"softmax"`: Used for multi-class probability outputs.
* `"linear"`: No activation applied.

> [!IMPORTANT]
> **The `from_logits` Rule for Output Layers:**
> When doing **Multi-class Classification**, we strongly recommend setting the final layer to `"linear"` during training. 
> ```java
> model.add(new Dense(10, "linear", "output_layer")); // Best practice for training
> ```
> The `Sequential.fit()` method automatically detects multi-class outputs and combines the **Softmax** activation with the **Cross-Entropy Loss** calculation (LogSumExp trick). This prevents division-by-zero (`NaN` or `Infinity`) errors during backpropagation.
> 
> You *can* set it to `"softmax"`, but you may experience numerical instability if the model makes extremely incorrect predictions early in training.

## 3. Compiling the Model

Before training, configure the optimizer and learning rate. Currently, the `"adam"` optimizer is supported.

```java
// Syntax: model.compile(optimizer_name, learning_rate)
model.compile("adam", 0.001);
```

## 4. Training (Fitting)

Train the model using your input features `X` and target labels `y`.
* `X` shape: `[samples][features]`
* `y` shape: `[samples][1]` (Even for multi-class, `y` should contain the integer class index, e.g., `0` to `9`).

```java
int epochs = 40;
model.fit(X, y, epochs);
```
During training, the loss will be printed to the console periodically.

## 5. Inference (Prediction)

To make predictions on new data, use the `predict()` method.

```java
double[][] newData = { X[0] }; // Must be a 2D array
double[][] prediction = model.predict(newData);
```

> [!NOTE]
> If you used `"linear"` as your output layer for training, the prediction will return raw **Logits**. To convert these to percentages/probabilities, pass the result through the Softmax function manually:
> ```java
> double[] probabilities = ML.tools.Activation.softmax(prediction[0]);
> ```

## 6. Model Summary

To view the architecture and the total number of parameters in your network, call:
```java
model.summary();
```

## 7. Saving and Loading Weights

You can export the trained weights and biases to text files so you don't have to retrain the model later.

```java
import ML.tools.ModelWeightsIO;

// Retrieve the layer by name
Dense layer1 = model.getLayer("hidden_layer_1");

// --- Save to file ---
if (layer1 != null) {
    ModelWeightsIO.saveWeights("ml_layer1_weights.txt", layer1.getWeightsW(), layer1.getWeightsB());
}

// --- Load from file ---
if (layer1 != null) {
    Object[] loaded = ModelWeightsIO.loadDenseWeights("ml_layer1_weights.txt");
    if (loaded != null) {
        double[][] W = (double[][]) loaded[0];
        double[] b = (double[]) loaded[1];
        layer1.setWeights(W, b); // Inject loaded weights into the layer
    }
}
```
