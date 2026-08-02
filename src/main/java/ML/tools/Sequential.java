package ML.tools;

import java.util.ArrayList;
import java.util.List;

/**
 * A Sequential model groups a linear stack of layers into a single neural network.
 * It provides methods for compiling, training, and predicting.
 */
public class Sequential {
    private List<Dense> layers;
    private String optimizer = "sgd";
    private double learningRate = 0.01;

    public Sequential() {
        this.layers = new ArrayList<>();
    }

    /**
     * Compile the model with an optimizer and learning rate.
     */
    public void compile(String optimizer, double learningRate) {
        this.optimizer = optimizer;
        this.learningRate = learningRate;
    }

    /**
     * Adds a layer to the model.
     *
     * @param layer The Dense layer instance to add.
     */
    public void add(Dense layer) {
        this.layers.add(layer);
    }

    public Dense getLayer(String name) {
        for (Dense layer : layers) {
            if (layer.getName() != null && layer.getName().equals(name)) {
                return layer;
            }
        }
        return null;
    }
    
    public List<Dense> getLayers() {
        return layers;
    }

    /**
     * Prints a summary representation of the model architecture,
     * including layer types, output shapes, and parameter counts.
     */
    public void summary() {
        System.out.println("Model: \"sequential\"");
        System.out.println("_________________________________________________________________");
        System.out.printf("%-30s %-15s %-15s%n", "Layer (type)", "Output Shape", "Param #");
        System.out.println("=================================================================");
        
        int totalParams = 0;
        for (Dense layer : layers) {
            String type = layer.getClass().getSimpleName();
            String nameAndType = (layer.getName() != null ? layer.getName() : "layer") + " (" + type + ")";
            int params = layer.getParamCount();
            String outputShape = "(None, " + layer.getUnits() + ")";
            
            System.out.printf("%-30s %-15s %-15d%n", nameAndType, outputShape, params);
            totalParams += params;
        }
        
        System.out.println("=================================================================");
        System.out.println("Total params: " + totalParams);
        System.out.println("Trainable params: " + totalParams);
        System.out.println("Non-trainable params: 0");
        System.out.println("_________________________________________________________________");
    }

    public void save(String directoryPath) {
        save(directoryPath, "");
    }

    public void save(String directoryPath, String prefix) {
        java.io.File dir = new java.io.File(directoryPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        String prefixStr = (prefix == null || prefix.isEmpty()) ? "" : prefix + "_";
        for (int i = 0; i < layers.size(); i++) {
            Dense layer = layers.get(i);
            if (layer.isInitialized()) {
                String name = layer.getName() != null ? layer.getName() : "layer" + (i + 1);
                String filePath = directoryPath + "/" + prefixStr + name + "_weights.csv";
                ModelWeightsIO.saveWeights(filePath, layer.getWeightsW(), layer.getWeightsB());
            } else {
                System.out.println("Skipping uninitialized layer: " + (layer.getName() != null ? layer.getName() : "layer" + (i + 1)));
            }
        }
        System.out.println("Model saved successfully to directory: " + directoryPath);
    }

    public void load(String directoryPath) {
        load(directoryPath, "");
    }

    public void load(String directoryPath, String prefix) {
        String prefixStr = (prefix == null || prefix.isEmpty()) ? "" : prefix + "_";
        for (int i = 0; i < layers.size(); i++) {
            Dense layer = layers.get(i);
            String name = layer.getName() != null ? layer.getName() : "layer" + (i + 1);
            String filePath = directoryPath + "/" + prefixStr + name + "_weights.csv";
            java.io.File file = new java.io.File(filePath);
            if (file.exists()) {
                Object[] loaded = ModelWeightsIO.loadDenseWeights(filePath);
                if (loaded != null) {
                    double[][] W = (double[][]) loaded[0];
                    double[] b = (double[]) loaded[1];
                    layer.setWeights(W, b);
                }
            } else {
                System.err.println("Warning: Weight file not found for layer " + name + " at " + filePath);
            }
        }
        System.out.println("Model loaded successfully from directory: " + directoryPath);
    }

    public double[] forward(double[] x) {
        double[] a = x;
        for (Dense layer : layers) {
            a = layer.forward(a);
        }
        return a;
    }

    /**
     * Generates output predictions for the input samples.
     *
     * @param X The 2D array of input features (samples x features).
     * @return A 2D array of predictions (samples x outputs).
     */
    public double[][] predict(double[][] X) {
        int m = X.length;
        double[][] predictions = new double[m][];
        
        for (int i = 0; i < m; i++) {
            predictions[i] = forward(X[i]);
        }
        
        return predictions;
    }

    /**
     * Train the model using Binary Cross Entropy Loss.
     */
    public void fit(double[][] X, double[][] Y, int epochs) {
        int m = X.length;  // Train size
        int t = 1; // Step counter for Adam

        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Forward pass
            double[][] A = X;
            for (Dense layer : layers) {
                A = layer.forwardBatch(A);
            }

            // Compute Loss (Binary Cross Entropy or Sparse Categorical Crossentropy)
            double loss = 0.0;
            int outputSize = A[0].length;
            double[][] dA = new double[m][outputSize];
            
            if (outputSize == 1) {
                String finalActivation = layers.get(layers.size() - 1).getActivation();
                if (finalActivation.equalsIgnoreCase("linear")) {
                    // Mean Squared Error (MSE) -> Dense(1, "linear")
                    for (int i = 0; i < m; i++) {
                        double a = A[i][0];
                        double y = Y[i][0];
                        
                        loss += Math.pow(a - y, 2) / 2.0;
                        
                        // Derivative of MSE Loss w.r.t Activation:
                        dA[i][0] = (a - y);
                    }
                } else {
                    // Binary Cross Entropy -> Dense(1, "sigmoid")
                    for (int i = 0; i < m; i++) {
                        double a = A[i][0];
                        double y = Y[i][0];
                        
                        // Clip a to avoid log(0) and division by zero
                        a = Math.max(Math.min(a, 1.0 - 1e-15), 1e-15);
                        
                        loss += -y * Math.log(a) - (1 - y) * Math.log(1 - a);
                        
                        // Derivative of BCE Loss w.r.t Activation:
                        dA[i][0] = (a - y) / (a * (1.0 - a));
                    }
                }
            } else {
                String finalActivation = layers.get(layers.size() - 1).getActivation();
                
                if (finalActivation.equalsIgnoreCase("softmax")) {
                    // Sparse Categorical Crossentropy (from_logits=False)
                    // The output A is already a probability distribution from Softmax.
                    for (int i = 0; i < m; i++) {
                        int y = (int) Y[i][0]; // true class
                        double prob = Math.max(A[i][y], 1e-15);
                        loss += -Math.log(prob);
                        
                        // Derivative: dL/dA.
                        for (int j = 0; j < outputSize; j++) {
                            if (j == y) {
                                dA[i][j] = -1.0 / prob;
                            } else {
                                dA[i][j] = 0.0;
                            }
                        }
                    }
                } else {
                    // Sparse Categorical Crossentropy (from_logits=True)
                    // The output A contains raw logits (linear activation).
                    // Computing Softmax internally here with Cross-Entropy (LogSumExp) provides maximum numerical stability.
                    for (int i = 0; i < m; i++) {
                        int y = (int) Y[i][0]; // true class
                        
                        // Softmax computation (stable)
                        double maxLogit = -Double.MAX_VALUE;
                        for (int j = 0; j < outputSize; j++) {
                            if (A[i][j] > maxLogit) maxLogit = A[i][j];
                        }
                        double sumExp = 0.0;
                        double[] probs = new double[outputSize];
                        for (int j = 0; j < outputSize; j++) {
                            probs[j] = Math.exp(A[i][j] - maxLogit);
                            sumExp += probs[j];
                        }
                        for (int j = 0; j < outputSize; j++) {
                            probs[j] /= sumExp;
                        }
                        
                        loss += -Math.log(Math.max(probs[y], 1e-15));
                        
                        // Derivative: dL/dA. 
                        // Since the last layer should be 'linear' (g'(Z) = 1), dL/dZ = dL/dA * 1
                        // For Softmax + CE, dL/dZ = (probs - true_y)
                        for (int j = 0; j < outputSize; j++) {
                            if (j == y) {
                                dA[i][j] = probs[j] - 1.0;
                            } else {
                                dA[i][j] = probs[j];
                            }
                        }
                    }
                }
            }
            loss /= m;
            
            double regLoss = 0.0;
            for (Dense layer : layers) {
                if (layer.getLambdaL2() > 0) {
                    double layerReg = 0;
                    double[][] W = layer.getWeightsW();
                    for (int k = 0; k < W.length; k++) {
                        for (int j = 0; j < W[0].length; j++) {
                            layerReg += W[k][j] * W[k][j];
                        }
                    }
                    regLoss += (layer.getLambdaL2() / (2.0 * m)) * layerReg;
                }
            }
            loss += regLoss;
            
            if (epoch % 100 == 0 || epoch == 1 || epoch == epochs) {
                System.out.printf("Epoch %d/%d - loss: %.4f%n", epoch, epochs, loss);
            }

            // Backward pass to adjust weight and bias
            for (int l = layers.size() - 1; l >= 0; l--) {
                dA = layers.get(l).backward(dA, learningRate, optimizer, t);
            }
            t++;
        }
    }
}
