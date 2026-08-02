package ML.tools;

import java.util.Random;

/**
 * Represents a fully connected (Dense) layer in a Neural Network.
 * Implements forward and backward propagation steps for this layer.
 */
public class Dense {
    private double[][] W; // W shape: (inputSize, units)
    private double[] b;   // b shape: (units)
    private int units;
    private String activation;
    private double lambdaL2 = 0.0;
    private String name;
    private boolean initialized = false;

    // Cache for backpropagation
    private double[][] A_in;
    private double[][] Z;
    private double[][] A_out;

    // Adam Optimizer State
    private double[][] mW;
    private double[][] vW;
    private double[] mb;
    private double[] vb;
    private static final double BETA1 = 0.9;
    private static final double BETA2 = 0.999;
    private static final double EPSILON = 1e-8;

    /**
     * Constructs a Dense layer.
     *
     * @param units      The number of neurons (dimensionality of the output space).
     * @param activation The activation function to use (e.g., "relu", "sigmoid", "linear").
     * @param name       The unique name for this layer.
     */
    public Dense(int units, String activation, String name) {
        this.units = units;
        this.activation = activation;
        this.name = name;
    }

    public Dense(int units, String activation, double lambdaL2, String name) {
        this.units = units;
        this.activation = activation;
        this.lambdaL2 = lambdaL2;
        this.name = name;
    }

    public void setWeights(double[][] W, double[] b) {
        this.W = W;
        this.b = b;
        this.initialized = true;
    }
    
    public double[][] getWeightsW() {
        return W;
    }
    
    public double getLambdaL2() {
        return lambdaL2;
    }
    
    public double[] getWeightsB() {
        return b;
    }

    public String getName() {
        return name;
    }
    
    public boolean isInitialized() {
        return initialized;
    }

    public int getParamCount() {
        if (!initialized) return 0;
        int inputSize = W.length;
        return (inputSize * units) + units;
    }

    public int getUnits() {
        return this.units;
    }

    public String getActivation() {
        return this.activation;
    }

    private void initWeights(int inputSize) {
        this.W = new double[inputSize][units];
        this.b = new double[units];
        Random rng = new Random();
        
        double scale;
        if (this.activation.equalsIgnoreCase("relu") || this.activation.equalsIgnoreCase("leaky_relu")) {
            // He Initialization: Specifically designed for ReLU and Leaky ReLU.
            // Since ReLU zeros out half of the values (the negative side), the variance is halved.
            // We scale by sqrt(2/inputSize) to double the variance and prevent vanishing gradients.
            scale = Math.sqrt(2.0 / inputSize);
        } else {
            // Xavier/Glorot Normal Initialization: Best for Sigmoid, Softmax, or Linear.
            // It scales the weights by sqrt(2/(inputSize + outputSize)) to keep the variance of 
            // activations and gradients consistent across all layers during forward and backward passes.
            scale = Math.sqrt(2.0 / (inputSize + units));
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < units; j++) {
                this.W[i][j] = rng.nextGaussian() * scale;
            }
        }
        this.initialized = true;
    }

    /**
     * Performs a forward pass for a single input sample.
     *
     * @param aIn The input array for this layer.
     * @return The activated output array.
     */
    public double[] forward(double[] aIn) {
        if (!initialized) {
            initWeights(aIn.length);
        }

        // Each unit in this layer
        double[] zOut = new double[units];
        for (int j = 0; j < units; j++) {
            double z = 0.0;

            // Each feature from previous layer
            for (int i = 0; i < aIn.length; i++) {
                z += W[i][j] * aIn[i];
            }
            z += b[j];
            zOut[j] = z;
        }
        return Activation.apply(zOut, activation);
    }

    /**
     * Performs a forward pass for a batch of input samples.
     * Caches inputs, pre-activation, and post-activation values for backpropagation.
     *
     * @param input The 2D input array for this layer (samples x features).
     * @return The 2D activated output array.
     */
    public double[][] forwardBatch(double[][] input) {
        if (!initialized) {
            initWeights(input[0].length);
        }
        int m = input.length;
        this.A_in = input;
        this.Z = new double[m][units];
        this.A_out = new double[m][units];

        // Each input data
        for (int i = 0; i < m; i++) {
            // Each node in this layer
            for (int j = 0; j < units; j++) {
                double z = 0.0;
                // Each feature weight in this node
                for (int k = 0; k < input[i].length; k++) {
                    z += W[k][j] * input[i][k];
                }
                z += b[j];
                this.Z[i][j] = z;
            }
            // A_out = g(Z) for each data
            this.A_out[i] = Activation.apply(this.Z[i], activation);
        }
        return this.A_out;
    }

    public double[][] backward(double[][] dA, double learningRate, String optimizer, int t) {
        int m = dA.length;
        int inputSize = W.length;
        
        // 1. Calculate dZ
        double[][] dZ = new double[m][units];  // Accumulated error at that specific node before pass to the activation function.
        for (int i = 0; i < m; i++) {
            // Calculate dZ using the array-based method to support full Softmax Jacobian
            dZ[i] = Activation.compute_dZ(dA[i], A_out[i], Z[i], activation);
        }

        // 2. Calculate dW and db
        double[][] dW = new double[inputSize][units];
        double[] db = new double[units];

        // dL/dW = dL/dZ * dZ/dW (average in each node)
        for (int k = 0; k < inputSize; k++) {
            for (int j = 0; j < units; j++) {
                double sum = 0;
                for (int i = 0; i < m; i++) {
                    sum += A_in[i][k] * dZ[i][j];
                }
                dW[k][j] = (sum / m) + (lambdaL2 / m) * W[k][j];
            }
        }

        // dL/db = dL/dZ * dZ/db (average in each node)
        for (int j = 0; j < units; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) {
                sum += dZ[i][j];
            }
            db[j] = sum / m;
        }

        // 3. Calculate dA_prev to pass back to the previous layer
        double[][] dA_prev = new double[m][inputSize];
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < inputSize; k++) {
                double sum = 0;
                for (int j = 0; j < units; j++) {
                    sum += dZ[i][j] * W[k][j];
                }
                // dA_in = dA_prev_out = dL/dA_in = dL/dZ * dZ/dA_in
                dA_prev[i][k] = sum;
            }
        }

        // 4. Update Weights
        if (optimizer != null && optimizer.equalsIgnoreCase("adam")) {
            applyAdam(dW, db, learningRate, t);
        } else {
            // Standard Gradient Descent
            for (int k = 0; k < inputSize; k++) {
                for (int j = 0; j < units; j++) {
                    W[k][j] -= learningRate * dW[k][j];
                }
            }
            for (int j = 0; j < units; j++) {
                b[j] -= learningRate * db[j];
            }
        }

        return dA_prev;
    }

    private void applyAdam(double[][] dW, double[] db, double learningRate, int t) {
        int inputSize = W.length;
        if (mW == null) {
            mW = new double[inputSize][units];
            vW = new double[inputSize][units];
            mb = new double[units];
            vb = new double[units];
        }

        for (int k = 0; k < inputSize; k++) {
            for (int j = 0; j < units; j++) {
                mW[k][j] = BETA1 * mW[k][j] + (1 - BETA1) * dW[k][j];
                vW[k][j] = BETA2 * vW[k][j] + (1 - BETA2) * (dW[k][j] * dW[k][j]);

                double mW_hat = mW[k][j] / (1 - Math.pow(BETA1, t));
                double vW_hat = vW[k][j] / (1 - Math.pow(BETA2, t));

                W[k][j] -= learningRate * mW_hat / (Math.sqrt(vW_hat) + EPSILON);
            }
        }

        for (int j = 0; j < units; j++) {
            mb[j] = BETA1 * mb[j] + (1 - BETA1) * db[j];
            vb[j] = BETA2 * vb[j] + (1 - BETA2) * (db[j] * db[j]);

            double mb_hat = mb[j] / (1 - Math.pow(BETA1, t));
            double vb_hat = vb[j] / (1 - Math.pow(BETA2, t));

            b[j] -= learningRate * mb_hat / (Math.sqrt(vb_hat) + EPSILON);
        }
    }
}
