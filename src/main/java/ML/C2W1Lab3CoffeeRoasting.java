package ML;

import ML.tools.Dense;
import ML.tools.Normalization;
import ML.tools.Sequential;

import java.util.Random;

/**
 * Implementation of the C2W1 Lab 3 Coffee Roasting assignment.
 * Demonstrates basic neural network usage on a simple coffee roasting dataset.
 */
public final class C2W1Lab3CoffeeRoasting {

    // Prevent instantiation
    private C2W1Lab3CoffeeRoasting() {
    }

    public static class CoffeeData {
        public double[][] X;
        public double[][] Y;

        public CoffeeData(double[][] X, double[][] Y) {
            this.X = X;
            this.Y = Y;
        }
    }

    /**
     * Creates a coffee roasting data set.
     * Roasting duration: 12-15 minutes is best.
     * Temperature range: 175-260C is best.
     */
    public static CoffeeData loadCoffeeData() {
        Random rng = new Random(2);
        int numRows = 200;
        double[][] X = new double[numRows][2];
        double[][] Y = new double[numRows][1];

        for (int i = 0; i < numRows; i++) {
            double t = rng.nextDouble(); // first random feature
            double d = rng.nextDouble(); // second random feature

            d = d * 4 + 11.5;          // 12-15 min is best
            t = t * (285 - 150) + 150; // 175-260 C is best

            X[i][0] = t;
            X[i][1] = d;

            double yLine = -3.0 / (260 - 175) * t + 21;
            if (t > 175 && t < 260 && d > 12 && d < 15 && d <= yLine) {
                Y[i][0] = 1.0;
            } else {
                Y[i][0] = 0.0;
            }
        }
        return new CoffeeData(X, Y);
    }

    /**
     * Executes the coffee roasting neural network lab.
     */
    public static void start() {
        System.out.println("Starting Coffee Roasting Lab");

        // 1. Load Data
        CoffeeData data = loadCoffeeData();
        double[][] X = data.X;
        double[][] Y = data.Y;

        System.out.println("X shape: [" + X.length + ", " + X[0].length + "]");
        System.out.println("Y shape: [" + Y.length + ", " + Y[0].length + "]");

        // 2. Normalize Data
        Normalization normLayer = new Normalization();
        normLayer.adapt(X);
        
        System.out.println("Adapted Mean: [" + normLayer.getMean()[0] + ", " + normLayer.getMean()[1] + "]");
        System.out.println("Adapted Variance: [" + normLayer.getVariance()[0] + ", " + normLayer.getVariance()[1] + "]");
        double[][] Xn = normLayer.normalize(X);

        // 3. Build Sequential Model with dynamic Dense layers (Keras-style)
        Sequential model = new Sequential();
        model.add(new Dense(3, "sigmoid", "layer1"));
        model.add(new Dense(1, "sigmoid", "layer2"));

        // 4. Compile and Train the Model (Train from scratch)
        // We use Adam optimizer with learning rate 0.01.
        // Since we are doing Full-Batch Gradient Descent (instead of mini-batch like Keras), 
        // we use a larger number of epochs (e.g. 2000) on the original 200 examples to converge.
        model.compile("adam", 0.01);
        boolean loadModel = false; // Set to true to load existing model weights
        
        if (loadModel) {
            System.out.println("\nLoading saved weights...");
            for (ML.tools.Dense layer : model.getLayers()) {
                if (layer.getName() != null) {
                    String filename = "ml_c2w1_layer" + layer.getName().replace("layer", "") + "_weights.txt";
                    Object[] loaded = ML.tools.ModelWeightsIO.loadDenseWeights(filename);
                    if (loaded != null) layer.setWeights((double[][]) loaded[0], (double[]) loaded[1]);
                }
            }
        } else {
            System.out.println("\nTraining the model...");
            model.fit(Xn, Y, 4000);
        }
        
        // Print model summary
        System.out.println();
        model.summary();

        // Save weights for each layer
        Dense layer1 = model.getLayer("layer1");
        if (layer1 != null) {
            ML.tools.ModelWeightsIO.saveWeights("ml_c2w1_layer1_weights.txt", layer1.getWeightsW(), layer1.getWeightsB());
        }

        Dense layer2 = model.getLayer("layer2");
        if (layer2 != null) {
            ML.tools.ModelWeightsIO.saveWeights("ml_c2w1_layer2_weights.txt", layer2.getWeightsW(), layer2.getWeightsB());
        }

        // 5. Run prediction on test data
        double[][] X_tst = {
                {200, 13.9}, // positive example
                {200, 17}    // negative example
        };

        // Normalize test data before feeding to the network
        double[][] X_tstn = normLayer.normalize(X_tst);

        double[][] predictions = model.predict(X_tstn);

        System.out.println("Predictions:");
        for (int i = 0; i < predictions.length; i++) {
            System.out.printf("Example %d probability: %.4f%n", i + 1, predictions[i][0]);
        }

        // Apply threshold for final decision
        System.out.println("\nDecisions (Threshold >= 0.5):");
        for (int i = 0; i < predictions.length; i++) {
            int yhat = (predictions[i][0] >= 0.5) ? 1 : 0;
            System.out.println("Example " + (i + 1) + ": " + yhat);
        }

        System.out.println("=================================================================\n");
    }
}
