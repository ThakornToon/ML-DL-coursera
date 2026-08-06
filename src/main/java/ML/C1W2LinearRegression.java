package ML;

import ML.tools.ModelWeightsIO;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of the C1W2 Linear Regression Assignment.
 * This class handles loading data, computing cost, and performing gradient descent
 * for linear regression tasks.
 */
public final class C1W2LinearRegression {

    // Prevent instantiation
    private C1W2LinearRegression() {
    }

    public static double computeCost(double[] x, double[] y, double w, double b) {
        int m = x.length;
        double costSum = 0;

        for (int i = 0; i < m; i++) {
            double fWb = w * x[i] + b;
            double cost = Math.pow(fWb - y[i], 2);
            costSum += cost;
        }

        return (1.0 / (2 * m)) * costSum;
    }

    public static double[] computeGradient(double[] x, double[] y, double w, double b) {
        int m = x.length;
        double djDw = 0;
        double djDb = 0;

        for (int i = 0; i < m; i++) {
            double fWb = w * x[i] + b;
            double djDwI = (fWb - y[i]) * x[i];
            double djDbI = (fWb - y[i]);
            djDb += djDbI;
            djDw += djDwI;
        }

        djDw = djDw / m;
        djDb = djDb / m;

        return new double[]{djDw, djDb};
    }

    public static double[] gradientDescent(double[] x, double[] y, double wIn, double bIn, double alpha, int numIterations) {
        double w = wIn;
        double b = bIn;

        for (int i = 0; i < numIterations; i++) {
            double[] gradient = computeGradient(x, y, w, b);
            double djDw = gradient[0];
            double djDb = gradient[1];

            w = w - alpha * djDw;
            b = b - alpha * djDb;

            if (i % (Math.ceil(numIterations / 10.0)) == 0) {
                double cost = computeCost(x, y, w, b);
                System.out.printf("Iteration %4d: Cost %8.2f\n", i, cost);
            }
        }
        return new double[]{w, b};
    }

    // Function to load data from resources
    public static double[][] loadData(String filePath) {
        List<Double> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();

        try (InputStream is = C1W2LinearRegression.class.getResourceAsStream(filePath)) {
            if (is == null) {
                System.err.println("Could not find data file: " + filePath);
                return new double[][]{new double[0], new double[0]};
            }
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().isEmpty()) continue;
                    String[] parts = line.split(",");
                    xList.add(Double.parseDouble(parts[0]));
                    yList.add(Double.parseDouble(parts[1]));
                }
            }
        } catch (Exception e) {
            System.err.println("Error reading data: " + e.getMessage());
            return new double[][]{new double[0], new double[0]};
        }

        double[] xTrain = xList.stream().mapToDouble(Double::doubleValue).toArray();
        double[] yTrain = yList.stream().mapToDouble(Double::doubleValue).toArray();

        return new double[][]{xTrain, yTrain};
    }

    /**
     * Executes the linear regression workflow.
     */
    public static void start() {
        System.out.println("Starting C1W2 LinearRegression");

        // Load data using the separate function
        double[][] data = loadData("/ML/C1_W2_Linear_Regression_Data/ex1data1.txt");
        double[] xTrain = data[0];
        double[] yTrain = data[1];

        if (xTrain.length == 0) {
            System.err.println("No data loaded. Exiting...");
            return;
        }

        System.out.println("Type of xTrain: double[]");
        System.out.println("First five elements of xTrain are:");
        for (int i = 0; i < Math.min(5, xTrain.length); i++) {
            System.out.print(xTrain[i] + " ");
        }
        System.out.println("\n");

        // Compute cost with initial w=2, b=1
        double initialW = 2.0;
        double initialB = 1.0;
        double cost = computeCost(xTrain, yTrain, initialW, initialB);
        System.out.printf("Cost at initial w=2, b=1: %.3f\n\n", cost);

        // Compute gradient with test w=0.2, b=0.2
        double testW = 0.2;
        double testB = 0.2;
        double[] grads = computeGradient(xTrain, yTrain, testW, testB);
        System.out.println("Gradient at test w, b: " + grads[0] + " " + grads[1] + "\n");

        // Run gradient descent
        // Run gradient descent or load weights
        boolean loadModel = false; // Set to true to load saved weights
        double w, b;
        
        if (loadModel) {
            System.out.println("Loading saved weights...");
            Object[] loaded = ModelWeightsIO.loadDenseWeights("ml_c1w2_linear_weights.txt");
            if (loaded != null) {
                w = ((double[][]) loaded[0])[0][0];
                b = ((double[]) loaded[1])[0];
            } else {
                w = initialW;
                b = initialB;
            }
        } else {
            initialW = 0.0;
            initialB = 0.0;
            int iterations = 1500;
            double alpha = 0.01;

            double[] wb = gradientDescent(xTrain, yTrain, initialW, initialB, alpha, iterations);
            w = wb[0];
            b = wb[1];
            System.out.println("w,b found by gradient descent: " + w + " " + b + "\n");

            // Save the trained weights
            ModelWeightsIO.saveWeights("ml_c1w2_linear_weights.txt", w, b);
        }


        // Predict
        double predict1 = 3.5 * w + b;
        System.out.printf("For population = 35,000, we predict a profit of $%.2f\n", (predict1 * 10000));

        double predict2 = 7.0 * w + b;
        System.out.printf("For population = 70,000, we predict a profit of $%.2f\n", (predict2 * 10000));

        System.out.println("=================================================================\n");
    }
}
