package ML;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public final class C1W3LogisticRegression {

    // Prevent instantiation
    private C1W3LogisticRegression() {
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double computeCost(double[][] x, double[] y, double[] w, double b) {
        int m = x.length;
        double cost = 0.0;

        for (int i = 0; i < m; i++) {
            double z = 0.0;
            for (int j = 0; j < w.length; j++) {
                z += w[j] * x[i][j];
            }
            z += b;
            double fWb = sigmoid(z);
            
            // To prevent log(0) which results in NaN or -Infinity
            double epsilon = 1e-15;
            fWb = Math.max(epsilon, Math.min(1 - epsilon, fWb));
            
            cost += -y[i] * Math.log(fWb) - (1 - y[i]) * Math.log(1 - fWb);
        }
        
        return cost / m;
    }

    public static double computeCostReg(double[][] x, double[] y, double[] w, double b, double lambda) {
        int m = x.length;
        int n = w.length;
        
        double costWithoutReg = computeCost(x, y, w, b);
        
        double regCost = 0.0;
        for (int j = 0; j < n; j++) {
            regCost += w[j] * w[j];
        }
        regCost = (lambda / (2.0 * m)) * regCost;
        
        return costWithoutReg + regCost;
    }

    public static double[][] computeGradient(double[][] x, double[] y, double[] w, double b) {
        int m = x.length;
        int n = w.length;
        double[] djDw = new double[n];
        double djDb = 0.0;

        for (int i = 0; i < m; i++) {
            double z = 0.0;
            for (int j = 0; j < n; j++) {
                z += w[j] * x[i][j];
            }
            z += b;
            double fWb = sigmoid(z);
            double err = fWb - y[i];

            for (int j = 0; j < n; j++) {
                djDw[j] += err * x[i][j];
            }
            djDb += err;
        }

        for (int j = 0; j < n; j++) {
            djDw[j] /= m;
        }
        djDb /= m;

        return new double[][]{djDw, new double[]{djDb}};
    }

    public static double[][] computeGradientReg(double[][] x, double[] y, double[] w, double b, double lambda) {
        int m = x.length;
        int n = w.length;
        
        double[][] grad = computeGradient(x, y, w, b);
        double[] djDw = grad[0];
        double djDb = grad[1][0];
        
        for (int j = 0; j < n; j++) {
            djDw[j] += (lambda / m) * w[j];
        }
        
        return new double[][]{djDw, new double[]{djDb}};
    }

    public static Object[] gradientDescent(double[][] x, double[] y, double[] wIn, double bIn, double alpha, int numIterations, double lambda) {
        double[] w = wIn.clone();
        double b = bIn;
        int n = w.length;

        for (int i = 0; i < numIterations; i++) {
            double[][] gradient = computeGradientReg(x, y, w, b, lambda);
            double[] djDw = gradient[0];
            double djDb = gradient[1][0];

            for (int j = 0; j < n; j++) {
                w[j] = w[j] - alpha * djDw[j];
            }
            b = b - alpha * djDb;

            if (i % (Math.ceil(numIterations / 10.0)) == 0) {
                double cost = computeCostReg(x, y, w, b, lambda);
                System.out.printf("Iteration %4d: Cost %8.4f\n", i, cost);
            }
        }
        return new Object[]{w, b};
    }

    public static int[] predict(double[][] x, double[] w, double b) {
        int m = x.length;
        int[] p = new int[m];
        
        for (int i = 0; i < m; i++) {
            double z = 0.0;
            for (int j = 0; j < w.length; j++) {
                z += w[j] * x[i][j];
            }
            z += b;
            double fWb = sigmoid(z);
            p[i] = fWb >= 0.5 ? 1 : 0;
        }
        
        return p;
    }

    // Function to load data from resources
    public static Object[] loadData(String filePath) {
        List<double[]> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();

        try (InputStream is = C1W3LogisticRegression.class.getResourceAsStream(filePath)) {
            if (is == null) {
                System.err.println("Could not find data file: " + filePath);
                return new Object[]{new double[0][0], new double[0]};
            }
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().isEmpty()) continue;
                    String[] parts = line.split(",");
                    double[] features = new double[parts.length - 1];
                    for (int i = 0; i < parts.length - 1; i++) {
                        features[i] = Double.parseDouble(parts[i]);
                    }
                    xList.add(features);
                    yList.add(Double.parseDouble(parts[parts.length - 1]));
                }
            }
        } catch (Exception e) {
            System.err.println("Error reading data: " + e.getMessage());
            return new Object[]{new double[0][0], new double[0]};
        }

        double[][] x = xList.toArray(new double[0][0]);
        double[] y = yList.stream().mapToDouble(Double::doubleValue).toArray();

        return new Object[]{x, y};
    }
    
    // Normalization utility (Logistic Regression often requires normalized features for gradient descent to converge well)
    public static double[][] normalizeFeatures(double[][] x) {
        int m = x.length;
        int n = x[0].length;
        double[][] xNorm = new double[m][n];
        
        double[] mean = new double[n];
        double[] std = new double[n];
        
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += x[i][j];
            }
            mean[j] = sum / m;
            
            double sumSq = 0.0;
            for (int i = 0; i < m; i++) {
                sumSq += Math.pow(x[i][j] - mean[j], 2);
            }
            std[j] = Math.sqrt(sumSq / m);
            
            // Normalize
            for (int i = 0; i < m; i++) {
                if (std[j] != 0) {
                    xNorm[i][j] = (x[i][j] - mean[j]) / std[j];
                } else {
                    xNorm[i][j] = x[i][j] - mean[j];
                }
            }
        }
        
        return xNorm;
    }

    public static void start() {
        // Load data
        Object[] data = loadData("/C1_W3_Logistic_Regression_Data/ex2data1.txt");
        double[][] xTrain = (double[][]) data[0];
        double[] yTrain = (double[]) data[1];

        if (xTrain.length == 0) {
            System.err.println("No data loaded. Exiting...");
            return;
        }

        System.out.println("First five elements of xTrain are:");
        for (int i = 0; i < Math.min(5, xTrain.length); i++) {
            System.out.printf("[%.2f, %.2f]\n", xTrain[i][0], xTrain[i][1]);
        }
        System.out.println("First five elements of yTrain are:");
        for (int i = 0; i < Math.min(5, yTrain.length); i++) {
            System.out.print(yTrain[i] + " ");
        }
        System.out.println("\n");
        
        int n = xTrain[0].length;
        
        // Compute cost with initial w=[0,0], b=0
        double[] initialW = new double[n];
        double initialB = 0.0;
        double cost = computeCost(xTrain, yTrain, initialW, initialB);
        System.out.printf("Cost at initial w=[0,0], b=0: %.3f\n\n", cost);

        // Normalize features for faster convergence during gradient descent
        double[][] xTrainNorm = normalizeFeatures(xTrain);

        // Run gradient descent
        initialW = new double[n];
        initialB = 0.0;
        int iterations = 10000;
        double alpha = 0.1;
        double lambda = 0.0; // Unregularized for ex2data1

        System.out.println("Running gradient descent...");
        Object[] wb = gradientDescent(xTrainNorm, yTrain, initialW, initialB, alpha, iterations, lambda);
        double[] w = (double[]) wb[0];
        double b = (double) wb[1];
        System.out.printf("w found by gradient descent: [%.3f, %.3f]\n", w[0], w[1]);
        System.out.printf("b found by gradient descent: %.3f\n\n", b);

        // Save the trained weights
        ModelWeightsIO.saveWeights("c1w3_logistic_weights.txt", w, b);


        // Predict
        int[] predictions = predict(xTrainNorm, w, b);
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == (int) yTrain[i]) {
                correct++;
            }
        }
        double accuracy = (double) correct / predictions.length * 100.0;
        System.out.printf("Train Accuracy: %.2f%%\n", accuracy);
    }
}
