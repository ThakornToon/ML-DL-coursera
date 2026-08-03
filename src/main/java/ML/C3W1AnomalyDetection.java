package ML;

import ML.tools.AnomalyDetection;
import ML.tools.NumpyIO;

import java.io.IOException;

/**
 * C3W1AnomalyDetection implements the anomaly detection algorithms for
 * the Course 3 Week 1 assignment. It covers finding anomalies in both
 * 2D datasets and high-dimensional datasets using multivariate Gaussian
 * distributions.
 */
public final class C3W1AnomalyDetection {

    private C3W1AnomalyDetection() {}

    /**
     * Executes the anomaly detection processes.
     * It loads the datasets, estimates Gaussian parameters, calculates probabilities,
     * selects the best threshold (epsilon) using cross-validation, and identifies outliers.
     */
    public static void start() {
        System.out.println("Running Anomaly Detection...");
        String basePath = "src/main/resources/ML/C3_W1_Anomaly_Detection_Data/";
        
        try {
            // --- Part 1 ---
            System.out.println("\n--- Part 1: 2D Dataset ---");
            double[][] X_part1 = NumpyIO.loadDoubleMatrix(basePath + "X_part1.npy", 307, 2, true);
            double[][] X_val_part1 = NumpyIO.loadDoubleMatrix(basePath + "X_val_part1.npy", 307, 2, true);
            int[] y_val_part1 = NumpyIO.loadUint8Array(basePath + "y_val_part1.npy", 307);

            AnomalyDetection.GaussianParams params1 = AnomalyDetection.estimateGaussian(X_part1);
            double[] p_part1 = AnomalyDetection.multivariateGaussian(X_part1, params1.mu, params1.var);
            double[] p_val_part1 = AnomalyDetection.multivariateGaussian(X_val_part1, params1.mu, params1.var);

            AnomalyDetection.ThresholdResult res1 = AnomalyDetection.selectThreshold(y_val_part1, p_val_part1);
            System.out.printf("Best epsilon found using cross-validation: %e%n", res1.bestEpsilon);
            System.out.printf("Best F1 on Cross Validation Set: %f%n", res1.bestF1);

            int outliers1 = 0;
            for (double p : p_part1) {
                if (p < res1.bestEpsilon) outliers1++;
            }
            System.out.println("Outliers found: " + outliers1);
            System.out.println("(Expected: epsilon ~ 8.99e-05, F1 ~ 0.875, outliers = 6)");

            // --- Part 2 ---
            System.out.println("\n--- Part 2: High Dimensional Dataset ---");
            double[][] X_part2 = NumpyIO.loadDoubleMatrix(basePath + "X_part2.npy", 1000, 11, true);
            double[][] X_val_part2 = NumpyIO.loadDoubleMatrix(basePath + "X_val_part2.npy", 100, 11, true);
            int[] y_val_part2 = NumpyIO.loadUint8Array(basePath + "y_val_part2.npy", 100);

            AnomalyDetection.GaussianParams params2 = AnomalyDetection.estimateGaussian(X_part2);
            double[] p_part2 = AnomalyDetection.multivariateGaussian(X_part2, params2.mu, params2.var);
            double[] p_val_part2 = AnomalyDetection.multivariateGaussian(X_val_part2, params2.mu, params2.var);

            AnomalyDetection.ThresholdResult res2 = AnomalyDetection.selectThreshold(y_val_part2, p_val_part2);
            System.out.printf("Best epsilon found using cross-validation: %e%n", res2.bestEpsilon);
            System.out.printf("Best F1 on Cross Validation Set: %f%n", res2.bestF1);

            int outliers2 = 0;
            for (double p : p_part2) {
                if (p < res2.bestEpsilon) outliers2++;
            }
            System.out.println("Outliers found: " + outliers2);
            System.out.println("(Expected: epsilon ~ 1.38e-18, F1 ~ 0.615, outliers = 117)");

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("=================================================================\n");
    }
}
