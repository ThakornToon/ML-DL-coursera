package ML.tools;

/**
 * Tools for Anomaly Detection using Multivariate Gaussian Distribution.
 * This class provides methods to estimate parameters for a Gaussian distribution,
 * calculate probabilities, and select the optimal threshold for anomaly detection.
 */
public final class AnomalyDetection {

    // Prevent instantiation of this utility class
    private AnomalyDetection() {}

    /**
     * Container class for the parameters of a Gaussian distribution.
     */
    public static class GaussianParams {
        /** The mean of each feature. */
        public double[] mu;
        /** The variance of each feature. */
        public double[] var;

        /**
         * Constructs a new GaussianParams object.
         *
         * @param mu  Array representing the mean for each feature.
         * @param var Array representing the variance for each feature.
         */
        public GaussianParams(double[] mu, double[] var) {
            this.mu = mu;
            this.var = var;
        }
    }

    /**
     * Container class for the results of the threshold selection.
     */
    public static class ThresholdResult {
        /** The best epsilon threshold found. */
        public double bestEpsilon;
        /** The F1 score achieved with the best epsilon. */
        public double bestF1;

        /**
         * Constructs a new ThresholdResult object.
         *
         * @param bestEpsilon The optimal threshold for flagging anomalies.
         * @param bestF1      The F1 score corresponding to the optimal threshold.
         */
        public ThresholdResult(double bestEpsilon, double bestF1) {
            this.bestEpsilon = bestEpsilon;
            this.bestF1 = bestF1;
        }
    }

    /**
     * Estimates the parameters (mean and variance) of a Gaussian distribution 
     * based on the given dataset.
     *
     * @param X A 2D array of size [m][n] representing the dataset, where 'm' is 
     *          the number of examples and 'n' is the number of features.
     * @return A {@link GaussianParams} object containing the computed mean and variance 
     *         arrays of size [n].
     */
    public static GaussianParams estimateGaussian(double[][] X) {
        int m = X.length;
        int n = X[0].length;
        double[] mu = new double[n];
        double[] var = new double[n];

        // Calculate the mean (mu) for each feature
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                mu[j] += X[i][j];
            }
        }
        for (int j = 0; j < n; j++) {
            mu[j] /= m;
        }

        // Calculate the variance (var) for each feature
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double diff = X[i][j] - mu[j];
                var[j] += diff * diff;
            }
        }
        for (int j = 0; j < n; j++) {
            var[j] /= m;
        }

        return new GaussianParams(mu, var);
    }

    /**
     * Computes the probability density function of the multivariate Gaussian 
     * distribution for the given dataset. This implementation assumes that the 
     * features are independent (using a diagonal covariance matrix).
     *
     * @param X   A 2D array of size [m][n] representing the dataset.
     * @param mu  An array of size [n] containing the mean of each feature.
     * @param var An array of size [n] containing the variance of each feature.
     * @return An array of size [m] containing the computed probability for each example.
     */
    public static double[] multivariateGaussian(double[][] X, double[] mu, double[] var) {
        int m = X.length;
        int n = X[0].length;
        double[] p = new double[m];

        for (int i = 0; i < m; i++) {
            double prob = 1.0;
            for (int j = 0; j < n; j++) {
                double expPart = Math.exp(-Math.pow(X[i][j] - mu[j], 2) / (2 * var[j]));
                double factor = 1.0 / Math.sqrt(2 * Math.PI * var[j]);
                prob *= factor * expPart;
            }
            p[i] = prob;
        }
        return p;
    }

    /**
     * Finds the best threshold (epsilon) for anomaly detection by selecting the 
     * threshold that maximizes the F1 score on a cross-validation set.
     *
     * @param yVal An array of size [m] containing the ground truth labels 
     *             (1 for anomaly, 0 for normal).
     * @param pVal An array of size [m] containing the computed probabilities 
     *             for the cross-validation set.
     * @return A {@link ThresholdResult} object containing the best epsilon and 
     *         the corresponding best F1 score.
     */
    public static ThresholdResult selectThreshold(int[] yVal, double[] pVal) {
        double bestEpsilon = 0;
        double bestF1 = 0;

        // Find minimum and maximum probabilities to establish threshold range
        double minP = Double.MAX_VALUE;
        double maxP = -Double.MAX_VALUE;
        for (double p : pVal) {
            if (p < minP) minP = p;
            if (p > maxP) maxP = p;
        }

        // Search for the best epsilon over 1000 steps
        double stepSize = (maxP - minP) / 1000.0;

        for (double epsilon = minP; epsilon < maxP; epsilon += stepSize) {
            int tp = 0; // True positives
            int fp = 0; // False positives
            int fn = 0; // False negatives

            for (int i = 0; i < yVal.length; i++) {
                boolean pred = pVal[i] < epsilon; // Flag as anomaly if probability is below threshold
                boolean actual = yVal[i] == 1;

                if (pred && actual) {
                    tp++;
                } else if (pred && !actual) {
                    fp++;
                } else if (!pred && actual) {
                    fn++;
                }
            }

            double prec = (tp + fp > 0) ? (double) tp / (tp + fp) : 0;
            double rec = (tp + fn > 0) ? (double) tp / (tp + fn) : 0;
            
            double F1 = 0;
            if (prec + rec > 0) {
                F1 = 2 * prec * rec / (prec + rec);
            }

            if (F1 > bestF1) {
                bestF1 = F1;
                bestEpsilon = epsilon;
            }
        }

        return new ThresholdResult(bestEpsilon, bestF1);
    }
}
