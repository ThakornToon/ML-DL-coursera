package ML.tools;

/**
 * Normalization layer to normalize features by their mean and standard deviation (Z-score normalization).
 */
public class Normalization {
    private double[] mean;
    private double[] variance;
    private boolean adapted = false;

    /**
     * Adapts the normalization layer to the data by calculating mean and variance.
     * @param X training data to learn parameters from
     */
    public void adapt(double[][] X) {
        if (X == null || X.length == 0) return;
        
        int m = X.length;
        int n = X[0].length;
        
        mean = new double[n];
        variance = new double[n];

        // Calculate mean for each feature
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += X[i][j];
            }
            mean[j] = sum / m;
        }

        // Calculate variance for each feature
        for (int j = 0; j < n; j++) {
            double sumSq = 0.0;
            for (int i = 0; i < m; i++) {
                sumSq += Math.pow(X[i][j] - mean[j], 2);
            }
            variance[j] = sumSq / m;
        }
        
        adapted = true;
    }
    
    public double[] getMean() {
        return mean;
    }
    
    public double[] getVariance() {
        return variance;
    }
    
    /**
     * Manually set the mean and variance for the normalization layer.
     * @param mean the mean array
     * @param variance the variance array
     */
    public void setMeanAndVariance(double[] mean, double[] variance) {
        this.mean = mean;
        this.variance = variance;
        this.adapted = true;
    }

    /**
     * Normalize data using learned mean and variance.
     * @param X data to normalize
     * @return normalized data
     */
    public double[][] normalize(double[][] X) {
        if (!adapted) {
            throw new IllegalStateException("Normalization layer must be adapted to data before calling normalize().");
        }
        
        int m = X.length;
        int n = X[0].length;
        double[][] Xn = new double[m][n];
        
        for (int j = 0; j < n; j++) {
            // standard deviation with a small epsilon to avoid division by zero
            double std = Math.sqrt(variance[j] + 1e-7); 
            for (int i = 0; i < m; i++) {
                Xn[i][j] = (X[i][j] - mean[j]) / std;
            }
        }
        
        return Xn;
    }
}
