package ML.tools;

import java.util.Arrays;
import java.util.Random;

/**
 * KMeans clustering algorithm implementation.
 */
public class KMeans {
    
    private int nClusters;
    private int maxIters;
    private double[][] centroids;

    /**
     * Constructs a new KMeans instance.
     *
     * @param nClusters the number of clusters (K)
     * @param maxIters the maximum number of iterations
     */
    public KMeans(int nClusters, int maxIters) {
        this.nClusters = nClusters;
        this.maxIters = maxIters;
    }

    /**
     * Fits the KMeans model to the given training data.
     *
     * @param X the training data points of shape (m, n) where m is the number of examples and n is the number of features
     */
    public void fit(double[][] X) {
        this.centroids = kMeansInitCentroids(X, this.nClusters);
        int[] idx = new int[X.length];

        for (int i = 0; i < this.maxIters; i++) {
            idx = findClosestCentroids(X, this.centroids);
            this.centroids = computeCentroids(X, idx, this.nClusters);
        }
    }

    /**
     * Predicts the closest cluster for each data point in X.
     *
     * @param X the data points to predict of shape (m, n)
     * @return an array of cluster indices (0 to nClusters - 1) for each data point
     */
    public int[] predict(double[][] X) {
        return findClosestCentroids(X, this.centroids);
    }

    /**
     * Gets the computed centroids after fitting the model.
     *
     * @return a 2D array of centroids of shape (nClusters, n)
     */
    public double[][] getCentroids() {
        return this.centroids;
    }

    /**
     * Computes the centroid memberships for every example.
     *
     * @param X the input data points of shape (m, n)
     * @param centroids the current centroids of shape (K, n)
     * @return an array containing the index of the closest centroid for each example
     */
    public static int[] findClosestCentroids(double[][] X, double[][] centroids) {
        int m = X.length;
        int K = centroids.length;
        int[] idx = new int[m];

        for (int i = 0; i < m; i++) {
            double minDistance = Double.MAX_VALUE;
            int bestK = 0;
            for (int j = 0; j < K; j++) {
                double dist = 0;
                for (int d = 0; d < X[i].length; d++) {
                    dist += Math.pow(X[i][d] - centroids[j][d], 2);
                }
                if (dist < minDistance) {
                    minDistance = dist;
                    bestK = j;
                }
            }
            idx[i] = bestK;
        }

        return idx;
    }

    /**
     * Returns the new centroids by computing the means of the data points assigned to each centroid.
     *
     * @param X the input data points of shape (m, n)
     * @param idx an array containing the index of the closest centroid for each example
     * @param K the number of clusters/centroids
     * @return a 2D array of the newly computed centroids of shape (K, n)
     */
    public static double[][] computeCentroids(double[][] X, int[] idx, int K) {
        int m = X.length;
        int n = X[0].length;
        double[][] newCentroids = new double[K][n];
        int[] counts = new int[K];

        // For Each data point, add it to the cluster that it belongs to
        for (int i = 0; i < m; i++) {
            int k = idx[i];
            counts[k]++;
            // For Each feature, add the data point to the cluster
            for (int j = 0; j < n; j++) {
                newCentroids[k][j] += X[i][j];
            }
        }

        // For Each cluster, divide by the count of data points
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                // For Each feature, divide by the count of data points
                for (int j = 0; j < n; j++) {
                    newCentroids[k][j] /= counts[k];
                }
            }
        }

        return newCentroids;
    }

    /**
     * Initializes K centroids that are to be used in K-Means on the dataset X.
     *
     * @param X the input data points of shape (m, n)
     * @param K the number of clusters/centroids
     * @return a 2D array of initialized centroids randomly selected from X
     */
    public static double[][] kMeansInitCentroids(double[][] X, int K) {
        int m = X.length;
        int n = X[0].length;
        double[][] initialCentroids = new double[K][n];
        
        Random rand = new Random();
        int[] randidx = new int[m];
        for (int i = 0; i < m; i++) {
            randidx[i] = i;
        }
        for (int i = 0; i < m; i++) {
            int swapIdx = i + rand.nextInt(m - i);
            int temp = randidx[i];
            randidx[i] = randidx[swapIdx];
            randidx[swapIdx] = temp;
        }

        for (int i = 0; i < K; i++) {
            initialCentroids[i] = Arrays.copyOf(X[randidx[i]], n);
        }

        return initialCentroids;
    }
}
