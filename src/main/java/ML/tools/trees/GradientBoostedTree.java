package ML.tools.trees;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.io.Serializable;

/**
 * An implementation of Gradient Boosted Trees for both regression and binary classification.
 * This model builds a series of weak learners (decision trees), where each subsequent
 * tree aims to correct the pseudo-residuals of the combined previous trees.
 */
public class GradientBoostedTree implements Serializable {
    /** The number of boosting stages (trees) to perform. */
    private int nEstimators;
    
    /** The learning rate (shrinkage) that scales the contribution of each tree. */
    private double learningRate;
    
    /** The maximum depth of individual regression estimators. */
    private int maxDepth;
    
    /** True if this is a regression model, false for binary classification. */
    private boolean isRegression;
    
    /** The sequence of fitted decision trees. */
    private List<DecisionTree> trees;
    
    /** The initial constant prediction value (base score) before adding trees. */
    private double initialPrediction;
    
    /** Random number generator. */
    private Random random;

    /**
     * Initializes a Gradient Boosted Tree model.
     *
     * @param nEstimators  The number of boosting stages to perform.
     * @param learningRate The learning rate that shrinks the contribution of each tree.
     * @param maxDepth     The maximum depth of the individual trees.
     * @param isRegression True for regression tasks, false for classification.
     * @param randomState  The seed for the random number generator.
     */
    public GradientBoostedTree(int nEstimators, double learningRate, int maxDepth, boolean isRegression, int randomState) {
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.isRegression = isRegression;
        this.trees = new ArrayList<>();
        this.random = new Random(randomState);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Trains the Gradient Boosted Tree model.
     *
     * @param X The 2D array of training features.
     * @param y The 1D array of target values or class labels (0 or 1).
     */
    public void fit(double[][] X, double[] y) {
        int nSamples = X.length;
        
        if (isRegression) {
            double sum = 0;
            for (double v : y) sum += v;
            initialPrediction = sum / nSamples;
        } else {
            // Initial prediction (log odds) for classification
            int posCount = 0;
            for (double v : y) {
                if (v == 1.0) posCount++;
            }
            double p = (double) posCount / nSamples;
            p = Math.max(1e-5, Math.min(1 - 1e-5, p));
            initialPrediction = Math.log(p / (1.0 - p));
        }

        double[] F = new double[nSamples];
        for (int i = 0; i < nSamples; i++) {
            F[i] = initialPrediction;
        }

        for (int m = 0; m < nEstimators; m++) {
            double[] residuals = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                if (isRegression) {
                    residuals[i] = y[i] - F[i];
                } else {
                    double prob = sigmoid(F[i]);
                    residuals[i] = y[i] - prob;
                }
            }

            // Fit regression tree to residuals
            DecisionTree tree = new DecisionTree(maxDepth, 2, -1, true, random.nextInt());
            tree.fit(X, residuals);

            // Update F
            double[] treePreds = tree.predict(X);
            for (int i = 0; i < nSamples; i++) {
                F[i] += learningRate * treePreds[i];
            }
            trees.add(tree);
        }
    }

    private double[] computeF(double[][] X) {
        double[] F = new double[X.length];
        for (int i = 0; i < X.length; i++) F[i] = initialPrediction;

        for (DecisionTree tree : trees) {
            double[] treePreds = tree.predict(X);
            for (int i = 0; i < X.length; i++) {
                F[i] += learningRate * treePreds[i];
            }
        }
        return F;
    }

    /**
     * Predicts class probabilities for binary classification.
     * Only supported when isRegression is false.
     *
     * @param X The 2D array of input features to evaluate.
     * @return A 1D array of predicted probabilities for the positive class (1).
     * @throws UnsupportedOperationException if called on a regression model.
     */
    public double[] predictProbabilities(double[][] X) {
        if (isRegression) {
            throw new UnsupportedOperationException("predictProbabilities is not supported for regression models.");
        }
        double[] F = computeF(X);
        double[] probs = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            probs[i] = sigmoid(F[i]);
        }
        return probs;
    }

    /**
     * Predicts target values (regression) or class labels (classification).
     * For classification, it thresholds the probability at 0.5.
     *
     * @param X The 2D array of input features to evaluate.
     * @return A 1D array of predicted values or labels.
     */
    public double[] predict(double[][] X) {
        double[] F = computeF(X);
        
        if (isRegression) {
            return F;
        }

        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double prob = sigmoid(F[i]);
            preds[i] = prob >= 0.5 ? 1.0 : 0.0;
        }
        return preds;
    }
}
