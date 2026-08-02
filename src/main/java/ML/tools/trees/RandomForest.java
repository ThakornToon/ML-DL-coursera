package ML.tools.trees;

import java.util.*;
import java.io.Serializable;

/**
 * An implementation of the Random Forest ensemble learning method for classification.
 * It builds a "forest" of decision trees trained on bootstrap samples of the data
 * and outputs the mode of the classes predicted by individual trees.
 */
public class RandomForest implements Serializable {
    /** The number of trees in the forest. */
    private int nEstimators;
    
    /** The maximum depth of each decision tree. */
    private int maxDepth;
    
    /** The minimum number of samples required to split an internal node. */
    private int minSamplesSplit;
    
    /** The collection of trained decision trees in this forest. */
    private List<DecisionTree> trees;
    
    /** Random number generator for reproducibility. */
    private Random random;

    /**
     * Initializes a Random Forest model with specified hyperparameters.
     *
     * @param nEstimators     The number of trees in the forest.
     * @param maxDepth        The maximum depth of the trees.
     * @param minSamplesSplit The minimum number of samples required to split a node.
     * @param randomState     The seed for the random number generator.
     */
    public RandomForest(int nEstimators, int maxDepth, int minSamplesSplit, int randomState) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.trees = new ArrayList<>();
        this.random = new Random(randomState);
    }

    /**
     * Trains the Random Forest model using the provided training data.
     *
     * @param X The 2D array of training features (samples x features).
     * @param y The 1D array of training labels.
     */
    public void fit(double[][] X, double[] y) {
        int nSamples = X.length;
        int maxFeatures = (int) Math.sqrt(X[0].length);

        for (int i = 0; i < nEstimators; i++) {
            DecisionTree tree = new DecisionTree(maxDepth, minSamplesSplit, maxFeatures, false, random.nextInt());
            
            // Bootstrap sampling
            double[][] XBootstrap = new double[nSamples][X[0].length];
            double[] yBootstrap = new double[nSamples];

            for (int j = 0; j < nSamples; j++) {
                int idx = random.nextInt(nSamples);
                XBootstrap[j] = X[idx];
                yBootstrap[j] = y[idx];
            }

            tree.fit(XBootstrap, yBootstrap);
            trees.add(tree);
        }
    }

    /**
     * Predicts the class labels for a given set of input features.
     *
     * @param X The 2D array of input features to evaluate.
     * @return A 1D array of predicted class labels (majority vote from all trees).
     */
    public double[] predict(double[][] X) {
        double[][] allPreds = new double[nEstimators][X.length];
        for (int i = 0; i < nEstimators; i++) {
            allPreds[i] = trees.get(i).predict(X);
        }

        double[] finalPreds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            Map<Double, Integer> counts = new HashMap<>();
            for (int j = 0; j < nEstimators; j++) {
                double val = allPreds[j][i];
                counts.put(val, counts.getOrDefault(val, 0) + 1);
            }
            double maxClass = -1;
            int maxCount = -1;
            for (Map.Entry<Double, Integer> entry : counts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    maxClass = entry.getKey();
                }
            }
            finalPreds[i] = maxClass;
        }
        return finalPreds;
    }
}
