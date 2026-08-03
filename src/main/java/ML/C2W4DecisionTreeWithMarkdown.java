package ML;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Note: This class implements a Decision Tree from scratch specifically for the
 * C2 W4 Markdown Lab. We do not use the existing ML.tools.trees.DecisionTree here
 * because this lab is designed for educational purposes to demonstrate step-by-step calculations.
 * 
 * Key differences from ML.tools.trees.DecisionTree:
 * 1. Uses Entropy (Information Gain) instead of Gini Impurity.
 * 2. Uses hardcoded binary categorical splits (X == 1) instead of continuous thresholds (X <= threshold).
 * 3. Designed to print intermediate calculation steps (like entropy at each node and information gain)
 *    to precisely match the expected output of the original Jupyter Notebook.
 */
public final class C2W4DecisionTreeWithMarkdown {

    private C2W4DecisionTreeWithMarkdown() {
    }

    /**
     * Computes the entropy (information gain measure) of a node given the labels.
     * 
     * @param y Array of binary labels (0 or 1).
     * @return The calculated entropy value.
     */
    public static double computeEntropy(int[] y) {
        if (y.length == 0) return 0.0;
        int count1 = 0;
        for (int val : y) {
            if (val == 1) count1++;
        }
        double p1 = (double) count1 / y.length;
        if (p1 == 0.0 || p1 == 1.0) return 0.0;
        return -p1 * (Math.log(p1) / Math.log(2)) - (1 - p1) * (Math.log(1 - p1) / Math.log(2));
    }

    /**
     * Represents the result of a dataset split, containing indices for the left and right child nodes.
     */
    public static class Split {
        public List<Integer> leftIndices = new ArrayList<>();
        public List<Integer> rightIndices = new ArrayList<>();
    }

    /**
     * Splits the dataset based on a specific feature.
     * 
     * @param X The feature matrix.
     * @param nodeIndices Indices of the data points at the current node.
     * @param feature The index of the feature to split on.
     * @return A Split object containing left and right indices.
     */
    public static Split splitDataset(double[][] X, List<Integer> nodeIndices, int feature) {
        Split split = new Split();
        for (int i : nodeIndices) {
            if (X[i][feature] == 1.0) {
                split.leftIndices.add(i);
            } else {
                split.rightIndices.add(i);
            }
        }
        return split;
    }

    /**
     * Computes the information gain from splitting a node on a specific feature.
     * 
     * @param X The feature matrix.
     * @param y Array of binary labels.
     * @param nodeIndices Indices of the data points at the current node.
     * @param feature The index of the feature to evaluate.
     * @return The calculated information gain.
     */
    public static double computeInformationGain(double[][] X, int[] y, List<Integer> nodeIndices, int feature) {
        Split split = splitDataset(X, nodeIndices, feature);

        int[] yNode = new int[nodeIndices.size()];
        for (int i = 0; i < nodeIndices.size(); i++) yNode[i] = y[nodeIndices.get(i)];

        int[] yLeft = new int[split.leftIndices.size()];
        for (int i = 0; i < split.leftIndices.size(); i++) yLeft[i] = y[split.leftIndices.get(i)];

        int[] yRight = new int[split.rightIndices.size()];
        for (int i = 0; i < split.rightIndices.size(); i++) yRight[i] = y[split.rightIndices.get(i)];

        double nodeEntropy = computeEntropy(yNode);
        double leftEntropy = computeEntropy(yLeft);
        double rightEntropy = computeEntropy(yRight);

        double wLeft = (double) split.leftIndices.size() / nodeIndices.size();
        double wRight = (double) split.rightIndices.size() / nodeIndices.size();

        double weightedEntropy = wLeft * leftEntropy + wRight * rightEntropy;

        return nodeEntropy - weightedEntropy;
    }

    /**
     * Finds the best feature to split on by maximizing information gain.
     * 
     * @param X The feature matrix.
     * @param y Array of binary labels.
     * @param nodeIndices Indices of the data points at the current node.
     * @return The index of the best feature to split on, or -1 if no features exist.
     */
    public static int getBestSplit(double[][] X, int[] y, List<Integer> nodeIndices) {
        if (X.length == 0) return -1;
        int numFeatures = X[0].length;
        int bestFeature = -1;
        double maxInfoGain = 0;

        for (int feature = 0; feature < numFeatures; feature++) {
            double infoGain = computeInformationGain(X, y, nodeIndices, feature);
            if (infoGain > maxInfoGain) {
                maxInfoGain = infoGain;
                bestFeature = feature;
            }
        }
        return bestFeature;
    }

    /**
     * Recursively builds the decision tree and prints the steps and splits.
     * 
     * @param X The feature matrix.
     * @param y Array of binary labels.
     * @param nodeIndices Indices of the data points at the current node.
     * @param branchName The name of the branch ("Root", "Left", "Right").
     * @param maxDepth The maximum depth the tree is allowed to reach.
     * @param currentDepth The current depth in the recursive tree building process.
     */
    public static void buildTreeRecursive(double[][] X, int[] y, List<Integer> nodeIndices, String branchName, int maxDepth, int currentDepth) {
        if (currentDepth == maxDepth) {
            String formatting = " ".repeat(currentDepth) + "-".repeat(currentDepth);
            System.out.println(formatting + " " + branchName + " leaf node with indices " + nodeIndices);
            return;
        }

        int bestFeature = getBestSplit(X, y, nodeIndices);
        String formatting = "-".repeat(currentDepth);
        System.out.println(formatting + (currentDepth > 0 ? " " : "") + "Depth " + currentDepth + ", " + branchName + ": Split on feature: " + bestFeature);

        Split split = splitDataset(X, nodeIndices, bestFeature);
        buildTreeRecursive(X, y, split.leftIndices, "Left", maxDepth, currentDepth + 1);
        buildTreeRecursive(X, y, split.rightIndices, "Right", maxDepth, currentDepth + 1);
    }

    /**
     * Executes the decision tree building process using the dataset defined for the lab.
     */
    public static void start() {
        System.out.println("Starting C2W4 Decision Tree with Markdown Lab");

        double[][] X_train = {
                {1, 1, 1}, {1, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1},
                {0, 1, 1}, {0, 0, 0}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}
        };
        int[] y_train = {1, 1, 0, 0, 1, 0, 0, 1, 1, 0};

        System.out.println("Entropy at root node: " + computeEntropy(y_train));

        List<Integer> rootIndices = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        int feature = 0;

        Split split1 = splitDataset(X_train, rootIndices, feature);
        System.out.println("\nCASE 1:");
        System.out.println("Left indices: " + split1.leftIndices);
        System.out.println("Right indices: " + split1.rightIndices);

        List<Integer> rootIndicesSubset = Arrays.asList(0, 2, 4, 6, 8);
        Split split2 = splitDataset(X_train, rootIndicesSubset, feature);
        System.out.println("\nCASE 2:");
        System.out.println("Left indices: " + split2.leftIndices);
        System.out.println("Right indices: " + split2.rightIndices);
        System.out.println();

        System.out.println("Information Gain from splitting the root on brown cap: " + computeInformationGain(X_train, y_train, rootIndices, 0));
        System.out.println("Information Gain from splitting the root on tapering stalk shape: " + computeInformationGain(X_train, y_train, rootIndices, 1));
        System.out.println("Information Gain from splitting the root on solitary: " + computeInformationGain(X_train, y_train, rootIndices, 2));

        System.out.println("Best feature to split on: " + getBestSplit(X_train, y_train, rootIndices));
        System.out.println();

        buildTreeRecursive(X_train, y_train, rootIndices, "Root", 2, 0);

        System.out.println("=================================================================\n");
    }
}
