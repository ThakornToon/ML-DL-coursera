package ML.tools.trees;

import java.util.*;
import java.io.Serializable;

public class DecisionTree implements Serializable {
    private int maxDepth;
    private int minSamplesSplit;
    private int maxFeatures;
    private boolean isRegression;
    private TreeNode root;
    private Random random;

    public DecisionTree(int maxDepth, int minSamplesSplit, int maxFeatures, boolean isRegression, int randomState) {
        this.maxDepth = maxDepth <= 0 ? Integer.MAX_VALUE : maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.isRegression = isRegression;
        this.random = new Random(randomState);
    }

    public void fit(double[][] X, double[] y) {
        List<Integer> rowIndices = new ArrayList<>();
        for (int i = 0; i < X.length; i++) rowIndices.add(i);
        this.root = buildTree(X, y, rowIndices, 0);
    }

    private TreeNode buildTree(double[][] X, double[] y, List<Integer> rowIndices, int depth) {
        int nSamples = rowIndices.size();
        
        // Check stopping criteria
        if (depth >= maxDepth || nSamples < minSamplesSplit || isPure(y, rowIndices)) {
            return new TreeNode(leafValue(y, rowIndices));
        }

        // Find best split
        Split bestSplit = getBestSplit(X, y, rowIndices);
        
        if (bestSplit == null || bestSplit.gain <= 1e-9) {
            return new TreeNode(leafValue(y, rowIndices));
        }

        TreeNode leftChild = buildTree(X, y, bestSplit.leftIndices, depth + 1);
        TreeNode rightChild = buildTree(X, y, bestSplit.rightIndices, depth + 1);

        return new TreeNode(bestSplit.featureIndex, bestSplit.threshold, leftChild, rightChild);
    }

    private boolean isPure(double[] y, List<Integer> rowIndices) {
        double first = y[rowIndices.get(0)];
        for (int i = 1; i < rowIndices.size(); i++) {
            if (y[rowIndices.get(i)] != first) return false;
        }
        return true;
    }

    private double leafValue(double[] y, List<Integer> rowIndices) {
        if (isRegression) {
            double sum = 0;
            for (int idx : rowIndices) sum += y[idx];
            return sum / rowIndices.size();
        } else {
            // Majority class
            Map<Double, Integer> counts = new HashMap<>();
            for (int idx : rowIndices) {
                double val = y[idx];
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
            return maxClass;
        }
    }

    private class Split {
        int featureIndex;
        double threshold;
        List<Integer> leftIndices;
        List<Integer> rightIndices;
        double gain;
    }

    private Split getBestSplit(double[][] X, double[] y, List<Integer> rowIndices) {
        Split bestSplit = null;
        double bestGain = -Double.MAX_VALUE;
        
        int nFeatures = X[0].length;
        List<Integer> featureIndices = new ArrayList<>();
        for (int i = 0; i < nFeatures; i++) featureIndices.add(i);

        if (maxFeatures > 0 && maxFeatures < nFeatures) {
            Collections.shuffle(featureIndices, random);
            featureIndices = featureIndices.subList(0, maxFeatures);
        }

        double currentScore = calculateScore(y, rowIndices);

        for (int featIdx : featureIndices) {
            Set<Double> uniqueVals = new TreeSet<>();
            for (int idx : rowIndices) {
                uniqueVals.add(X[idx][featIdx]);
            }
            List<Double> sortedVals = new ArrayList<>(uniqueVals);

            for (int i = 0; i < sortedVals.size() - 1; i++) {
                double threshold = (sortedVals.get(i) + sortedVals.get(i+1)) / 2.0;
                
                List<Integer> leftIndices = new ArrayList<>();
                List<Integer> rightIndices = new ArrayList<>();
                
                for (int idx : rowIndices) {
                    if (X[idx][featIdx] <= threshold) leftIndices.add(idx);
                    else rightIndices.add(idx);
                }

                if (leftIndices.isEmpty() || rightIndices.isEmpty()) continue;

                double gain = currentScore - (leftIndices.size() * calculateScore(y, leftIndices) + rightIndices.size() * calculateScore(y, rightIndices)) / rowIndices.size();

                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = new Split();
                    bestSplit.featureIndex = featIdx;
                    bestSplit.threshold = threshold;
                    bestSplit.leftIndices = leftIndices;
                    bestSplit.rightIndices = rightIndices;
                    bestSplit.gain = gain;
                }
            }
        }
        return bestSplit;
    }

    private double calculateScore(double[] y, List<Integer> indices) {
        if (indices.isEmpty()) return 0;
        if (isRegression) {
            // Variance
            double mean = 0;
            for (int idx : indices) mean += y[idx];
            mean /= indices.size();
            double variance = 0;
            for (int idx : indices) variance += Math.pow(y[idx] - mean, 2);
            return variance / indices.size();
        } else {
            // Gini impurity
            Map<Double, Integer> counts = new HashMap<>();
            for (int idx : indices) {
                double val = y[idx];
                counts.put(val, counts.getOrDefault(val, 0) + 1);
            }
            double impurity = 1.0;
            for (int count : counts.values()) {
                double p = (double) count / indices.size();
                impurity -= p * p;
            }
            return impurity;
        }
    }

    public double[] predict(double[][] X) {
        double[] preds = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            preds[i] = predictRow(X[i], root);
        }
        return preds;
    }

    private double predictRow(double[] row, TreeNode node) {
        if (node.isLeaf) return node.value;
        if (row[node.featureIndex] <= node.threshold) {
            return predictRow(row, node.left);
        } else {
            return predictRow(row, node.right);
        }
    }
}
