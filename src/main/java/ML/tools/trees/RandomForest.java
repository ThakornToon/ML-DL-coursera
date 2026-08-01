package ML.tools.trees;

import java.util.*;
import java.io.Serializable;

public class RandomForest implements Serializable {
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private List<DecisionTree> trees;
    private Random random;

    public RandomForest(int nEstimators, int maxDepth, int minSamplesSplit, int randomState) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.trees = new ArrayList<>();
        this.random = new Random(randomState);
    }

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
