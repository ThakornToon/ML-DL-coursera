package ML;

import ML.tools.DataFrame;
import ML.tools.Metrics;
import ML.tools.trees.DecisionTree;
import ML.tools.trees.GradientBoostedTree;
import ML.tools.trees.RandomForest;

import java.util.Arrays;
import java.util.List;

public final class C2W4Lab2TreeEnsemble {

    private C2W4Lab2TreeEnsemble() {
    }

    public static void start() {
        System.out.println("Starting C2W4 Lab 2: Tree Ensemble");
        int RANDOM_STATE = 55;

        // 1. Load data
        String path = "src/main/resources/C2_W4_Lab_02_Tree_Ensemble_Data/heart.csv";
        DataFrame df = DataFrame.readCSV(path);

        // 2. One-hot encoding
        List<String> catVariables = Arrays.asList("Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope");
        df.getDummies(catVariables);

        // 3. Train-test split
        List<String> features = df.getFeatures("HeartDisease");
        System.out.println("Number of features after one-hot encoding: " + features.size());
        
        double[][] X = df.toNumericMatrix(features);
        double[] y = df.toNumericArray("HeartDisease");

        DataFrame.Split split = DataFrame.trainTestSplit(X, y, 0.8, RANDOM_STATE);
        
        System.out.println("Train samples: " + split.XTrain.length);
        System.out.println("Validation samples: " + split.XVal.length);
        
        double targetSum = 0;
        for (double val : split.yTrain) targetSum += val;
        System.out.printf("Target proportion: %.4f%n", targetSum / split.yTrain.length);

        // --- Decision Tree ---
        System.out.println("\n--- Decision Tree ---");
        int[] minSamplesSplitList = {2, 10, 30, 50, 100, 200, 300, 700};
        System.out.println("Varying min_samples_split:");
        for (int minSamplesSplit : minSamplesSplitList) {
            DecisionTree dt = new DecisionTree(-1, minSamplesSplit, -1, false, RANDOM_STATE);
            dt.fit(split.XTrain, split.yTrain);
            double accTrain = Metrics.accuracyScore(split.yTrain, dt.predict(split.XTrain));
            double accVal = Metrics.accuracyScore(split.yVal, dt.predict(split.XVal));
            System.out.printf("min_samples_split: %3d | Train Acc: %.4f | Val Acc: %.4f%n", minSamplesSplit, accTrain, accVal);
        }

        int[] maxDepthList = {1, 2, 3, 4, 8, 16, 32, 64, -1}; // -1 for None
        System.out.println("\nVarying max_depth:");
        for (int maxDepth : maxDepthList) {
            DecisionTree dt = new DecisionTree(maxDepth, 2, -1, false, RANDOM_STATE);
            dt.fit(split.XTrain, split.yTrain);
            double accTrain = Metrics.accuracyScore(split.yTrain, dt.predict(split.XTrain));
            double accVal = Metrics.accuracyScore(split.yVal, dt.predict(split.XVal));
            System.out.printf("max_depth: %2d | Train Acc: %.4f | Val Acc: %.4f%n", maxDepth, accTrain, accVal);
        }

        DecisionTree decisionTreeModel = new DecisionTree(4, 50, -1, false, RANDOM_STATE);
        decisionTreeModel.fit(split.XTrain, split.yTrain);
        System.out.println("\nFinal Decision Tree Metrics:");
        System.out.printf("\tTrain Accuracy: %.4f%n", Metrics.accuracyScore(split.yTrain, decisionTreeModel.predict(split.XTrain)));
        System.out.printf("\tVal Accuracy: %.4f%n", Metrics.accuracyScore(split.yVal, decisionTreeModel.predict(split.XVal)));

        // --- Random Forest ---
        System.out.println("\n--- Random Forest ---");
        System.out.println("Varying min_samples_split:");
        for (int minSamplesSplit : minSamplesSplitList) {
            RandomForest rf = new RandomForest(100, -1, minSamplesSplit, RANDOM_STATE);
            rf.fit(split.XTrain, split.yTrain);
            double accTrain = Metrics.accuracyScore(split.yTrain, rf.predict(split.XTrain));
            double accVal = Metrics.accuracyScore(split.yVal, rf.predict(split.XVal));
            System.out.printf("min_samples_split: %3d | Train Acc: %.4f | Val Acc: %.4f%n", minSamplesSplit, accTrain, accVal);
        }
        
        int[] rfMaxDepthList = {2, 4, 8, 16, 32, 64, -1};
        System.out.println("\nVarying max_depth:");
        for (int maxDepth : rfMaxDepthList) {
            RandomForest rf = new RandomForest(100, maxDepth, 2, RANDOM_STATE);
            rf.fit(split.XTrain, split.yTrain);
            double accTrain = Metrics.accuracyScore(split.yTrain, rf.predict(split.XTrain));
            double accVal = Metrics.accuracyScore(split.yVal, rf.predict(split.XVal));
            System.out.printf("max_depth: %2d | Train Acc: %.4f | Val Acc: %.4f%n", maxDepth, accTrain, accVal);
        }

        int[] rfNEstimatorsList = {10, 50, 100, 500};
        System.out.println("\nVarying n_estimators:");
        for (int nEstimators : rfNEstimatorsList) {
            RandomForest rf = new RandomForest(nEstimators, -1, 2, RANDOM_STATE);
            rf.fit(split.XTrain, split.yTrain);
            double accTrain = Metrics.accuracyScore(split.yTrain, rf.predict(split.XTrain));
            double accVal = Metrics.accuracyScore(split.yVal, rf.predict(split.XVal));
            System.out.printf("n_estimators: %3d | Train Acc: %.4f | Val Acc: %.4f%n", nEstimators, accTrain, accVal);
        }

        RandomForest rfModel = new RandomForest(100, 16, 10, RANDOM_STATE);
        rfModel.fit(split.XTrain, split.yTrain);
        System.out.println("\nFinal Random Forest Metrics:");
        System.out.printf("\tTrain Accuracy: %.4f%n", Metrics.accuracyScore(split.yTrain, rfModel.predict(split.XTrain)));
        System.out.printf("\tVal Accuracy: %.4f%n", Metrics.accuracyScore(split.yVal, rfModel.predict(split.XVal)));

        // --- XGBoost (Gradient Boosted Tree) ---
        System.out.println("\n--- XGBoost (Gradient Boosted Tree) ---");
        int n = (int) (split.XTrain.length * 0.8);
        double[][] XTrainFit = new double[n][];
        double[] yTrainFit = new double[n];
        double[][] XTrainEval = new double[split.XTrain.length - n][];
        double[] yTrainEval = new double[split.yTrain.length - n];

        System.arraycopy(split.XTrain, 0, XTrainFit, 0, n);
        System.arraycopy(split.yTrain, 0, yTrainFit, 0, n);
        System.arraycopy(split.XTrain, n, XTrainEval, 0, split.XTrain.length - n);
        System.arraycopy(split.yTrain, n, yTrainEval, 0, split.yTrain.length - n);

        GradientBoostedTree xgb = new GradientBoostedTree(500, 0.1, 3, false, RANDOM_STATE);
        // Note: early stopping using XTrainEval is not implemented for simplicity,
        // but we train on XTrainFit just to match the split logic in the notebook.
        xgb.fit(XTrainFit, yTrainFit);
        System.out.println("Final XGBoost Metrics:");
        System.out.printf("\tTrain Accuracy: %.4f%n", Metrics.accuracyScore(split.yTrain, xgb.predict(split.XTrain)));
        System.out.printf("\tVal Accuracy: %.4f%n", Metrics.accuracyScore(split.yVal, xgb.predict(split.XVal)));

        System.out.println("=================================================================\n");
    }
}
