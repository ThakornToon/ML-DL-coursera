package ML;

import ML.tools.Dense;
import ML.tools.Sequential;

public class C2W3Assignment {

    // Prevent instantiation
    private C2W3Assignment() {
    }

    public static class BlobsData {
        public double[][] XTrain, yTrain;
        public double[][] XCv, yCv;
        public double[][] XTest, yTest;

        public BlobsData(double[][] XTrain, double[][] yTrain, double[][] XCv, double[][] yCv, double[][] XTest, double[][] yTest) {
            this.XTrain = XTrain;
            this.yTrain = yTrain;
            this.XCv = XCv;
            this.yCv = yCv;
            this.XTest = XTest;
            this.yTest = yTest;
        }
    }

    public static BlobsData generateData() {
        int m = 800;
        int classes = 6;
        double[][] centers = {
            {-1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {-2.0, 1.0}, {-2.0, -1.0}
        };
        double std = 0.4;

        double[][] X = new double[m][2];
        double[][] y = new double[m][1];

        java.util.Random rand = new java.util.Random(2); // random_state=2

        for (int i = 0; i < m; i++) {
            int cluster = i % classes;
            X[i][0] = centers[cluster][0] + rand.nextGaussian() * std;
            X[i][1] = centers[cluster][1] + rand.nextGaussian() * std;
            y[i][0] = cluster;
        }

        // Split: 400 train, 200 CV, 200 test
        double[][] XTrain = new double[400][2], yTrain = new double[400][1];
        double[][] XCv = new double[200][2], yCv = new double[200][1];
        double[][] XTest = new double[200][2], yTest = new double[200][1];
        
        for (int i = 0; i < m; i++) {
            if (i < 400) {
                XTrain[i] = X[i]; yTrain[i] = y[i];
            } else if (i < 600) {
                XCv[i-400] = X[i]; yCv[i-400] = y[i];
            } else {
                XTest[i-600] = X[i]; yTest[i-600] = y[i];
            }
        }

        return new BlobsData(XTrain, yTrain, XCv, yCv, XTest, yTest);
    }

    public static class RegressionData {
        public double[][] XTrain;
        public double[][] yTrain;
        public double[][] XCv;
        public double[][] yCv;

        public RegressionData(double[][] XTrain, double[][] yTrain, double[][] XCv, double[][] yCv) {
            this.XTrain = XTrain;
            this.yTrain = yTrain;
            this.XCv = XCv;
            this.yCv = yCv;
        }
    }

    public static RegressionData generateRegressionData() {
        int mTrain = 20;
        int mCv = 10;
        double[][] XTrain = new double[mTrain][1];
        double[][] yTrain = new double[mTrain][1];
        double[][] XCv = new double[mCv][1];
        double[][] yCv = new double[mCv][1];

        java.util.Random rand = new java.util.Random(1);
        double scale = 0.7;

        for (int i = 0; i < mTrain; i++) {
            double x = (49.0 / (mTrain - 1)) * i;
            double yIdeal = x * x;
            double noise = scale * yIdeal * (rand.nextDouble() - 0.5);
            XTrain[i][0] = x;
            yTrain[i][0] = yIdeal + noise;
        }

        for (int i = 0; i < mCv; i++) {
            double x = (49.0 / (mCv - 1)) * i;
            double yIdeal = x * x;
            double noise = scale * yIdeal * (rand.nextDouble() - 0.5);
            XCv[i][0] = x;
            yCv[i][0] = yIdeal + noise;
        }
        return new RegressionData(XTrain, yTrain, XCv, yCv);
    }

    public static double[][] expandPolynomialFeatures(double[][] X, int degree) {
        int m = X.length;
        double[][] XPoly = new double[m][degree];
        for (int i = 0; i < m; i++) {
            double xVal = X[i][0];
            for (int d = 1; d <= degree; d++) {
                XPoly[i][d - 1] = Math.pow(xVal, d);
            }
        }
        return XPoly;
    }

    public static void scaleFeatures(double[][] XTrain, double[][] XCv) {
        int degree = XTrain[0].length;
        int mTrain = XTrain.length;
        int mCv = XCv.length;

        for (int j = 0; j < degree; j++) {
            double sum = 0;
            for (int i = 0; i < mTrain; i++) sum += XTrain[i][j];
            double mean = sum / mTrain;

            double sumSq = 0;
            for (int i = 0; i < mTrain; i++) sumSq += Math.pow(XTrain[i][j] - mean, 2);
            double std = Math.sqrt(sumSq / mTrain);

            if (std < 1e-8) std = 1e-8;

            for (int i = 0; i < mTrain; i++) XTrain[i][j] = (XTrain[i][j] - mean) / std;
            for (int i = 0; i < mCv; i++) XCv[i][j] = (XCv[i][j] - mean) / std;
        }
    }

    public static void findOptimalDegree() {
        System.out.println("\n--- Finding the optimal degree ---");
        RegressionData data = generateRegressionData();

        int maxDegree = 9;
        int optimalDegree = 1;
        double minCvError = Double.MAX_VALUE;

        for (int degree = 1; degree <= maxDegree; degree++) {
            double[][] XTrainPoly = expandPolynomialFeatures(data.XTrain, degree);
            double[][] XCvPoly = expandPolynomialFeatures(data.XCv, degree);

            scaleFeatures(XTrainPoly, XCvPoly);

            Sequential model = new Sequential();
            model.add(new Dense(1, "linear", "output"));
            model.compile("adam", 0.1);

            // Hide epoch outputs for degree search to keep console clean, using 1000 epochs.
            model.fit(XTrainPoly, data.yTrain, 1000);

            double[][] predTrain = model.predict(XTrainPoly);
            double[][] predCv = model.predict(XCvPoly);

            double[] yTrainHat = new double[predTrain.length];
            double[] yTrainTrue = new double[data.yTrain.length];
            for (int i = 0; i < predTrain.length; i++) {
                yTrainHat[i] = predTrain[i][0];
                yTrainTrue[i] = data.yTrain[i][0];
            }

            double[] yCvHat = new double[predCv.length];
            double[] yCvTrue = new double[data.yCv.length];
            for (int i = 0; i < predCv.length; i++) {
                yCvHat[i] = predCv[i][0];
                yCvTrue[i] = data.yCv[i][0];
            }

            double trainErr = evaluateMse(yTrainTrue, yTrainHat);
            double cvErr = evaluateMse(yCvTrue, yCvHat);

            System.out.printf("Degree %d | Train MSE: %.2f | CV MSE: %.2f%n", degree, trainErr, cvErr);
            System.out.println();

            if (cvErr < minCvError) {
                minCvError = cvErr;
                optimalDegree = degree;
            }
        }
        System.out.println("==> The optimal degree is: " + optimalDegree + "\n");
    }

    // Mean Squared Error
    public static double evaluateMse(double[] y, double[] yHat) {
        int m = y.length;
        double err = 0.0;
        for (int i = 0; i < m; i++) {
            double errI = Math.pow(yHat[i] - y[i], 2);
            err += errI;
        }
        err = err / (2.0 * m);
        return err;
    }

    public static double evaluateCategoricalError(double[] y, double[] yHat) {
        int m = y.length;
        int incorrect = 0;
        for (int i = 0; i < m; i++) {
            if (yHat[i] != y[i]) {
                incorrect += 1;
            }
        }
        return (double) incorrect / m;
    }

    public static double calculateCategoricalError(Sequential model, double[][] X, double[][] y) {
        double[][] predictions = model.predict(X);
        double[] yHat = new double[y.length];
        double[] yTrue = new double[y.length];

        for (int i = 0; i < y.length; i++) {
            yTrue[i] = y[i][0];
            int maxIdx = 0;
            double maxVal = predictions[i][0];
            for (int c = 1; c < predictions[i].length; c++) {
                if (predictions[i][c] > maxVal) {
                    maxVal = predictions[i][c];
                    maxIdx = c;
                }
            }
            yHat[i] = maxIdx;
        }
        return evaluateCategoricalError(yTrue, yHat);
    }

    public static Sequential buildComplexModel(int classes) {
        Sequential model = new Sequential();
        model.add(new Dense(120, "relu", "L1"));
        model.add(new Dense(40, "relu", "L2"));
        model.add(new Dense(classes, "linear", "L3"));
        model.compile("adam", 0.01);
        return model;
    }

    public static Sequential buildSimpleModel(int classes) {
        Sequential model = new Sequential();
        model.add(new Dense(6, "relu", "L1"));
        model.add(new Dense(classes, "linear", "L2"));
        model.compile("adam", 0.01);
        return model;
    }

    public static Sequential buildRegularizedModel(int classes) {
        Sequential model = new Sequential();
        model.add(new Dense(120, "relu", 0.1, "L1"));
        model.add(new Dense(40, "relu", 0.1, "L2"));
        model.add(new Dense(classes, "linear", "L3"));
        model.compile("adam", 0.01);
        return model;
    }

    public static void start() {
        System.out.println("Starting C2W3 Assignment: Advice for Applying Machine Learning");

        System.out.println("\nGenerating Dataset (make_blobs)...");
        BlobsData data = generateData();
        System.out.println("Train shape: [" + data.XTrain.length + "], CV shape: [" + data.XCv.length + "], Test shape: [" + data.XTest.length + "]");

        System.out.println("\n--- Complex Model ---");
        Sequential complexModel = buildComplexModel(6);
        complexModel.summary();
        complexModel.fit(data.XTrain, data.yTrain, 500);
        
        System.out.printf("Categorical Error on Train: %.4f%n", calculateCategoricalError(complexModel, data.XTrain, data.yTrain));
        System.out.printf("Categorical Error on CV: %.4f%n", calculateCategoricalError(complexModel, data.XCv, data.yCv));

        System.out.println("\n--- Simple Model ---");
        Sequential simpleModel = buildSimpleModel(6);
        simpleModel.summary();
        simpleModel.fit(data.XTrain, data.yTrain, 500);
        
        System.out.printf("Categorical Error on Train: %.4f%n", calculateCategoricalError(simpleModel, data.XTrain, data.yTrain));
        System.out.printf("Categorical Error on CV: %.4f%n", calculateCategoricalError(simpleModel, data.XCv, data.yCv));

        System.out.println("\n--- Regularized Model ---");
        Sequential regularizedModel = buildRegularizedModel(6);
        regularizedModel.summary();
        regularizedModel.fit(data.XTrain, data.yTrain, 500);
        
        System.out.printf("Categorical Error on Train: %.4f%n", calculateCategoricalError(regularizedModel, data.XTrain, data.yTrain));
        System.out.printf("Categorical Error on CV: %.4f%n", calculateCategoricalError(regularizedModel, data.XCv, data.yCv));

        findOptimalDegree();

        System.out.println("=================================================================\n");
    }
}
