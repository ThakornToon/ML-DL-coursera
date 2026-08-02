package ML.tools;

/**
 * Utility class for calculating machine learning evaluation metrics.
 */
public final class Metrics {
    
    private Metrics() {}

    /**
     * Calculates the accuracy score for classification tasks.
     *
     * @param yTrue The ground truth (correct) labels.
     * @param yPred The predicted labels.
     * @return The proportion of correctly predicted labels.
     */
    public static double accuracyScore(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Lengths must match");
        }
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (Double.compare(yTrue[i], yPred[i]) == 0) {
                correct++;
            }
        }
        return (double) correct / yTrue.length;
    }

    /**
     * Calculates the Mean Squared Error (MSE) for regression tasks.
     *
     * @param yTrue The ground truth target values.
     * @param yPred The predicted target values.
     * @return The average of the squares of the errors.
     */
    public static double meanSquaredError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Lengths must match");
        }
        double err = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            err += Math.pow(yPred[i] - yTrue[i], 2);
        }
        return err / yTrue.length;
    }

    public static double halfMeanSquaredError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Lengths must match");
        }
        double err = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            err += Math.pow(yPred[i] - yTrue[i], 2);
        }
        return err / (2.0 * yTrue.length);
    }

    public static double categoricalError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Lengths must match");
        }
        int incorrect = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (Double.compare(yTrue[i], yPred[i]) != 0) {
                incorrect++;
            }
        }
        return (double) incorrect / yTrue.length;
    }
}
