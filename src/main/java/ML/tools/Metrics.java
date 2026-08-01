package ML.tools;

public final class Metrics {
    
    private Metrics() {}

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
