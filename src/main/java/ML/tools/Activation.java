package ML.tools;

/**
 * Utility class for common neural network activation functions.
 */
public final class Activation {

    // Prevent instantiation
    private Activation() {
    }

    public static double sigmoid(double z) {
        // Prevent overflow by bounding z
        if (z < -40.0) return 0.0;
        if (z > 40.0) return 1.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double relu(double z) {
        return Math.max(0.0, z);
    }

    public static double linear(double z) {
        return z;
    }

    public static double apply(double z, String activationName) {
        switch (activationName.toLowerCase()) {
            case "sigmoid":
                return sigmoid(z);
            case "relu":
                return relu(z);
            case "linear":
            default:
                return linear(z);
        }
    }
    
    /**
     * Computes the derivative of the activation function with respect to z.
     */
    public static double derivative(double a, double z, String activationName) {
        switch (activationName.toLowerCase()) {
            case "sigmoid":
                // For sigmoid, derivative is a * (1 - a)
                return a * (1.0 - a);
            case "relu":
                return z > 0 ? 1.0 : 0.0;
            case "linear":
            default:
                return 1.0;
        }
    }
}
