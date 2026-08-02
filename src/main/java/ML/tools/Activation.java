package ML.tools;

/**
 * Utility class for common neural network activation functions.
 */
public final class Activation {

    // Prevent instantiation
    private Activation() {
    }

    /**
     * Computes the Sigmoid activation function.
     * Maps the input value to a range between 0 and 1.
     *
     * @param z The input value.
     * @return The sigmoid of z.
     */
    public static double sigmoid(double z) {
        // Prevent overflow by bounding z
        if (z < -40.0) return 0.0;
        if (z > 40.0) return 1.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Computes the Rectified Linear Unit (ReLU) activation function.
     * Returns the maximum of 0 and the input value.
     *
     * @param z The input value.
     * @return The ReLU of z.
     */
    public static double relu(double z) {
        return Math.max(0.0, z);
    }

    /**
     * Computes the Leaky ReLU activation function.
     * Allows a small, positive gradient when the unit is not active.
     *
     * @param z The input value.
     * @return The Leaky ReLU of z.
     */
    public static double leaky_relu(double z) {
        return z > 0 ? z : 0.01 * z;
    }

    /**
     * Computes the Linear activation function (identity function).
     *
     * @param z The input value.
     * @return The exact input value z.
     */
    public static double linear(double z) {
        return z;
    }

    /**
     * Computes the Softmax activation function.
     * Converts an array of unnormalized log probabilities to a probability distribution.
     *
     * @param z The input array of values.
     * @return An array of probabilities summing to 1.
     */
    public static double[] softmax(double[] z) {
        double maxZ = -Double.MAX_VALUE;
        for (double v : z) {
            if (v > maxZ) maxZ = v;
        }
        
        double[] a = new double[z.length];
        double sum = 0.0;
        for (int i = 0; i < z.length; i++) {
            a[i] = Math.exp(z[i] - maxZ);
            sum += a[i];
        }
        for (int i = 0; i < z.length; i++) {
            a[i] /= sum;
        }
        return a;
    }

    /**
     * Applies a specified activation function to a single value.
     *
     * @param z              The input value.
     * @param activationName The name of the activation function (e.g., "relu", "sigmoid").
     * @return The activated value.
     */
    public static double apply(double z, String activationName) {
        switch (activationName.toLowerCase()) {
            case "sigmoid":
                return sigmoid(z);
            case "relu":
                return relu(z);
            case "leaky_relu":
                return leaky_relu(z);
            case "softmax":
                // Softmax on a single value is always 1.0
                return 1.0;
            case "linear":
            default:
                return linear(z);
        }
    }

    public static double[] apply(double[] z, String activationName) {
        if (activationName.equalsIgnoreCase("softmax")) {
            return softmax(z);
        }
        double[] a = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            a[i] = apply(z[i], activationName);
        }
        return a;
    }
    
    /**
     * Computes the derivative of the activation function with respect to z.
     */
    public static double derivative(double a, double z, String activationName) {
        switch (activationName.toLowerCase()) {
            case "sigmoid":
            case "softmax":
                // For sigmoid, derivative is a * (1 - a)
                // For softmax, this is the diagonal of the Jacobian matrix
                return a * (1.0 - a);
            case "relu":
                return z > 0 ? 1.0 : 0.0;
            case "leaky_relu":
                return z > 0 ? 1.0 : 0.01;
            case "linear":
            default:
                return 1.0;
        }
    }

    /**
     * Computes dZ (dL/dZ) for a whole layer, handling full Jacobian for Softmax.
     */
    public static double[] compute_dZ(double[] dA, double[] a, double[] z, String activationName) {
        double[] dZ = new double[a.length];
        if (activationName.equalsIgnoreCase("softmax")) {
            // Full Jacobian for Softmax: dZ_k = a_k * (dA_k - sum_j(dA_j * a_j))
            double sum_dA_a = 0.0;
            for (int j = 0; j < a.length; j++) {
                sum_dA_a += dA[j] * a[j];
            }
            for (int k = 0; k < a.length; k++) {
                dZ[k] = a[k] * (dA[k] - sum_dA_a);
            }
            return dZ;
        }
        // Element-wise for others
        // dZ = dA * g'(Z)
        // A_in -> Z = WA + b -> A_out = g(Z)
        // dL/dZ = dL/dA * dA/dZ = dL/dA * g'(Z)
        for (int j = 0; j < a.length; j++) {
            dZ[j] = dA[j] * derivative(a[j], z[j], activationName);
        }
        return dZ;
    }
}
