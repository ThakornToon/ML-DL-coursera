package ML.tools;

import java.io.*;

/**
 * Utility class for saving and loading model weights to and from disk.
 */
public final class ModelWeightsIO {

    // Prevent instantiation
    private ModelWeightsIO() {
    }

    // --- SAVE METHODS ---

    // For Neural Network Dense Layer (2D array W, 1D array b) (Core Save Method)
    /**
     * Saves 2D weights and 1D biases to a CSV file.
     * Primarily used for Dense neural network layers.
     *
     * @param filePath The destination file path.
     * @param W        The 2D weight matrix.
     * @param b        The 1D bias array.
     */
    public static void saveWeights(String filePath, double[][] W, double[] b) {
        try (PrintWriter out = new PrintWriter(new FileWriter(filePath))) {
            for (int i = 0; i < b.length; i++) {
                out.print(b[i] + (i == b.length - 1 ? "" : ","));
            }
            out.println();
            for (int i = 0; i < W.length; i++) {
                for (int j = 0; j < W[i].length; j++) {
                    out.print(W[i][j] + (j == W[i].length - 1 ? "" : ","));
                }
                out.println();
            }
            System.out.println("Saved weights to " + filePath);
        } catch (IOException e) {
            System.err.println("Error saving weights: " + e.getMessage());
        }
    }

    // For Logistic Regression (array w, scalar b)
    public static void saveWeights(String filePath, double[] w, double b) {
        saveWeights(filePath, new double[][]{w}, new double[]{b});
    }

    // For Linear Regression (scalar w, scalar b)
    public static void saveWeights(String filePath, double w, double b) {
        saveWeights(filePath, new double[][]{{w}}, new double[]{b});
    }

    // --- LOAD METHODS ---

    // Core Load Method
    /**
     * Loads 2D weights and 1D biases from a CSV file.
     *
     * @param filePath The source file path.
     * @return An Object array containing {double[][] W, double[] b}, or null on failure.
     */
    public static Object[] loadDenseWeights(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String[] bStrs = reader.readLine().split(",");
            double[] b = new double[bStrs.length];
            for (int i = 0; i < bStrs.length; i++) {
                b[i] = Double.parseDouble(bStrs[i]);
            }
            
            java.util.List<double[]> wList = new java.util.ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null && !line.trim().isEmpty()) {
                String[] wStrs = line.split(",");
                double[] wRow = new double[wStrs.length];
                for (int j = 0; j < wStrs.length; j++) {
                    wRow[j] = Double.parseDouble(wStrs[j]);
                }
                wList.add(wRow);
            }
            
            double[][] W = wList.toArray(new double[0][0]);
            System.out.println("Loaded weights from " + filePath);
            return new Object[]{W, b};
        } catch (Exception e) {
            System.err.println("Error loading weights: " + e.getMessage());
            return null;
        }
    }

    // Load Logistic Regression
    public static Object[] loadArrayWeights(String filePath) {
        Object[] loaded = loadDenseWeights(filePath);
        if (loaded != null) {
            double[][] W = (double[][]) loaded[0];
            double[] b = (double[]) loaded[1];
            return new Object[]{W[0], b[0]};
        }
        return null;
    }

    // Load Linear Regression
    public static double[] loadSingleWeights(String filePath) {
        Object[] loaded = loadDenseWeights(filePath);
        if (loaded != null) {
            double[][] W = (double[][]) loaded[0];
            double[] b = (double[]) loaded[1];
            return new double[]{W[0][0], b[0]};
        }
        return null;
    }
}
