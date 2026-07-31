package ML;

import java.io.*;

public class ModelWeightsIO {

    // For Logistic Regression (array w, scalar b)
    public static void saveWeights(String filePath, double[] w, double b) {
        try (PrintWriter out = new PrintWriter(new FileWriter(filePath))) {
            out.println(b);
            for (int i = 0; i < w.length; i++) {
                out.print(w[i] + (i == w.length - 1 ? "" : ","));
            }
            out.println();
            System.out.println("Saved weights to " + filePath);
        } catch (IOException e) {
            System.err.println("Error saving weights: " + e.getMessage());
        }
    }

    public static Object[] loadArrayWeights(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            double b = Double.parseDouble(reader.readLine());
            String[] wStrs = reader.readLine().split(",");
            double[] w = new double[wStrs.length];
            for (int i = 0; i < wStrs.length; i++) {
                w[i] = Double.parseDouble(wStrs[i]);
            }
            System.out.println("Loaded weights from " + filePath);
            return new Object[]{w, b};
        } catch (Exception e) {
            System.err.println("Error loading weights: " + e.getMessage());
            return null;
        }
    }

    // For Linear Regression (scalar w, scalar b)
    public static void saveWeights(String filePath, double w, double b) {
        try (PrintWriter out = new PrintWriter(new FileWriter(filePath))) {
            out.println(b);
            out.println(w);
            System.out.println("Saved weights to " + filePath);
        } catch (IOException e) {
            System.err.println("Error saving weights: " + e.getMessage());
        }
    }

    public static double[] loadSingleWeights(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            double b = Double.parseDouble(reader.readLine());
            double w = Double.parseDouble(reader.readLine());
            System.out.println("Loaded weights from " + filePath);
            return new double[]{w, b};
        } catch (Exception e) {
            System.err.println("Error loading weights: " + e.getMessage());
            return null;
        }
    }
}
