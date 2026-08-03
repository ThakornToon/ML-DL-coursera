package ML;

import java.io.*;
import java.util.*;

/**
 * Implementation of the Collaborative Filtering Recommender Systems Assignment (C3W2).
 * 
 * Implementation Notes:
 * 1. Why we don't use DataFrame.java: The data files in this lab (e.g., small_movies_Y.csv) 
 *    are purely numerical without a header row. DataFrame.java is designed to parse the 
 *    first line as a header. Thus, loading the data via a custom loadCSVMatrix method 
 *    is both more efficient and structurally correct for this specific case.
 * 2. Why we don't use Sequential / Dense from ML.tools:
 *    Collaborative Filtering requires a highly specific custom cost function (only computing 
 *    the error when R[i][j] == 1) and simultaneously updates two sets of weights (W for users 
 *    and X for movies). Because our Java project lacks an Auto-Differentiation engine 
 *    like TensorFlow's tf.GradientTape, we have to manually derive and implement the gradients 
 *    and the Adam Optimizer loop. This is why the code appears longer than its Python counterpart.
 */
/**
 * Implementation of the C3W2 Collaborative Filtering Recommender System Assignment.
 * Deals with computing cost functions and gradients for collaborative filtering.
 */
public final class C3W2CollaborativeRecSysAssignment {

    private C3W2CollaborativeRecSysAssignment() {}

    /**
     * Executes the collaborative filtering recommender system demonstration.
     */
    public static void start() {
        System.out.println("Collaborative Filtering Recommender System...");
        String basePath = "src/main/resources/ML/C3_W2_Collaborative_RecSys_Assignment_Data/";

        try {
            // 1. Load Data
            double[][] Y = loadCSVMatrix(basePath + "small_movies_Y.csv");
            double[][] R = loadCSVMatrix(basePath + "small_movies_R.csv");
            double[][] X = loadCSVMatrix(basePath + "small_movies_X.csv");
            double[][] W = loadCSVMatrix(basePath + "small_movies_W.csv");
            double[][] b = loadCSVMatrix(basePath + "small_movies_b.csv");

            int num_movies = Y.length;
            int num_users = Y[0].length;

            System.out.println("Y shape: (" + num_movies + ", " + num_users + ")");
            System.out.println("R shape: (" + R.length + ", " + R[0].length + ")");
            System.out.println("X shape: (" + X.length + ", " + X[0].length + ")");
            System.out.println("W shape: (" + W.length + ", " + W[0].length + ")");
            System.out.println("b shape: (" + b.length + ", " + b[0].length + ")");

            // Check cost function with sub-arrays to match expected output
            double[][] X_r = subMatrix(X, 5, 3);
            double[][] W_r = subMatrix(W, 4, 3);
            double[][] b_r = subMatrix(b, 1, 4);
            double[][] Y_r = subMatrix(Y, 5, 4);
            double[][] R_r = subMatrix(R, 5, 4);

            double J = cofiCostFunc(X_r, W_r, b_r, Y_r, R_r, 0.0);
            System.out.printf("Cost (lambda = 0): %.2f (Expected ~ 13.67)\n", J);
            double J_reg = cofiCostFunc(X_r, W_r, b_r, Y_r, R_r, 1.5);
            System.out.printf("Cost (with regularization): %.2f (Expected ~ 28.09)\n", J_reg);

            // 2. Add my ratings
            double[] my_ratings = new double[num_movies];
            my_ratings[2700] = 5; // Toy Story 3
            my_ratings[2609] = 2; // Persuasion
            my_ratings[929] = 5;  // LOTR: Return of the King
            my_ratings[246] = 5;  // Shrek
            my_ratings[2716] = 3; // Inception
            my_ratings[1150] = 5; // Incredibles
            my_ratings[382] = 2;  // Amelie
            my_ratings[366] = 5;  // Harry Potter 1
            my_ratings[622] = 5;  // Harry Potter 2
            my_ratings[988] = 3;  // Eternal Sunshine
            my_ratings[2925] = 1; // Louis Theroux
            my_ratings[2937] = 1; // Nothing to Declare
            my_ratings[793] = 5;  // Pirates

            // Append my_ratings to Y, R (as the first column)
            num_users++;
            double[][] Y_new = new double[num_movies][num_users];
            double[][] R_new = new double[num_movies][num_users];
            for(int i = 0; i < num_movies; i++) {
                Y_new[i][0] = my_ratings[i];
                R_new[i][0] = my_ratings[i] > 0 ? 1 : 0;
                for(int j = 0; j < num_users - 1; j++) {
                    Y_new[i][j+1] = Y[i][j];
                    R_new[i][j+1] = R[i][j];
                }
            }
            Y = Y_new;
            R = R_new;

            // 3. Normalize Ratings
            double[][] Ynorm = new double[num_movies][num_users];
            double[] Ymean = new double[num_movies];
            for (int i = 0; i < num_movies; i++) {
                double sum = 0;
                int count = 0;
                for (int j = 0; j < num_users; j++) {
                    if (R[i][j] == 1) {
                        sum += Y[i][j];
                        count++;
                    }
                }
                if (count > 0) {
                    Ymean[i] = sum / count;
                    for (int j = 0; j < num_users; j++) {
                        if (R[i][j] == 1) {
                            Ynorm[i][j] = Y[i][j] - Ymean[i];
                        }
                    }
                }
            }

            // 4. Setup parameters
            int num_features = 100;
            String trainedXPath = "ml_c3w2_collaborative_X_weights.txt";
            String trainedWPath = "ml_c3w2_collaborative_W_weights.txt";
            String trainedBPath = "ml_c3w2_collaborative_b_weights.txt";
            
            boolean usePretrained = true; // Set to false to force re-training
            
            if (usePretrained && new File(trainedXPath).exists()) {
                System.out.println("\nLoading pre-trained model weights from CSV files...");
                X = loadCSVMatrix(trainedXPath);
                W = loadCSVMatrix(trainedWPath);
                b = loadCSVMatrix(trainedBPath);
            } else {
                System.out.println("\nInitializing new parameters for training...");
                Random random = new Random(1234);
                X = randomMatrix(num_movies, num_features, random);
                W = randomMatrix(num_users, num_features, random);
                b = new double[1][num_users];
    
                // 5. Train
                double lambda = 1.0;
                double alpha = 0.1;
                int iterations = 200;
    
                System.out.println("Training Collaborative Filtering model (Adam Optimizer)...");
    
                double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7;
                double[][] vdw = new double[num_users][num_features];
                double[][] sdw = new double[num_users][num_features];
                double[][] vdx = new double[num_movies][num_features];
                double[][] sdx = new double[num_movies][num_features];
                double[] vdb = new double[num_users];
                double[] sdb = new double[num_users];
    
                for (int iter = 1; iter <= iterations; iter++) {
                    double[][] dW = new double[num_users][num_features];
                    double[][] dX = new double[num_movies][num_features];
                    double[] db = new double[num_users];
    
                    // Compute gradients
                    for (int i = 0; i < num_movies; i++) {
                        for (int j = 0; j < num_users; j++) {
                            if (R[i][j] == 1) {
                                double pred = b[0][j];
                                for (int k = 0; k < num_features; k++) pred += W[j][k] * X[i][k];
    
                                double error = pred - Ynorm[i][j];
    
                                for (int k = 0; k < num_features; k++) {
                                    dW[j][k] += error * X[i][k];
                                    dX[i][k] += error * W[j][k];
                                }
                                db[j] += error;
                            }
                        }
                    }
    
                    // Add regularization to dW and dX
                    for(int j=0; j<num_users; j++) {
                        for(int k=0; k<num_features; k++) {
                            dW[j][k] += lambda * W[j][k];
                        }
                    }
                    for(int i=0; i<num_movies; i++) {
                        for(int k=0; k<num_features; k++) {
                            dX[i][k] += lambda * X[i][k];
                        }
                    }
    
                    // Adam Update
                    for(int j=0; j<num_users; j++) {
                        vdb[j] = beta1 * vdb[j] + (1 - beta1) * db[j];
                        sdb[j] = beta2 * sdb[j] + (1 - beta2) * (db[j] * db[j]);
                        double vdb_corr = vdb[j] / (1 - Math.pow(beta1, iter));
                        double sdb_corr = sdb[j] / (1 - Math.pow(beta2, iter));
                        b[0][j] -= alpha * vdb_corr / (Math.sqrt(sdb_corr) + epsilon);
    
                        for(int k=0; k<num_features; k++) {
                            vdw[j][k] = beta1 * vdw[j][k] + (1 - beta1) * dW[j][k];
                            sdw[j][k] = beta2 * sdw[j][k] + (1 - beta2) * (dW[j][k] * dW[j][k]);
                            double vdw_corr = vdw[j][k] / (1 - Math.pow(beta1, iter));
                            double sdw_corr = sdw[j][k] / (1 - Math.pow(beta2, iter));
                            W[j][k] -= alpha * vdw_corr / (Math.sqrt(sdw_corr) + epsilon);
                        }
                    }
                    for(int i=0; i<num_movies; i++) {
                        for(int k=0; k<num_features; k++) {
                            vdx[i][k] = beta1 * vdx[i][k] + (1 - beta1) * dX[i][k];
                            sdx[i][k] = beta2 * sdx[i][k] + (1 - beta2) * (dX[i][k] * dX[i][k]);
                            double vdx_corr = vdx[i][k] / (1 - Math.pow(beta1, iter));
                            double sdx_corr = sdx[i][k] / (1 - Math.pow(beta2, iter));
                            X[i][k] -= alpha * vdx_corr / (Math.sqrt(sdx_corr) + epsilon);
                        }
                    }
    
                    if (iter % 20 == 0) {
                        System.out.println("Iteration " + iter + " completed.");
                    }
                }
                System.out.println("Training complete.");
                
                System.out.println("Saving trained weights to CSV files...");
                saveCSVMatrix(X, trainedXPath);
                saveCSVMatrix(W, trainedWPath);
                saveCSVMatrix(b, trainedBPath);
            }

            // 6. Predict for user 0 (my_ratings)
            double[] my_predictions = new double[num_movies];
            for (int i = 0; i < num_movies; i++) {
                double pred = b[0][0];
                for (int k = 0; k < num_features; k++) {
                    pred += W[0][k] * X[i][k];
                }
                my_predictions[i] = pred + Ymean[i];
            }

            // Load movie names
            String[] movieNames = loadMovieNames(basePath + "small_movie_list.csv");

            // Print Top Recommendations
            System.out.println("\nTop recommendations for you:");
            Integer[] indices = new Integer[num_movies];
            for (int i = 0; i < num_movies; i++) indices[i] = i;

            Arrays.sort(indices, (i1, i2) -> Double.compare(my_predictions[i2], my_predictions[i1]));

            int count = 0;
            for (int i = 0; i < num_movies && count < 15; i++) {
                int idx = indices[i];
                if (my_ratings[idx] == 0) { // Not rated by me
                    System.out.printf("Predicting rating %.2f for movie %s\n", my_predictions[idx], movieNames[idx]);
                    count++;
                }
            }

            System.out.println("\nOriginal vs Predicted ratings:");
            for (int i = 0; i < num_movies; i++) {
                if (my_ratings[i] > 0) {
                    System.out.printf("Original rating %.1f, Predicted rating %.2f for %s\n", my_ratings[i], my_predictions[i], movieNames[i]);
                }
            }
            System.out.println("=================================================================\n");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double cofiCostFunc(double[][] X, double[][] W, double[][] b, double[][] Y, double[][] R, double lambda) {
        double J = 0;
        int num_movies = Y.length;
        int num_users = Y[0].length;
        int num_features = X[0].length;

        for (int i = 0; i < num_movies; i++) {
            for (int j = 0; j < num_users; j++) {
                if (R[i][j] == 1) {
                    double pred = b[0][j];
                    for (int k = 0; k < num_features; k++) {
                        pred += W[j][k] * X[i][k];
                    }
                    J += Math.pow(pred - Y[i][j], 2);
                }
            }
        }
        J = J / 2.0;

        double regW = 0;
        for (int j = 0; j < num_users; j++) {
            for (int k = 0; k < num_features; k++) {
                regW += W[j][k] * W[j][k];
            }
        }
        double regX = 0;
        for (int i = 0; i < num_movies; i++) {
            for (int k = 0; k < num_features; k++) {
                regX += X[i][k] * X[i][k];
            }
        }

        J += (lambda / 2.0) * (regW + regX);
        return J;
    }

    private static double[][] subMatrix(double[][] mat, int rows, int cols) {
        double[][] sub = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sub[i][j] = mat[i][j];
            }
        }
        return sub;
    }

    private static double[][] randomMatrix(int rows, int cols, Random random) {
        double[][] mat = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                // Initialize to small random values to match normal distribution (0, 1) or smaller
                // Setting to 0.1 * gaussian to help with stability initially
                mat[i][j] = 0.1 * random.nextGaussian();
            }
        }
        return mat;
    }

    private static double[][] loadCSVMatrix(String path) throws IOException {
        List<double[]> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                list.add(row);
            }
        }
        return list.toArray(new double[0][0]);
    }

    private static String[] loadMovieNames(String path) throws IOException {
        List<String> names = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",", 4);
                if (parts.length >= 4) {
                    String title = parts[3];
                    if (title.startsWith("\"") && title.endsWith("\"")) {
                        title = title.substring(1, title.length() - 1);
                    }
                    names.add(title);
                } else {
                    names.add("Unknown Title");
                }
            }
        }
        return names.toArray(new String[0]);
    }

    private static void saveCSVMatrix(double[][] matrix, String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            for (double[] row : matrix) {
                StringBuilder line = new StringBuilder();
                for (int i = 0; i < row.length; i++) {
                    line.append(row[i]);
                    if (i < row.length - 1) line.append(",");
                }
                bw.write(line.toString());
                bw.newLine();
            }
        }
    }
}
