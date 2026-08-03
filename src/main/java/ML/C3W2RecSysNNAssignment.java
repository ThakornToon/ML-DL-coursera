package ML;

import ML.tools.Dense;
import ML.tools.Sequential;
import java.util.Arrays;

/**
 * Implementation of the Neural Network Recommender Systems Assignment (C3W2).
 *
 * This assignment implements a Two-Tower Neural Network architecture for 
 * Recommender Systems, which computes user and item vectors and uses their 
 * dot product (or squared distance) for predictions.
 * 
 * Note: Training a two-tower model requires an Auto-Differentiation engine 
 * to handle the custom Dot layer and L2 normalization gradients. This Java 
 * implementation focuses on the architecture definition, forward pass concepts,
 * and the graded sq_dist function.
 */
public final class C3W2RecSysNNAssignment {

    private C3W2RecSysNNAssignment() {}

    public static void start() {
        System.out.println("Neural Network Recommender System (Two-Tower Model)");
        
        String basePath = "src/main/resources/ML/C3_W2_RecSysNN_Assignment_Data/";
        
        System.out.println("Loading data...");
        // In python notebook, u_s = 3 (drop 3 cols), i_s = 1 (drop 1 col)
        double[][] userTrain = loadCSVMatrix(basePath + "content_user_train.csv", 3);
        double[][] itemTrain = loadCSVMatrix(basePath + "content_item_train.csv", 1);
        double[][] yTrain = loadCSVMatrix(basePath + "content_y_train.csv", 0);
        
        System.out.println("User train size: " + userTrain.length + "x" + userTrain[0].length);
        System.out.println("Item train size: " + itemTrain.length + "x" + itemTrain[0].length);
        System.out.println("Y train size: " + yTrain.length + "x" + yTrain[0].length);
        
        // Scale data
        System.out.println("Scaling data...");
        ML.tools.Normalization userScaler = new ML.tools.Normalization();
        userScaler.adapt(userTrain);
        for(int i = 0; i < userTrain.length; i++) {
            for(int j = 0; j < userTrain[0].length; j++) {
                double variance = userScaler.getVariance()[j];
                userTrain[i][j] = (userTrain[i][j] - userScaler.getMean()[j]) / Math.sqrt(variance == 0 ? 1 : variance);
            }
        }
        
        ML.tools.Normalization itemScaler = new ML.tools.Normalization();
        itemScaler.adapt(itemTrain);
        for(int i = 0; i < itemTrain.length; i++) {
            for(int j = 0; j < itemTrain[0].length; j++) {
                double variance = itemScaler.getVariance()[j];
                itemTrain[i][j] = (itemTrain[i][j] - itemScaler.getMean()[j]) / Math.sqrt(variance == 0 ? 1 : variance);
            }
        }
        
        minMaxScaler(yTrain);
        
        // Take a small sample to train quickly (e.g., 500 rows) just to prove it works
        int sampleSize = Math.min(500, userTrain.length);
        double[][] uTrainSub = Arrays.copyOfRange(userTrain, 0, sampleSize);
        double[][] iTrainSub = Arrays.copyOfRange(itemTrain, 0, sampleSize);
        double[][] yTrainSub = Arrays.copyOfRange(yTrain, 0, sampleSize);
        
        System.out.println("Using a sample of " + sampleSize + " for demonstration training.");
        
        // 1. Define the User Neural Network (user_NN)
        Sequential userNN = new Sequential();
        userNN.add(new Dense(256, "relu", "user_layer1"));
        userNN.add(new Dense(128, "relu", "user_layer2"));
        userNN.add(new Dense(32, "linear", "user_layer3"));
        
        // 2. Define the Item Neural Network (item_NN)
        Sequential itemNN = new Sequential();
        itemNN.add(new Dense(256, "relu", "item_layer1"));
        itemNN.add(new Dense(128, "relu", "item_layer2"));
        itemNN.add(new Dense(32, "linear", "item_layer3"));
        
        // Train
        trainTwoTower(userNN, itemNN, uTrainSub, iTrainSub, yTrainSub, 5, 0.01);
    }

    public static void trainTwoTower(Sequential userNN, Sequential itemNN, 
                                     double[][] userX, double[][] itemX, double[][] Y, 
                                     int epochs, double learningRate) {
        int m = userX.length;
        int t = 1;
        String optimizer = "adam";
        int numOutputs = 32;
        
        System.out.println("\nStarting Custom Two-Tower Training for " + epochs + " epochs...");
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            // --- Forward Pass ---
            double[][] zu = userX;
            for (int l = 0; l < userNN.getLayers().size(); l++) {
                zu = userNN.getLayers().get(l).forwardBatch(zu);
            }
            
            double[][] zm = itemX;
            for (int l = 0; l < itemNN.getLayers().size(); l++) {
                zm = itemNN.getLayers().get(l).forwardBatch(zm);
            }
            
            double[][] vu = new double[m][numOutputs];
            double[][] vm = new double[m][numOutputs];
            double[] y_pred = new double[m];
            
            double loss = 0;
            
            for (int i = 0; i < m; i++) {
                vu[i] = l2Normalize(zu[i]);
                vm[i] = l2Normalize(zm[i]);
                
                y_pred[i] = dotProduct(vu[i], vm[i]);
                loss += Math.pow(y_pred[i] - Y[i][0], 2);
            }
            loss /= (2.0 * m);
            
            System.out.printf("Epoch %d/%d - loss: %.4f%n", epoch, epochs, loss);
            
            // --- Backward Pass ---
            double[][] dL_dzu = new double[m][numOutputs];
            double[][] dL_dzm = new double[m][numOutputs];

            // u -> ... -> zu -> vu -
            //                        \
            //                          Y dot product <--compare--> Ans -> Loss
            //                        /
            // m -> ... -> zm -> vm -

            // For Each train data
            for (int i = 0; i < m; i++) {
                // Average loss each data
                double dL_dy = (y_pred[i] - Y[i][0]) / m; 
                
                double[] dL_dvu = new double[numOutputs];
                double[] dL_dvm = new double[numOutputs];

                // For Each node output before dot
                // Y = vu * vm -> dL/dY * dY/dvu = dL/dvu
                for(int j = 0; j < numOutputs; j++) {
                    dL_dvu[j] = dL_dy * vm[i][j];
                    dL_dvm[j] = dL_dy * vu[i][j];
                }

                // Diff through L2 Normalization
                double norm_u = norm2(zu[i]);
                double dot_u = dotProduct(dL_dvu, vu[i]);
                for(int j = 0; j < numOutputs; j++) {
                    dL_dzu[i][j] = (norm_u == 0) ? 0 : (dL_dvu[j] - dot_u * vu[i][j]) / norm_u;
                }
                
                double norm_m = norm2(zm[i]);
                double dot_m = dotProduct(dL_dvm, vm[i]);
                for(int j = 0; j < numOutputs; j++) {
                    dL_dzm[i][j] = (norm_m == 0) ? 0 : (dL_dvm[j] - dot_m * vm[i][j]) / norm_m;
                }
            }
            
            // Backward pass through User NN
            double[][] dA_u = dL_dzu;
            for (int l = userNN.getLayers().size() - 1; l >= 0; l--) {
                dA_u = userNN.getLayers().get(l).backward(dA_u, learningRate, optimizer, t);
            }
            
            // Backward pass through Item NN
            double[][] dA_m = dL_dzm;
            for (int l = itemNN.getLayers().size() - 1; l >= 0; l--) {
                dA_m = itemNN.getLayers().get(l).backward(dA_m, learningRate, optimizer, t);
            }
            
            t++;
        }
    }

    public static double[][] loadCSVMatrix(String path, int dropCols) {
        java.util.List<double[]> data = new java.util.ArrayList<>();
        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] values = line.split(",");
                double[] row = new double[values.length - dropCols];
                for (int i = dropCols; i < values.length; i++) {
                    row[i - dropCols] = Double.parseDouble(values[i].trim());
                }
                data.add(row);
            }
        } catch (java.io.IOException e) {
            System.err.println("Error reading " + path + ": " + e.getMessage());
        }
        return data.toArray(new double[0][]);
    }

    public static double[] minMaxScaler(double[][] Y) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < Y.length; i++) {
            if (Y[i][0] < min) min = Y[i][0];
            if (Y[i][0] > max) max = Y[i][0];
        }
        for (int i = 0; i < Y.length; i++) {
            Y[i][0] = 2.0 * (Y[i][0] - min) / (max - min) - 1.0;
        }
        return new double[]{min, max};
    }

    private static double norm2(double[] v) {
        double sumSq = 0.0;
        for (double val : v) {
            sumSq += val * val;
        }
        return Math.sqrt(sumSq);
    }
    
    /**
     * L2 Normalizes a vector.
     */
    private static double[] l2Normalize(double[] v) {
        double sumSq = 0.0;
        for (double val : v) {
            sumSq += val * val;
        }
        double norm = Math.sqrt(sumSq);
        if (norm == 0.0) return v;
        
        double[] res = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            res[i] = v[i] / norm;
        }
        return res;
    }
    
    /**
     * Computes the dot product of two vectors.
     */
    private static double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
