package ML;

import ML.tools.Dense;
import ML.tools.Sequential;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

public final class C2W2Assignment {

    // Prevent instantiation
    private C2W2Assignment() {
    }


    public static class DigitData {
        public double[][] X;
        public double[][] y;
        public DigitData(double[][] X, double[][] y) {
            this.X = X;
            this.y = y;
        }
    }

    // I trust AI to load data
    public static DigitData loadData() throws IOException {
        int m = 5000;
        int n = 400;
        double[][] X = new double[m][n];
        double[][] y = new double[m][1];

        // 1. Load X.npy
        byte[] xBytes = Files.readAllBytes(Paths.get("src/main/resources/ML/C2_W2_Assignment_Data/X.npy"));
        
        int xHeaderLen = (xBytes[8] & 0xFF) | ((xBytes[9] & 0xFF) << 8);
        int xOffset = 10 + xHeaderLen;
        
        ByteBuffer xBuffer = ByteBuffer.wrap(xBytes, xOffset, xBytes.length - xOffset);
        xBuffer.order(ByteOrder.LITTLE_ENDIAN);
        
        for (int c = 0; c < n; c++) { 
            for (int r = 0; r < m; r++) { 
                double val = xBuffer.getDouble(); 
                X[r][c] = val;
            }
        }

        // 2. Load y.npy
        byte[] yBytes = Files.readAllBytes(Paths.get("src/main/resources/ML/C2_W2_Assignment_Data/y.npy"));
        
        int yHeaderLen = (yBytes[8] & 0xFF) | ((yBytes[9] & 0xFF) << 8);
        int yOffset = 10 + yHeaderLen;
        
        for (int r = 0; r < m; r++) {
            y[r][0] = yBytes[yOffset + r] & 0xFF;
        }

        return new DigitData(X, y);
    }

    public static void start() {
        System.out.println("Starting C2W2 Assignment: Neural Networks for Multiclass Classification");
        
        try {
            DigitData data = loadData();
            System.out.println("X shape: [" + data.X.length + ", " + data.X[0].length + "]");
            System.out.println("y shape: [" + data.y.length + ", " + data.y[0].length + "]");
            
            // Build Sequential Model
            Sequential model = new Sequential();
            model.add(new Dense(25, "relu", "layer1"));
            model.add(new Dense(15, "relu", "layer2"));
            // Note: Our Sequential model fully supports both "softmax" (from_logits=False) 
            // and "linear" (from_logits=True) for the final layer's activation.
            // 
            // However, we strongly recommend using "linear" (Logits) during training for Numerical Stability:
            // 1. "softmax" (Step-by-step): Calculates Error (dA = -1/prob) then Jacobian (dZ). 
            //    If prob is near 0, `-1/prob` causes Infinity/NaN (Division by Zero) and ruins weights.
            // 2. "linear" (Fused): The math of Cross-Entropy + Softmax Jacobian simplifies beautifully
            //    so that the complex terms cancel out. The formula becomes a simple subtraction:
            //    `dZ = prob - true_label`. This is much faster and completely avoids Infinity/NaN.
            // 
            // If you change this to "softmax", the model will still train, 
            // but you risk floating-point explosions on extreme values.
            model.add(new Dense(10, "linear", "layer3"));
            
            // Compile Model
            model.compile("adam", 0.001); 
            
            System.out.println("\nTraining the model...");
            // Python notebook trains for 40 epochs
            model.fit(data.X, data.y, 40);
            
            System.out.println();
            model.summary();
            
            // Predict an example image
            int imageIndex = 1015; // From assignment: should be a 2
            double[][] image_of_two = new double[][]{data.X[imageIndex]};
            
            double[][] prediction = model.predict(image_of_two);
            System.out.println("\nPredicting image at index " + imageIndex + ":");
            System.out.println("Logits: ");
            for (int j=0; j<10; j++) {
                System.out.printf("%.4f ", prediction[0][j]);
            }
            System.out.println();
            
            double[] prediction_p = ML.tools.Activation.softmax(prediction[0]);
            System.out.println("Probability vector (softmax): ");
            for (int j=0; j<10; j++) {
                System.out.printf("%.4f ", prediction_p[j]);
            }
            System.out.println();
            
            int yhat = 0;
            double maxProb = -1;
            for (int j=0; j<10; j++) {
                if (prediction_p[j] > maxProb) {
                    maxProb = prediction_p[j];
                    yhat = j;
                }
            }
            System.out.println("Largest Prediction index: " + yhat);
            System.out.printf("Actual value: %.0f\n", data.y[imageIndex][0]);
            
            // Random prediction
            Random rand = new Random();
            int randomIndex = rand.nextInt(5000);
            double[][] random_image = new double[][]{data.X[randomIndex]};
            double[][] randPred = model.predict(random_image);
            double[] randProb = ML.tools.Activation.softmax(randPred[0]);
            
            int randYhat = 0;
            double randMaxProb = -1;
            for (int j=0; j<10; j++) {
                if (randProb[j] > randMaxProb) {
                    randMaxProb = randProb[j];
                    randYhat = j;
                }
            }
            System.out.println("\nRandom image at index " + randomIndex + " prediction:");
            System.out.println("Predicted: " + randYhat);
            System.out.printf("Actual value: %.0f\n", data.y[randomIndex][0]);

            // Save weights
            Dense layer1 = model.getLayer("layer1");
            if (layer1 != null) {
                ML.tools.ModelWeightsIO.saveWeights("ml_c2w2_assign_layer1_weights.txt", layer1.getWeightsW(), layer1.getWeightsB());
            }
            Dense layer2 = model.getLayer("layer2");
            if (layer2 != null) {
                ML.tools.ModelWeightsIO.saveWeights("ml_c2w2_assign_layer2_weights.txt", layer2.getWeightsW(), layer2.getWeightsB());
            }
            Dense layer3 = model.getLayer("layer3");
            if (layer3 != null) {
                ML.tools.ModelWeightsIO.saveWeights("ml_c2w2_assign_layer3_weights.txt", layer3.getWeightsW(), layer3.getWeightsB());
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("=================================================================\n");
    }
}
