package ML;

import ML.tools.Dense;
import ML.tools.Sequential;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

public final class C2W1Assignment {

    // Prevent instantiation
    private C2W1Assignment() {
    }

    public static class DigitData {
        public double[][] X;
        public double[][] y;
        public DigitData(double[][] X, double[][] y) {
            this.X = X;
            this.y = y;
        }
    }

    public static DigitData loadData() throws IOException {
        int m = 1000;
        int n = 400;
        double[][] X = new double[m][n];
        double[][] y = new double[m][1];

        String basePath = "src/main/resources/ML/C2_W1_Assignment_Data/";

        // 1. โหลดไฟล์ X.npy (ข้อมูลรูปภาพ) - โหลดทั้งหมด 5000 แถว แล้วเลือกมาแค่ 1000 แถวแรก
        double[][] X_full = ML.tools.NumpyIO.loadDoubleMatrix(basePath + "X.npy", 5000, 400, true);
        for (int r = 0; r < m; r++) {
            X[r] = X_full[r];
        }

        // 2. โหลดไฟล์ y.npy (ข้อมูลเฉลย)
        int[] y_full = ML.tools.NumpyIO.loadUint8Array(basePath + "y.npy", 5000);
        for (int r = 0; r < m; r++) {
            y[r][0] = y_full[r];
        }

        return new DigitData(X, y);
    }

    public static void start() {
        System.out.println("Starting C2W1 Assignment: Neural Networks for Handwritten Digit Recognition");
        
        try {
            DigitData data = loadData();
            System.out.println("X shape: [" + data.X.length + ", " + data.X[0].length + "]");
            System.out.println("y shape: [" + data.y.length + ", " + data.y[0].length + "]");
            
            // Build Sequential Model
            Sequential model = new Sequential();
            model.add(new Dense(25, "sigmoid", "layer1"));
            model.add(new Dense(15, "sigmoid", "layer2"));
            model.add(new Dense(1, "sigmoid", "layer3"));
            
            // Compile Model
            model.compile("adam", 0.001); // Keras defaults
            
            System.out.println("\nTraining the model...");
            // The python lab trains for 20 epochs using mini-batches of size 32.
            // Our Java implementation uses full-batch gradient descent. 
            // We need more epochs to converge. 2000 epochs should be sufficient.
            model.fit(data.X, data.y, 2000);
            
            System.out.println();
            model.summary();
            
            // Test on a few examples
            System.out.println("\nEvaluating on first 5 (should be 0) and last 5 (should be 1) of the 1000 examples:");
            
            for (int i = 0; i < 5; i++) {
                double[][] pred = model.predict(new double[][]{data.X[i]});
                int yhat = (pred[0][0] >= 0.5) ? 1 : 0;
                System.out.printf("Example %d: Prob=%.4f, Pred=%d, Actual=%.0f%n", i, pred[0][0], yhat, data.y[i][0]);
            }
            for (int i = 995; i < 1000; i++) {
                double[][] pred = model.predict(new double[][]{data.X[i]});
                int yhat = (pred[0][0] >= 0.5) ? 1 : 0;
                System.out.printf("Example %d: Prob=%.4f, Pred=%d, Actual=%.0f%n", i, pred[0][0], yhat, data.y[i][0]);
            }

            // Save weights
            Dense layer1 = model.getLayer("layer1");
            if (layer1 != null) {
                ML.tools.ModelWeightsIO.saveWeights("ml_c2w1_assign_layer1_weights.txt", layer1.getWeightsW(), layer1.getWeightsB());
            }
            Dense layer2 = model.getLayer("layer2");
            if (layer2 != null) {
                ML.tools.ModelWeightsIO.saveWeights("ml_c2w1_assign_layer2_weights.txt", layer2.getWeightsW(), layer2.getWeightsB());
            }
            Dense layer3 = model.getLayer("layer3");
            if (layer3 != null) {
                ML.tools.ModelWeightsIO.saveWeights("ml_c2w1_assign_layer3_weights.txt", layer3.getWeightsW(), layer3.getWeightsB());
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("=================================================================\n");
    }
}
