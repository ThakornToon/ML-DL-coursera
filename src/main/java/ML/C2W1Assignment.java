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

        // 1. โหลดไฟล์ X.npy (ข้อมูลรูปภาพ)
        byte[] xBytes = Files.readAllBytes(Paths.get("src/main/resources/C2_W1_Assignment_Data/X.npy"));
        
        // อ่านความยาวของ Header จากไบต์ที่ 8 และ 9 (Little-Endian)
        // ใช้ & 0xFF เพื่อป้องกันค่าติดลบเมื่อแปลงจาก byte เป็น int
        int xHeaderLen = (xBytes[8] & 0xFF) | ((xBytes[9] & 0xFF) << 8);
        
        // ข้อมูลตัวเลขจริงๆ จะเริ่มต้นที่ไบต์ที่ 10 + ความยาวของ Header
        int xOffset = 10 + xHeaderLen;
        
        // ใช้ ByteBuffer เพื่อความสะดวกในการแปลงไบต์เป็น double (64-bit float)
        ByteBuffer xBuffer = ByteBuffer.wrap(xBytes, xOffset, xBytes.length - xOffset);
        // ไฟล์ถูกเซฟมาแบบ Little-Endian จึงต้องตั้งค่า Order ให้ตรงกัน
        xBuffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // ไฟล์ X.npy ถูกบันทึกมาในรูปแบบ Fortran order (Column-Major) 
        // ซึ่งแปลว่าข้อมูลถูกเรียงต่อกันตาม "คอลัมน์" (ไม่ใช่ตามแถวเหมือนปกติ)
        // ไฟล์ต้นฉบับมีทั้งหมด 5000 แถว และ 400 คอลัมน์
        for (int c = 0; c < 400; c++) { // ลูปคอลัมน์อยู่ด้านนอก
            for (int r = 0; r < 5000; r++) { // ลูปแถวอยู่ด้านใน
                double val = xBuffer.getDouble(); // อ่านค่า 8 ไบต์ แปลงเป็น double
                
                // โจทย์ระบุว่าเราต้องการแค่เลข 0 และ 1 ซึ่งอยู่ใน 1000 แถวแรก
                if (r < m) {
                    X[r][c] = val;
                }
            }
        }

        // 2. โหลดไฟล์ y.npy (ข้อมูลเฉลย)
        byte[] yBytes = Files.readAllBytes(Paths.get("src/main/resources/C2_W1_Assignment_Data/y.npy"));
        
        // คำนวณหา Offset แบบเดียวกับด้านบน
        int yHeaderLen = (yBytes[8] & 0xFF) | ((yBytes[9] & 0xFF) << 8);
        int yOffset = 10 + yHeaderLen;
        
        // ไฟล์ y.npy เก็บข้อมูลด้วย Type '|u1' คือ 1-byte Unsigned Integer (เก็บค่า 0-255)
        // และเรียงข้อมูลตามแถว (C-order ปกติ) จึงสามารถลูปอ่านทีละแถวได้เลย
        for (int r = 0; r < m; r++) {
            // ดึงค่าไบต์และใช้ & 0xFF เพื่อให้ได้เป็นค่าจำนวนเต็มบวกใน Java
            y[r][0] = yBytes[yOffset + r] & 0xFF;
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
                ML.tools.ModelWeightsIO.saveWeights("c2w1_assign_layer1_weights.txt", layer1.getWeightsW(), layer1.getWeightsB());
            }
            Dense layer2 = model.getLayer("layer2");
            if (layer2 != null) {
                ML.tools.ModelWeightsIO.saveWeights("c2w1_assign_layer2_weights.txt", layer2.getWeightsW(), layer2.getWeightsB());
            }
            Dense layer3 = model.getLayer("layer3");
            if (layer3 != null) {
                ML.tools.ModelWeightsIO.saveWeights("c2w1_assign_layer3_weights.txt", layer3.getWeightsW(), layer3.getWeightsB());
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("=================================================================\n");
    }
}
