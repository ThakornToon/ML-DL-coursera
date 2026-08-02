package ML;

import ML.tools.KMeans;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public final class C3W1KMeansAssignment {

    private C3W1KMeansAssignment() {}

    public static void start() {
        System.out.println("Running K-Means on image compression...");
        try {
            // 1. Load the original image
            File file = new File("src/main/resources/ML/C3_W1_KMeans_Assignment_Data/bird_small.png");
            if (!file.exists()) {
                System.out.println("Could not find image: " + file.getAbsolutePath());
                return;
            }
            BufferedImage img = ImageIO.read(file);
            int width = img.getWidth();
            int height = img.getHeight();
            System.out.println("Image loaded: " + width + "x" + height);

            // 2. Convert image to X matrix (m x 3)
            int m = width * height;
            double[][] X_img = new double[m][3];
            int index = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = img.getRGB(x, y);
                    X_img[index][0] = (rgb >> 16) & 0xFF; // Red
                    X_img[index][1] = (rgb >> 8) & 0xFF;  // Green
                    X_img[index][2] = rgb & 0xFF;         // Blue
                    index++;
                }
            }

            // 3. Run K-Means
            int K = 16;
            int maxIters = 10;
            System.out.println("Running K-Means with K=" + K + ", maxIters=" + maxIters + "...");
            KMeans kmeans = new KMeans(K, maxIters);
            kmeans.fit(X_img);
            
            // 4. Find closest centroid for each pixel
            int[] idx = kmeans.predict(X_img);
            double[][] centroids = kmeans.getCentroids();

            // 5. Reconstruct the compressed image
            System.out.println("Reconstructing compressed image...");
            BufferedImage compressedImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            index = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int cluster = idx[index];
                    int r = (int) Math.round(centroids[cluster][0]);
                    int g = (int) Math.round(centroids[cluster][1]);
                    int b = (int) Math.round(centroids[cluster][2]);
                    
                    // Clamp values to 0-255 just in case
                    r = Math.min(255, Math.max(0, r));
                    g = Math.min(255, Math.max(0, g));
                    b = Math.min(255, Math.max(0, b));
                    
                    int rgb = (r << 16) | (g << 8) | b;
                    compressedImg.setRGB(x, y, rgb);
                    index++;
                }
            }

            // 6. Save the output
            File outputFile = new File("bird_small_compressed.png");
            ImageIO.write(compressedImg, "png", outputFile);
            System.out.println("Successfully compressed image and saved to: " + outputFile.getAbsolutePath());

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("=================================================================\n");
    }
}

