package ML.tools;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Utility class for reading basic .npy files.
 */
public final class NumpyIO {

    private NumpyIO() {}

    /**
     * Loads a 2D float64 matrix from a .npy file.
     *
     * @param filePath The path to the .npy file.
     * @param m The number of rows.
     * @param n The number of columns.
     * @param columnMajor True if the file was saved with fortran_order = True.
     * @return A 2D double array [m][n].
     * @throws IOException If the file cannot be read.
     */
    public static double[][] loadDoubleMatrix(String filePath, int m, int n, boolean columnMajor) throws IOException {
        byte[] bytes = Files.readAllBytes(Paths.get(filePath));
        int headerLen = (bytes[8] & 0xFF) | ((bytes[9] & 0xFF) << 8);
        int offset = 10 + headerLen;

        ByteBuffer buffer = ByteBuffer.wrap(bytes, offset, bytes.length - offset);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        double[][] X = new double[m][n];
        if (columnMajor) {
            for (int c = 0; c < n; c++) {
                for (int r = 0; r < m; r++) {
                    X[r][c] = buffer.getDouble();
                }
            }
        } else {
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    X[r][c] = buffer.getDouble();
                }
            }
        }
        return X;
    }

    /**
     * Loads a 1D uint8 array from a .npy file.
     *
     * @param filePath The path to the .npy file.
     * @param m The length of the array.
     * @return A 1D int array.
     * @throws IOException If the file cannot be read.
     */
    public static int[] loadUint8Array(String filePath, int m) throws IOException {
        byte[] bytes = Files.readAllBytes(Paths.get(filePath));
        int headerLen = (bytes[8] & 0xFF) | ((bytes[9] & 0xFF) << 8);
        int offset = 10 + headerLen;

        int[] y = new int[m];
        for (int r = 0; r < m; r++) {
            y[r] = bytes[offset + r] & 0xFF;
        }
        return y;
    }
}
