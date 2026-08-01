package ML.tools;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DataFrame {
    public List<String> columns;
    public List<List<String>> data;

    public DataFrame() {
        this.columns = new ArrayList<>();
        this.data = new ArrayList<>();
    }

    public static DataFrame readCSV(String path) {
        DataFrame df = new DataFrame();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();
            if (line != null) {
                String[] cols = line.split(",");
                df.columns.addAll(Arrays.asList(cols));
            }
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] values = line.split(",");
                df.data.add(new ArrayList<>(Arrays.asList(values)));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return df;
    }

    public void getDummies(List<String> categoricalCols) {
        for (String col : categoricalCols) {
            int colIdx = columns.indexOf(col);
            if (colIdx == -1) continue;

            // Find unique values
            Set<String> uniqueVals = new LinkedHashSet<>();
            for (List<String> row : data) {
                uniqueVals.add(row.get(colIdx));
            }

            // Create new columns and populate them
            List<String> sortedVals = new ArrayList<>(uniqueVals);
            Collections.sort(sortedVals); // For deterministic order
            
            for (String val : sortedVals) {
                columns.add(col + "_" + val);
            }

            for (List<String> row : data) {
                String rowVal = row.get(colIdx);
                for (String val : sortedVals) {
                    row.add(rowVal.equals(val) ? "1" : "0");
                }
            }
        }

        // Remove old categorical columns
        List<Integer> toRemove = new ArrayList<>();
        for (String col : categoricalCols) {
            int idx = columns.indexOf(col);
            if (idx != -1) toRemove.add(idx);
        }
        toRemove.sort(Collections.reverseOrder());
        for (int idx : toRemove) {
            columns.remove(idx);
            for (List<String> row : data) {
                row.remove(idx);
            }
        }
    }

    public List<String> getFeatures(String excludeTarget) {
        List<String> features = new ArrayList<>(columns);
        features.remove(excludeTarget);
        return features;
    }

    public double[][] toNumericMatrix(List<String> featureCols) {
        double[][] matrix = new double[data.size()][featureCols.size()];
        List<Integer> indices = new ArrayList<>();
        for (String col : featureCols) {
            indices.add(columns.indexOf(col));
        }
        
        for (int r = 0; r < data.size(); r++) {
            for (int c = 0; c < featureCols.size(); c++) {
                matrix[r][c] = Double.parseDouble(data.get(r).get(indices.get(c)));
            }
        }
        return matrix;
    }

    public double[] toNumericArray(String targetCol) {
        int idx = columns.indexOf(targetCol);
        double[] arr = new double[data.size()];
        for (int r = 0; r < data.size(); r++) {
            arr[r] = Double.parseDouble(data.get(r).get(idx));
        }
        return arr;
    }

    public static class Split {
        public double[][] XTrain, XVal;
        public double[] yTrain, yVal;
    }

    public static Split trainTestSplit(double[][] X, double[] y, double trainSize, int randomState) {
        int n = X.length;
        int nTrain = (int) (n * trainSize);
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n; i++) indices.add(i);
        
        Random rand = new Random(randomState);
        Collections.shuffle(indices, rand);

        Split split = new Split();
        split.XTrain = new double[nTrain][X[0].length];
        split.yTrain = new double[nTrain];
        split.XVal = new double[n - nTrain][X[0].length];
        split.yVal = new double[n - nTrain];

        for (int i = 0; i < n; i++) {
            int idx = indices.get(i);
            if (i < nTrain) {
                split.XTrain[i] = X[idx];
                split.yTrain[i] = y[idx];
            } else {
                split.XVal[i - nTrain] = X[idx];
                split.yVal[i - nTrain] = y[idx];
            }
        }
        return split;
    }
}
