package ML.tools.trees;

import java.io.*;

public class TreeModelSaver {

    /**
     * Saves a serializable model (like DecisionTree, RandomForest, GradientBoostedTree) to a file.
     *
     * @param model    The model to save.
     * @param filepath The path where the model should be saved (e.g., "my_model.ser").
     */
    public static void saveModel(Object model, String filepath) {
        try (FileOutputStream fileOut = new FileOutputStream(filepath);
             ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(model);
            System.out.println("Model successfully saved to: " + filepath);
        } catch (IOException e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Loads a serializable model from a file.
     *
     * @param filepath The path to the saved model file.
     * @return The loaded model object, which needs to be cast to the correct type.
     */
    public static Object loadModel(String filepath) {
        try (FileInputStream fileIn = new FileInputStream(filepath);
             ObjectInputStream in = new ObjectInputStream(fileIn)) {
            Object model = in.readObject();
            System.out.println("Model successfully loaded from: " + filepath);
            return model;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}
