package ML.tools.trees;

import java.io.Serializable;

public class TreeNode implements Serializable {
    public int featureIndex;
    public double threshold;
    public TreeNode left;
    public TreeNode right;
    public double value; // Leaf prediction
    public boolean isLeaf;

    // Leaf node constructor
    public TreeNode(double value) {
        this.value = value;
        this.isLeaf = true;
    }

    // Split node constructor
    public TreeNode(int featureIndex, double threshold, TreeNode left, TreeNode right) {
        this.featureIndex = featureIndex;
        this.threshold = threshold;
        this.left = left;
        this.right = right;
        this.isLeaf = false;
    }
}
