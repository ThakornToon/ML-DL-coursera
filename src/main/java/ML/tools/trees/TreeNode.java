package ML.tools.trees;

import java.io.Serializable;

/**
 * Represents a single node within a decision tree model.
 * A node can act as either a decision node (internal node with a split condition)
 * or a leaf node (terminal node with a final prediction value).
 */
public class TreeNode implements Serializable {
    /** The index of the feature used for splitting (applies to decision nodes). */
    public int featureIndex;
    
    /** The threshold value for the split condition (applies to decision nodes). */
    public double threshold;
    
    /** The left child node, taken when the feature value is <= threshold. */
    public TreeNode left;
    
    /** The right child node, taken when the feature value > threshold. */
    public TreeNode right;
    
    /** The predicted target value or class label (applies to leaf nodes). */
    public double value;
    
    /** Flag indicating whether this node is a terminal leaf node. */
    public boolean isLeaf;

    /**
     * Constructs a leaf node.
     *
     * @param value The prediction value or class label for this leaf.
     */
    public TreeNode(double value) {
        this.value = value;
        this.isLeaf = true;
    }

    /**
     * Constructs a decision (split) node.
     *
     * @param featureIndex The index of the feature to split on.
     * @param threshold    The threshold value for the feature.
     * @param left         The left child node.
     * @param right        The right child node.
     */
    public TreeNode(int featureIndex, double threshold, TreeNode left, TreeNode right) {
        this.featureIndex = featureIndex;
        this.threshold = threshold;
        this.left = left;
        this.right = right;
        this.isLeaf = false;
    }
}
