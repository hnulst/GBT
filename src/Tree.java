public class Tree {

    private TreeNode root;

    public Tree(float[][] matrix, float[] gradients, float[] hessians, int depth,
                int maxDepth, float minSplitGain, int minSplitNodeNum, float shrinkageRate, float λ) {
        this.root = new TreeNode(matrix, gradients, hessians, depth,
                maxDepth, minSplitGain, minSplitNodeNum, shrinkageRate, λ);
    }

    public void build() {
        this.root.build();
    }

    public float predict(float[] x) {
        return this.root.predict(x);
    }
}
