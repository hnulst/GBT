import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TreeNode {

    private boolean isLeaf;
    private float weight;

    private TreeNode leftChild;
    private TreeNode rightChild;

    private int splitFeatureId;
    private float splitFeatureValue;

    private float[][] matrix;
    private float[] gradients;
    private float[] hessians;
    private int depth;

    // const
    private int maxDepth;
    private float minSplitGain;
    private int minSplitNodeNum;
    private float shrinkageRate;
    private float λ;

    public TreeNode(float[][] matrix, float[] gradients, float[] hessians, int depth,
                    int maxDepth, float minSplitGain, int minSplitNodeNum, float shrinkageRate, float λ) {
        this.matrix = matrix;
        this.gradients = gradients;
        this.hessians = hessians;
        this.depth = depth;
        this.maxDepth = maxDepth;
        this.minSplitGain = minSplitGain;
        this.minSplitNodeNum = minSplitNodeNum;
        this.shrinkageRate = shrinkageRate;
        this.λ = λ;
    }


    private float[][] filterByIndex2D(float[][] matrix, int[] indexes) {
        float[][] m = new float[indexes.length][matrix[0].length];
        for (int i = 0; i < indexes.length; i++) {
            m[i] = matrix[indexes[i]];
        }
        return m;
    }

    private float[] filterByIndex1D(float[] derivates, int[] indexes) {
        float[] d = new float[indexes.length];
        for (int i = 0; i < indexes.length; i++) {
            d[i] = derivates[indexes[i]];
        }
        return d;
    }

    private int[] splitLeft(int[] sortedIndexes, int splitIndex) {
        int[] left = new int[splitIndex];
        for (int i = 0; i < splitIndex; i++) {
            left[i] = sortedIndexes[i];
        }
        return left;
    }

    private int[] splitRight(int[] sortedIndexes, int splitIndex) {
        int[] right = new int[sortedIndexes.length - splitIndex];
        for (int i = splitIndex; i < sortedIndexes.length; i++) {
            right[i - splitIndex] = sortedIndexes[i];
        }
        return right;
    }

    private float sumOf(float[] derivates) {
        float sum = 0f;
        for (float d : derivates) {
            sum += d;
        }
        return sum;
    }

    private float calculateGHλ(float g, float h) {
        return -(g * g) / 2 * (h + λ);
    }

    private float calculateSplitGain(float G, float H, float GL, float HL, float GR, float HR) {
        return this.calculateGHλ(G, H) - this.calculateGHλ(GL, HL) - this.calculateGHλ(GR, HR);
    }

    private float calculateLeafWeight() {
        return -sumOf(this.gradients) / (sumOf(this.hessians) + λ);
    }

    private int[] sortedIndexesByFeatureAt(float[][] matrix, int featureId) {
        List<FeatureAndIndex> sorted = new ArrayList<>(matrix.length);
        for (int i = 0; i < matrix.length; i++) {
            sorted.add(new FeatureAndIndex(matrix[i][featureId], i));
        }
        Collections.sort(sorted);
        int[] sortedIndexes = new int[matrix.length];
        for (int i = 0; i < sortedIndexes.length; i++) {
            sortedIndexes[i] = sorted.get(i).index;
        }
        return sortedIndexes;
    }

    private class FeatureAndIndex implements Comparable<FeatureAndIndex> {
        float value;
        int index;

        public FeatureAndIndex(float value, int index) {
            this.value = value;
            this.index = index;
        }

        @Override
        public int compareTo(FeatureAndIndex f) {
            return Float.compare(this.value, f.value);
        }
    }

    public void build() {
        if (this.depth > this.maxDepth
                || this.matrix.length < this.minSplitNodeNum) {
            this.isLeaf = true;
            this.weight = calculateLeafWeight();
            return;
        }
        // find best split
        float bestGain = Float.MIN_VALUE;
        int bestSplitIndex = 0;
        float bestSplitValue = 0f;
        int[] leftIndexes = new int[0];
        int[] rightIndexes = new int[0];

        final float G = sumOf(this.gradients);
        final float H = sumOf(this.hessians);
        float GL = 0f;
        float HL = 0f;
        float GR;
        float HR;

        for (int featureIndex = 0; featureIndex < this.matrix[0].length; featureIndex++) {
            int[] sortedIndexes = sortedIndexesByFeatureAt(this.matrix, featureIndex);
            for (int sampleIndex = 0; sampleIndex < this.matrix.length; sampleIndex++) {
                GL += this.gradients[sortedIndexes[sampleIndex]];
                HL += this.hessians[sortedIndexes[sampleIndex]];
                GR = G - GL;
                HR = H - HL;
                float splitGain = this.calculateSplitGain(G, H, GL, HL, GR, HR);
                if (splitGain > bestGain) {
                    leftIndexes = splitLeft(sortedIndexes, sampleIndex);
                    rightIndexes = splitRight(sortedIndexes, sampleIndex);
                    bestGain = splitGain;
                    bestSplitIndex = featureIndex;
                    bestSplitValue = matrix[sampleIndex][featureIndex];
                }
            }
        }

        if (bestGain < this.minSplitGain) {
            this.isLeaf = true;
            this.weight = calculateLeafWeight();
        } else {
            this.splitFeatureValue = bestSplitValue;
            this.splitFeatureId = bestSplitIndex;

            TreeNode left = new TreeNode(filterByIndex2D(this.matrix, leftIndexes),
                    filterByIndex1D(this.gradients, leftIndexes),
                    filterByIndex1D(this.hessians, leftIndexes),
                    depth + 1,
                    this.maxDepth,
                    this.minSplitGain,
                    this.minSplitNodeNum,
                    this.shrinkageRate,
                    this.λ);
            left.build();
            this.leftChild = left;

            TreeNode right = new TreeNode(filterByIndex2D(this.matrix, rightIndexes),
                    filterByIndex1D(this.gradients, rightIndexes),
                    filterByIndex1D(this.hessians, rightIndexes),
                    depth + 1,
                    this.maxDepth,
                    this.minSplitGain,
                    this.minSplitNodeNum,
                    this.shrinkageRate,
                    this.λ);
            right.build();
            this.rightChild = right;
        }
    }

    public float predict(float[] x) {
        if (this.isLeaf) {
            return this.weight;
        }
        if (x[this.splitFeatureId] <= this.splitFeatureValue) {
            return this.leftChild.predict(x);
        } else {
            return this.rightChild.predict(x);
        }
    }

}
