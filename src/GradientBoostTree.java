import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class GradientBoostTree {

    private List<Tree> trees;
    private float[][] trainingSet;
    private float[] labels;

    // const
    private final int maxTrainingRound = 200;
    private final int maxDepth = 5;
    private final float shrinkageRate = 0.3f;
    private final float minSplitGain = 0.1f;
    private final int minSplitNodeNum = 3;
    private final float λ = 0.1f;

    private static final String path =
            "C:\\Users\\Jenny\\IdeaProjects\\GBT\\src\\TempLinkoping2016.txt";
    private static final int sampleNum = 366;

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        float[][] X = new float[sampleNum][1];
        float[] Y = new float[sampleNum];
        String line = null;
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            if (cnt != 0) {
                String[] strs = line.split("\t");
                float x = Float.parseFloat(strs[0].trim());
                float y = Float.parseFloat(strs[1].trim());
                X[cnt - 1][0] = x;
                Y[cnt - 1] = y;
            }
            cnt++;
        }
        GradientBoostTree gbt = new GradientBoostTree(X, Y);
        gbt.train();
    }

    public GradientBoostTree(float[][] trainingSet, float[] labels) {
        this.trainingSet = trainingSet;
        this.labels = labels;
        this.trees = new ArrayList<>();
    }

    public void train() {
        long start = System.currentTimeMillis();
        for (int round = 0; round < maxTrainingRound; round++) {
            float shrink = 1f;
            float[] predictions = this.predictBatch(trainingSet);
            float[] gradients = this.calculateMseGradients(predictions, labels);
            float[] hessians = this.calculateMseHessians(predictions, labels);
            float mseLoss = this.calculateMseLoss(predictions, labels);

            System.out.println("mse loss at round " + round + ": " + mseLoss);
            if (round != 0) {
                shrink *= this.shrinkageRate;
            }
            Tree tree = new Tree(trainingSet, gradients, hessians, 0,
                    this.maxDepth, this.minSplitGain, this.minSplitNodeNum, shrink, this.λ
            );
            tree.build();
            trees.add(tree);
        }
        long end = System.currentTimeMillis();
        System.out.println("training cost " + String.valueOf(end - start) + " ms");
    }

    private float[] calculateMseGradients(float[] predictions, float[] labels) {
        float[] grads = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            grads[i] = 2 * (predictions[i] - labels[i]);
        }
        return grads;
    }

    private float[] calculateMseHessians(float[] predictions, float[] labels) {
        float[] hessians = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            hessians[i] = 2;
        }
        return hessians;
    }

    private float calculateMseLoss(float[] predictions, float[] labels) {
        float sum = 0f;
        for (int i = 0; i < labels.length; i++) {
            sum += ((predictions[i] - labels[i]) * (predictions[i] - labels[i]));
        }
        return sum / labels.length;
    }

    private float[] predictBatch(float[][] matrix) {
        float[] preds = new float[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            preds[i] = this.predict(matrix[i]);
        }
        return preds;
    }

    public float predict(float[] x) {
        if (trees == null || trees.size() < 1) {
            return 0f;
        }
        float pred = 0f;
        for (int i = 0; i < trees.size(); i++) {
            pred += trees.get(i).predict(x);
        }
        return pred;
    }
}
