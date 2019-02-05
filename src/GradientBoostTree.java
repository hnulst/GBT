import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class GradientBoostTree {

    private List<Tree> trees;
    private float[][] trainingSet;
    private float[] labels;

    // const
    private final int maxTrainingRound = 100;   //n_estimators
    private final int maxDepth = 6;             //max_depth
    private final float shrinkageRate = 0.3f;   //eta
    private final float minSplitGain = 0.0f;    //gama
    private final int minSplitNodeNum = 1;      //min_child_weight
    private final float λ = 0.1f;               //lambda

    private static final String path =
            "C:\\Users\\lsj1984\\IdeaProjects\\GBT\\src\\regression.train";
    private static final int sampleNum = 7000;

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        float[][] X = new float[sampleNum][28];
        float[] Y = new float[sampleNum];
        String line = null;
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            String[] strs = line.split("\t");
            float y = Float.parseFloat(strs[0].trim());
            Y[cnt] = y;
            for (int i = 1; i < 29; i++) {
                X[cnt][i - 1] = Float.parseFloat(strs[i].trim());
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
            float[] gradients = this.calculateLogisticGradients(predictions, labels);
            float[] hessians = this.calculateLogisticHessians(predictions, labels);
            float mseLoss = this.calculateMeanLogisticLoss(predictions, labels);

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
        float[] probs = predictBatch(trainingSet);
        float sum = 0f;
        for (int i = 0; i < labels.length; i++) {
            probs[i] = logistic(probs[i]);
            float pred = probs[i] > 0.5f ? 1f : 0f;
            sum += ((pred - labels[i]) * (pred - labels[i]));
        }
        System.out.println(sum);
        long end = System.currentTimeMillis();
        System.out.println("training cost " + (end - start) + " ms");
    }

    private float logistic(float y) {
        return (float) (1d / (1 + Math.exp(-y)));
    }

    private float[] calculateLogisticGradients(float[] predictions, float[] labels) {
        float[] grads = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            grads[i] = logistic(predictions[i]) - labels[i];
        }
        return grads;
    }

    private float[] calculateLogisticHessians(float[] predictions, float[] labels) {
        float[] hessians = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            float p = logistic(predictions[i]);
            hessians[i] = p * (1 - p);
        }
        return hessians;
    }

    private float calculateMeanLogisticLoss(float[] predictions, float[] labels) {
        float sum = 0f;
        for (int i = 0; i < labels.length; i++) {
            float y = labels[i];
            float y_ = predictions[i];
            sum += (y * Math.log((1d + Math.exp(-y_)))
                    + (1 - y) * Math.log(1d + Math.exp(y_)));
        }
        return sum / labels.length;
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
