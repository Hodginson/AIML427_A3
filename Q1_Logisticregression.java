package AIML427_Q1;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.api.java.JavaRDD;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;

public class Q1_Logisticregression {
    public static void main(String[] args) {
    	
    	//Generate a random seed
    	final long SEED = new Random().nextLong();
    	String results = "Seed used: " + SEED + "\n";

        // start time
        final long timer_start = System.nanoTime();

        SparkSession spark = SparkSession.builder().appName("logisticregression").getOrCreate();
        SparkContext sc = spark.sparkContext();
	JavaSparkContext  context = new JavaSparkContext(sc);


        // From week 12 slide 17
        JavaRDD<String> lines = sc.textFile(args[0],0).toJavaRDD();
        
        JavaRDD<LabeledPoint> linesRDD = lines.map(line -> {
            String[] tokens = line.split(",");
            double[] features = new double[tokens.length - 1];
            for (int i = 0; i < features.length; i++) {
                features[i] = Double.parseDouble(tokens[i]);}
            Vector v = new DenseVector(features);
            if (tokens[features.length].equals("normal")) {
                return new LabeledPoint(0.0, v);
            } else {
                return new LabeledPoint(1.0, v);
            }
        });

        Dataset<Row> data = spark.createDataFrame(linesRDD, LabeledPoint.class);


      
        //From week 12 slide 23 - create train and test splits
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3},SEED);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];


        
        // From week 12 slide 23 - define logistic regression instance
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(25)             //set max iterations
                .setRegParam(0.3)           //set lambda
                .setElasticNetParam(0.8);   //set alpha

        //fit model
        LogisticRegressionModel lrModel = lr.fit(training);
        results += "Coefficients: " + lrModel.coefficients() + "\n";
        results += "Intercept: " + lrModel.intercept() + "\n";

      
        //From week 12 slide 24 - extract summary from the model produced
        BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();
        // retrieve the loss per iteration
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        for (double lossPerIteration: objectiveHistory) {
            System.out.println(lossPerIteration);
        }
        // obtain the ROC as dataframe and areaUnderROC
        Dataset<Row> roc = trainingSummary.roc();
        roc.show();
        roc.select("FPR").show();
        results += "Area Under ROC: " + trainingSummary.areaUnderROC() + "\n";
        
        //Get the training accuracy 
        double training_accuracy = trainingSummary.accuracy();

        //output the training accuracy
        results += "Train Accuracy = " + training_accuracy + "\n";
        results += "Train Error = " + (1.0 - training_accuracy) + "\n";
        

        // get threshold of the max f-measure
        Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();
        double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
        double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold")
                .head()
                .getDouble(0);
        // set it as threshold for model
        lrModel.setThreshold(bestThreshold);


        //From week 12 slide 26 - make predictions
        Dataset<Row> predictions = lrModel.transform(test);

        // select example rows to display
        predictions.show(5);

        // select (prediction, true label) to compute test error
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // print accuracy
        double test_accuracy = evaluator.evaluate(predictions);
        results += "Test Accuracy = " + test_accuracy + "\n";
        results += "Test Error = " + (1.0 - test_accuracy) + "\n";
        
        final long CompletionTime = (System.nanoTime() - timer_start)/1000000;
        results += "Time taken(ms): " + CompletionTime + "\n";
        
        List<String> resultsString = new ArrayList<String>();
        resultsString.add(results);
 		
 		JavaRDD<String> output = context.parallelize(resultsString);
 		output.saveAsTextFile(args[1]);

    }
}