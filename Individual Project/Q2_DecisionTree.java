import org.apache.spark.SparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.mllib.evaluation.*;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.ml.classification.*;


import org.apache.spark.ml.PipelineModel;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;

import scala.Tuple2;

public class Q2_DecisionTree {



public static void main(String[] args) {

	        //Generate a random seed
	    	final long SEED = new Random().nextLong();

	    	String results = "Seed used: " + SEED + "\n";


	        // start time
	        final long timer_start = System.nanoTime();

	        // configure spark session & context
	        SparkSession spark = SparkSession.builder().appName("DecisionTree").getOrCreate();
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
	            if (tokens[features.length].equals("Minor")) {
	                return new LabeledPoint(0.0, v);
	            } else if(tokens[features.length].equals("Moderate")){
	                return new LabeledPoint(1.0, v);
	            }else if(tokens[features.length].equals("Major")){
	                return new LabeledPoint(2.0, v);
	            }else{
	                return new LabeledPoint(3.0, v);
	            }
	        });

	        Dataset<Row> data = spark.createDataFrame(linesRDD, LabeledPoint.class);


	        
	        //From Week 12 - Slide 36 - index labels
	        StringIndexerModel labelIndexer = new StringIndexer()
	                .setInputCol("label")
	                .setOutputCol("indexedLabel")
	                .fit(data);

	        // identify categorical features and index
	        VectorIndexerModel featureIndexer = new VectorIndexer()
	                .setInputCol("features")
	                .setOutputCol("indexedFeatures")
	                .setMaxCategories(20)
	                .fit(data);

		//Scale the data
		StandardScalerModel scaler = new StandardScaler()
			.setInputCol("features")
	               .setOutputCol("ScaledFeatures")
	               .fit(data);

		//Dataset<Row> scaledData = scaler.transform(data);//.select("ScaledFeatures");
	       
		PCAModel pca = new PCA()
  			.setInputCol("features")
  			.setOutputCol("pcaFeatures")
  			.setK(1)
  			.fit(data);

		//Dataset<Row> scaledData = scaler.transform(data);//.select("ScaledFeatures");
		//Dataset<Row> pcaData = scaler.transform(scaledData);//.select("ScaledFeatures");

	        // split data
	        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3}, SEED);
	        Dataset<Row> trainingData = splits[0];
	        Dataset<Row> testData = splits[1];

	        // train decision tree model
	        RandomForestClassifier dt = new RandomForestClassifier()
	                .setLabelCol("indexedLabel")
	                .setFeaturesCol("pcaFeatures");
			//.setNumTrees(20);

	        // convert indexed labels back to original labels
	        IndexToString labelConverter = new IndexToString()
	                .setInputCol("prediction")
	                .setOutputCol("predictedLabel")
	                .setLabels(labelIndexer.labels());

	        // chain indexes and tree in a pipeline
	        Pipeline pipeline = new Pipeline()
	                .setStages(new PipelineStage[]{labelIndexer, featureIndexer,scaler, dt, labelConverter});

	        // train decision tree model
	        PipelineModel model = pipeline.fit(trainingData);

	        // make predictions
	        Dataset<Row> predictions = model.transform(testData);
	       // predictions.select("predictedLabel", "label", "ScaledFeatures").show(5);

	        // select (prediction, true label) and compute test error
	        MulticlassClassificationEvaluator acc_evaluator = new MulticlassClassificationEvaluator()
	                .setLabelCol("indexedLabel")
	                .setPredictionCol("prediction")
	                .setMetricName("accuracy");
		
		MulticlassClassificationEvaluator rec_evaluator = new MulticlassClassificationEvaluator()
	                .setLabelCol("indexedLabel")
	                .setPredictionCol("prediction")
			.setMetricName("weightedRecall");

	        //rec_evaluator.metricName("weightedRecall");   
 
		MulticlassClassificationEvaluator pre_evaluator = new MulticlassClassificationEvaluator()
	                .setLabelCol("indexedLabel")
	               .setPredictionCol("prediction")
	                .setMetricName("weightedPrecision");

		
	        // print test accuracy
	        double test_accuracy = acc_evaluator.evaluate(predictions);
	        results += "Test Accuracy = " + test_accuracy + "\n";
	        results += "Test Error = " + (1.0 - test_accuracy) + "\n";

	        // print test accuracy
	        double test_Recall = rec_evaluator.evaluate(predictions);
	        results += "Test Recall = " + test_Recall + "\n";


	        // print test accuracy
	        double test_Precision = pre_evaluator.evaluate(predictions);
	        results += "Test Precision = " + test_Precision + "\n";

		
	        // print train accuracy
	        Dataset<Row> train_instances = model.transform(trainingData);
	        double training_accuracy = acc_evaluator.evaluate(train_instances);
	        results += "Train Accuracy = " + training_accuracy + "\n";
	        results += "Train Error = " + (1.0 - training_accuracy) + "\n";

	        // print train accuracy
	        double train_Recall = rec_evaluator.evaluate(train_instances);
	        results += "Recall = " + train_Recall + "\n";


	        // print train accuracy
	        double train_Precision = pre_evaluator.evaluate(train_instances);
	        results += "Train Precision = " + train_Precision + "\n";


	        // runtime of program in ms
	        final long CompletionTime = (System.nanoTime() - timer_start)/1000000;
	        results += "Time taken(ms): " + CompletionTime + "\n";

		RandomForestClassificationModel treeModel = (RandomForestClassificationModel) (model.stages()[2]);
		

		
	        List<String> resultsString = new ArrayList<String>();
	        resultsString.add(results);
	 		
	 		JavaRDD<String> output = context.parallelize(resultsString);
	 		output.saveAsTextFile(args[1]);



	    }
	
}
