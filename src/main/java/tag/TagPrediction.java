package tag;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Attribute;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.mllib.*;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;
import scala.reflect.ClassTag;
import scala.reflect.api.TypeTags.TypeTag;
import scala.xml.*;
import org.jsoup.*;

public class TestSpark {

	private static Dataset<Row> XMLParser(SparkSession spark, JavaRDD<String> postRDD){
		// filter the valid xml rows
		postRDD = postRDD.filter(new Function<String, Boolean>() {
			public Boolean call(String line) throws Exception {
				return line.contains("row Id");
			}
		});
		
		JavaPairRDD<String, String> postRDDTuple= postRDD.mapToPair(new PairFunction<String, String, String>() {
		
			public Tuple2<String, String> call(String line) throws Exception {
				Node x = XML.loadString(line);
				String body=x.attribute("Body").get().toString();
				String tags=x.attribute("Tags").isDefined()?x.attribute("Tags").get().toString():"";
				String title=x.attribute("Title").isDefined()?x.attribute("Title").get().toString():"";
				
				org.jsoup.nodes.Document doc = Jsoup.parse(body);
				body = doc.body().text().toLowerCase().replaceAll("\\<.*?\\>", "");
				
				doc = Jsoup.parse(tags);
				String[] tagsArray = tags.equals("")?null:(doc.body().text().replaceAll("<", "")
						.replaceAll(">", ",").toLowerCase().split(","));
				
				doc = Jsoup.parse(title);
				title = doc.body().text().replaceAll("\\<.*?\\>", "");
				
				return new Tuple2<String, String>(body, tagsArray==null?null:String.join(",", tagsArray));
			}
			
		}).filter(new Function<Tuple2<String,String>, Boolean>() {
			// filter the rows with bothbody and tags
			public Boolean call(Tuple2<String, String> bodyTag) throws Exception {
				return !(bodyTag._1==null||bodyTag._2==null);
			}
		});
		
		
		JavaRDD<Row> rowRDD = postRDDTuple.map(new Function<Tuple2<String,String>, Row>() {
			public Row call(Tuple2<String, String> line) throws Exception {
				return RowFactory.create(line._1, line._2);
			}
		});
		
		StructType schema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("body", DataTypes.StringType, false),
						DataTypes.createStructField("tags", DataTypes.StringType, false)
					 });
		
		return spark.createDataFrame(rowRDD, schema);
	}
	
	public static void main(String[] args) {
		SparkConf sc = new SparkConf()
				.setAppName("TestApp")
				.setMaster("local")
				// Spark 2.0.0 is to set spark.sql.warehouse.dir to some properly-referenced directory
				.set("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse");
		
		
		JavaSparkContext jsc = new JavaSparkContext(sc);
		
		JavaRDD<String> textLoadRdd = jsc.textFile("c:/Users/pratik/workspace_new/tag/src/main/resources/beer.stackexchange.com");
		
		SparkSession spark = SparkSession.builder().config(sc).getOrCreate();
		
		Dataset<Row> brewing = XMLParser(spark, textLoadRdd);
		
		JavaPairRDD<String,Double> tags	=	
			brewing.withColumn("tag", org.apache.spark.sql.functions.explode(org.apache.spark.sql.functions.split(brewing.col("tags"), ",")))
				.select("tag")
				.distinct()
				.javaRDD().map(new Function<Row, String>() {
					public String call(Row row) throws Exception {
						// TODO Auto-generated method stub
						return row.getString(0);
					}
				})
				.zipWithIndex()
				.mapValues(new Function<Long, Double>() {
					public Double call(Long l) throws Exception {
						return l.doubleValue()+1;
					}
				})
		;
		
		// broadcast it to all the nodes
		final Map<String, Double> labelTagsMap = tags.collectAsMap();
		
		final Broadcast<Map<String, Double>> bLabelTags = jsc.broadcast(labelTagsMap);
		// print the distinct tags with label
		Dataset<Row> tagsDS = spark.createDataset(tags.collect(), Encoders.tuple(Encoders.STRING(),Encoders.DOUBLE())).toDF("tag", "label");
		tagsDS.createOrReplaceTempView("TAGS");
		Dataset<Row> sqlDF = spark.sql("SELECT * FROM TAGS");
	    sqlDF.show();
	    
		JavaPairRDD<String, String> brewingRDD = brewing.javaRDD()
		.mapToPair(new PairFunction<Row, String, String>() {
			public Tuple2<String, String> call(Row row) throws Exception {
				return new Tuple2(row.get(0),row.get(1));
			}
		});
		
		
		JavaPairRDD<String, String> CleanBrewingRDD = stem(brewingRDD);
		
		final HashingTF hashingTf = new HashingTF();
		final Normalizer normalizer = new Normalizer();
		
		System.out.println(hashingTf.numFeatures());
		
		JavaRDD<LabeledPoint> labeledPoints = CleanBrewingRDD.mapValues(new Function<String, String[]>() {
			public String[] call(String tags) throws Exception {
				// TODO Auto-generated method stub
				return tags.split(",");
			}
		}).flatMap(new FlatMapFunction<Tuple2<String,String[]>, LabeledPoint>() {
			public Iterator<LabeledPoint> call(Tuple2<String, String[]> row) throws Exception {
				Vector corpus = hashingTf.transform(Arrays.asList(row._1.split(" ")));
				Vector normVector = normalizer.transform(corpus);
				ArrayList<LabeledPoint> list = new ArrayList<LabeledPoint>();
				for(String tag: row._2){
					list.add(new LabeledPoint(bLabelTags.value().getOrDefault(tag, 0.0d), normVector));
				}
				return list.iterator();
			}
		
		});
		
		
		for (LabeledPoint i : labeledPoints.take(5)){
			System.out.println(i.label() + ":" + i.features());
		}
		
		JavaRDD<LabeledPoint>[] tmp = labeledPoints.randomSplit(new double[]{0.6, 0.4});
		JavaRDD<LabeledPoint> training = tmp[0]; // training set
		JavaRDD<LabeledPoint> test = tmp[1]; // test set
		
		RDD<LabeledPoint> labelPointsRdd = training.rdd();
		final NaiveBayesModel model = NaiveBayes.train(labelPointsRdd);
		labelPointsRdd.unpersist(false);
		
		// test
		JavaPairRDD<Double, Double> predictionAndLabel =
				test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					public Tuple2<Double, Double> call(LabeledPoint l) throws Exception {
						return new Tuple2(model.predict(l.features()), l.label());
					}
				});
		
		double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double,Double>, Boolean>() {
			public Boolean call(Tuple2<Double, Double> row) throws Exception {
				return row._1().equals(row._2());
			}
		}).count()/(double)test.count();
		
		System.out.println(accuracy);
				
		
		Dataset<Row> pL = spark.createDataset(predictionAndLabel.collect(), Encoders.tuple(Encoders.DOUBLE(),Encoders.DOUBLE())).toDF("Prediction", "label");
		pL.createOrReplaceTempView("predictionAndLabel");
		Dataset<Row> plDF = spark.sql("SELECT * FROM predictionAndLabel");
	    plDF.show();
	    
	    // sample results
	    spark.sql("Select p.*, tp.tag as predictedTag, tl.tag as actualtag from predictionAndLabel p "
	    		+ "join TAGS tp ON p.Prediction = tp.label "
	    		+ "join TAGS tl ON p.label = tl.label").show();
	}
	
	private static JavaPairRDD<String, String> stem(JavaPairRDD<String, String> rdd){
		JavaPairRDD<String, String> cleanRDD = 
		rdd
		.mapToPair(new PairFunction<Tuple2<String,String>, String, String>() {
			
			public Tuple2<String, String> call(Tuple2<String, String> current) throws Exception {
					EnglishAnalyzer ea = new EnglishAnalyzer();
		        
					// remove punctuation
			        String body = current._1.replaceAll("\\W", " ");
			        StringReader sr = new StringReader(body);
			        TokenStream ts = ea.tokenStream("contents", sr);
					CharTermAttribute tr = ts.addAttribute(CharTermAttribute.class);
					StringBuffer terms = new StringBuffer();
					ts.reset();
					while(ts.incrementToken()){
						String cleanToken = tr.toString();
						if(cleanToken.length()>3){
							terms.append(cleanToken);
							terms.append(" ");
						}
					}
					
				ea.close();
				
				// remove the last comma
				terms = terms.deleteCharAt(terms.length()-1);
				
				return new Tuple2<String, String>(terms.toString(), current._2);
			}
		});
		return cleanRDD;
	}
}
