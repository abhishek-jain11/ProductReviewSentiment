/**
  * Created by jaina2 on 6/9/17.
  */

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer, Tokenizer}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._

import scala.tools.nsc.interactive.Lexer.Token


object Driver extends App {
  val spark = SparkSession
    .builder()
    .appName("Product Review Sentiment")
    .master("local[2]")
    .getOrCreate()

  val df = spark.read.option("header", "true").csv(args(0)).filter(col("review").isNotNull)

  val tokenizer = new RegexTokenizer().setInputCol("review").setOutputCol("parsed_review").setPattern("\\W")


  val tokenized = tokenizer.transform(df).select("name","review","parsed_review","rating","index").filter(df.col("rating") =!= (3)).
    withColumn("label", (when(df("rating").geq(3) ,0).otherwise(1)) )//write.format("json").mode(SaveMode.Overwrite).save("/Users/jaina2/Desktop/TestResult")

  val train_Data_idx = spark.read.csv("/Users/jaina2/Downloads/module-2-assignment-train-idx.json").withColumnRenamed("_c0","index")
  val test_Data_idx = spark.read.csv("/Users/jaina2/Downloads/module-2-assignment-test-idx.json").withColumnRenamed("_c0","index")

  val train_Data = tokenized.join(train_Data_idx, "index" )
  val test_Data = tokenized.join(test_Data_idx,"index")


  val cvTrainModel: CountVectorizerModel = new CountVectorizer()
    .setInputCol("parsed_review")
    .setOutputCol("features")
    .fit(train_Data)

  /*val cvTestModel:CountVectorizerModel = new CountVectorizer()
    .setInputCol("parsed_review")
    .setOutputCol("vectors")
    .fit(test_Data)*/

  val train_matrix = cvTrainModel.transform(train_Data).select("features","label","index")
  val test_matrix = cvTrainModel.transform(test_Data).select("features","label","index")

  val lr = new LogisticRegression().setMaxIter(20).setRegParam(0.3)
  val mlr = lr.fit(train_matrix)

  val matrix = mlr.coefficientMatrix

  val rows = matrix.numRows-1
  val cols = matrix.numCols-1

  var x=0
  var y=0
  var count =0
  for( x <- 0 to rows;  y <- 0 to cols){
    if (matrix.apply(x,y)>0) {
      count = count +1
    }

  }

  println(count)
  val lst:List[Int] = (10 to 12).toList
  test_matrix.take(20).foreach(row=>println(row))
val sample_test_Data = test_matrix.filter(col("index") .isin(lst:_*))
sample_test_Data.show()
}
