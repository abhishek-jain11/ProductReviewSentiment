/**
  * Created by jaina2 on 6/9/17.
  */
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.SparkSession

import scala.tools.nsc.interactive.Lexer.Token


object Driver extends App {
  val spark = SparkSession
    .builder()
    .appName("Product Review Sentiment")
    .getOrCreate()

  val df = spark.read.csv(args(0))
  val tokenizer = new Tokenizer().setInputCol("review").setOutputCol("parsed_review")

  val tokenized = tokenizer.transform(df).select("review","parsed_review").show()


}
