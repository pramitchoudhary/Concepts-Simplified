/**
  * Created by pramitchoudhary on 9/12/16.
  */

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline

val df = sqlContext.createDataFrame(Seq(
  (0, "a", "Desktop", 2, 0.3),
  (1, "b", "Mobile", 2, 0.2),
  (2, "c", "", 3, 0.5),
  (3, "", "appleTV", 3, 0.4),
  (4, "a", "Desktop", 2, 0.2),
  (5, "c", "Mobile", 2, 0.3)
)).toDF("user_id", "category", "device", "channel_id", "value")
df.show()

val temp = df.na.replace(Seq("category", "device"), Map("" -> "NA"))
temp.show()

// To handle categorical parameters
val categorical_features = Array("category", "device")
val indexer: Array[org.apache.spark.ml.PipelineStage] = categorical_features.map(
  fName => new StringIndexer()
    .setInputCol(fName)
    .setOutputCol(s"${fName}Index"))
val index_pipeline = new Pipeline().setStages(indexer)
val index_model = index_pipeline.fit(temp)
val dfIndexed = index_model.transform(temp)

// One Hot Encoding
val indexColumns  = dfIndexed.columns.filter(x => x contains "Index")
val encoder:Array[org.apache.spark.ml.PipelineStage] = indexColumns.map(
  fName => new OneHotEncoder()
    .setInputCol(fName)
    .setOutputCol(s"${fName}Vec"))

val encoderPipeline = new Pipeline().setStages(encoder)
val hotEncodedDF = encoderPipeline.fit(dfIndexed).transform(dfIndexed)

val assembler = new VectorAssembler()
  .setInputCols(Array("categoryIndexVec", "deviceIndexVec", "channel_id", "value")).setOutputCol("features")
val pipeline = new Pipeline().setStages(Array(assembler))
val assemblerFit = pipeline.fit(hotEncodedDF)
val inputDf = assemblerFit.transform(hotEncodedDF)

// Fit a model
val gmm = new GaussianMixture().setSeed(0).setK(5).setPredictionCol("prediction")
val dataDf = inputDf.select("features")
val modelFit = gmm.fit(dataDf)
val result = modelFit.transform(dataDf)
// Display the result as DataFrame
result.show()

// output parameters of mixture model model
for (i <- 0 until modelFit.getK) {
  println("weight=%f\nmu=%s\nsigma=\n%s\n" format
    (modelFit.weights(i), modelFit.gaussians(i).mean, modelFit.gaussians(i).cov))
}