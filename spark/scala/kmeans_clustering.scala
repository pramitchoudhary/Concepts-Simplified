import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline

val df = sqlContext.createDataFrame(Seq(
  (0, "a", "Desktop", 2),
  (1, "b", "Mobile", 2),
  (2, "c", "", 3),
  (3, "", "appleTV", 3),
  (4, "a", "Desktop", 2),
  (5, "c", "Mobile", 2)
)).toDF("id", "category", "device", "channel_id")

val temp = df.na.replace(Seq("category", "device"), Map("" -> "NA"))
temp.show()

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")
val encoder = new OneHotEncoder().setInputCol("categoryIndex").setOutputCol("categoryVec")
val assembler = new VectorAssembler().setInputCols(Array("categoryVec", "channel_id")).setOutputCol("features")
val model = new KMeans().setSeed(0).setK(5).setMaxIter(10).setFeaturesCol("features").setPredictionCol("prediction")

val pipeline = new Pipeline().setStages(Array(indexer, encoder, assembler, model))

val kMeansModelFit = pipeline.fit(temp)

val transformedFeaturesDF = kMeansModelFit.transform(temp)
transformedFeaturesDF.show()
val cost = kMeansModelFit.stages(3).asInstanceOf[KMeansModel].computeCost(transformedFeaturesDF)
println(cost)
println("Cluster Centers: ")
kMeansModelFit.stages(3).asInstanceOf[KMeansModel].clusterCenters.foreach(println)