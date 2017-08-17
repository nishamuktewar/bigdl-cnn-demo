package com.cloudera.datascience.dl.bigdl

import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BGRImgToBatch, BytesToBGRImg, BGRImgNormalizer}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim._

object FineTuneTest {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    Utils.testParser.parse(args, Utils.TestParams()).foreach { param =>

      val conf = Engine.createSparkConf().setAppName("BigDL: Testing fine-tuned VGG16 model")
      val sc = new SparkContext(conf)
      sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")
      // Set environment variables and verify them
      Engine.init

      // Read test data images and transform them
      val testSet =
        DataSet.rdd(Utils.readJpegs(sc, param.folder, param.imageSize)) ->
          BytesToBGRImg() ->
          BGRImgNormalizer(104, 117, 123, 1, 1, 1) ->
          BGRImgToBatch(param.batchSize)

      // Load model
      val model = Module.load[Float](param.model)
      val validator = Validator(model, testSet)

      val result = validator.test(Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
      result.foreach{
        case (value, metric) => println(s"$metric is $value")
      }
    }
  }
}
