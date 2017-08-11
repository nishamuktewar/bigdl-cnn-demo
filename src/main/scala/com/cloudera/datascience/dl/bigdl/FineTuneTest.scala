package com.cloudera.datascience.dl.bigdl

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BGRImgToBatch, BytesToBGRImg, BGRImgNormalizer}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext


object FineTuneTest {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    Utils.testParser.parse(args, Utils.TestParams()).foreach { param =>

      val conf = Engine.createSparkConf().setAppName("BigDL: Testing fine-tuned VGG16 model")
      val sc = new SparkContext(conf)
      sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")

      Engine.init

      val validationSet =
        DataSet.array(Utils.readJpegs(sc, param.folder, param.imageSize, param.numNodes, param.numCores), sc) ->
          BytesToBGRImg() ->
          BGRImgNormalizer(104, 117, 123, 1, 1, 1) ->
          BGRImgToBatch(param.batchSize)

      // Load model
      val model = Module.load[Float](param.model)
      val validator = Validator(model, validationSet)

      val result = validator.test(Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
      result.foreach(r => println(s"${r._2} is ${r._1}"))
    }
  }
}
