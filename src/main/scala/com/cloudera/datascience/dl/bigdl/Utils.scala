package com.cloudera.datascience.dl.bigdl

import org.apache.spark.SparkContext
import scopt.OptionParser
import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.bigdl.dataset.image.BGRImage
import javax.imageio.ImageIO
import org.apache.spark.rdd.RDD

private[bigdl] object Utils {
  def readJpegs(sc: SparkContext, imagePath: String, imageSize: Int): RDD[ByteRecord] = {
    val jpegs = sc.binaryFiles(imagePath).mapPartitions{ it =>
      it.map { case (path, img) =>
        val regex = ".+\\/(\\d{3})_\\d{4}\\.jpg".r
        val label = path match {
          case regex(l) => l.toFloat
          case _ =>
            throw new IllegalArgumentException(s"Could not parse label from path: $path")
        }
        val inputStream = img.open()
        val bufferedImage = ImageIO.read(inputStream)
        inputStream.close()
        val byteImage = BGRImage.resizeImage(bufferedImage, imageSize, imageSize)
        (byteImage, label)
      }
    }
    // Randomize the data before it is converted to mini batches, this will help reduce the loss
    val jpegs2 = jpegs.sample(withReplacement = false, 1.0, seed=123)
      .map { case(i, l) => ByteRecord(i, l) }
    jpegs2
  }

  case class TrainParams(
                          numNodes: Int = 4,
                          numCores: Int = 4,
                          trainFolder: String = "",
                          valFolder: String = "",
                          classNum: Int = 257,
                          modelName: String = "VGG16",
                          imageSize: Int = 224,   // VGG16 pre-trained model requires the image size to be 224 * 224
                          checkpoint: Option[String] = None,
                          modelDefPath: String = "",
                          modelPath: String = "",
                          batchSize: Int = 16,
                          learningRate: Double = 0.1,
                          learningRateDecay: Double = 0.0,
                          dropPercentage: Double = 0.0,
                          momentum: Double = 0.9,
                          maxEpoch: Int = 10,
                          dampening: Double = 0.0,
                          nesterov: Boolean = false,
                          weightDecay: Double = 0.0,
                          iteration: Int = 1000,
                          optimTechnique: String = "SGD",
                          logDir: String = "",
                          appName: String = "",
                          modelSnapshot: Option[String] = None,
                          optimSnapshot: Option[String] = None
                        )

  val trainParser = new OptionParser[TrainParams]("BigDL CNN FineTune Example") {
    head("Use pre-trained model")
    opt[Int]("numNodes")
      .text("num of nodes/ executors")
      .action((x, c) => c.copy(numNodes = x))
    opt[Int]("numCores")
      .text("num of cores")
      .action((x, c) => c.copy(numCores = x))
    opt[String]("trainFolder")
      .text("hdfs location of train image files")
      .action((x, c) => c.copy(trainFolder = x))
    opt[String]("valFolder")
      .text("hdfs location of validation image files")
      .action((x, c) => c.copy(valFolder = x))
    opt[Int]("classNum")
      .text("num of class labels")
      .action((x, c) => c.copy(classNum = x))
    opt[String]("modelName")
      .text("model name")
      .action((x, c) => c.copy(modelName = x))
    opt[String]("modelDefPath")
      .text("model definition file")
      .action((x, c) => c.copy(modelDefPath = x))
    opt[String]("modelPath")
      .text("existing model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]("imageSize")
      .text("image size for the model specified")
      .action((x, c) => c.copy(imageSize = x))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]("iteration")
      .text("when to save the model")
      .action((x, c) => c.copy(iteration = x))
    opt[String]("optimTechnique")
      .text("specify optimization technique")
      .action((x, c) => c.copy(optimTechnique = x))
    opt[Int]("batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Double]("learningRate")
      .text("Learning Rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("learningRateDecay")
      .text("Learning Rate Decay")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[Double]("dropPercentage")
      .text("Drop Percentage")
      .action((x, c) => c.copy(dropPercentage = x))
    opt[Double]("momentum")
      .text("Momentum")
      .action((x, c) => c.copy(momentum = x))
    opt[Int]("maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Double]("dampening")
      .text("Dampening")
      .action((x, c) => c.copy(dampening = x))
    opt[Boolean]("nesterov")
      .text("Nesterov")
      .action((x, c) => c.copy(nesterov = x))
    opt[Double]("weightDecay")
      .text("weightDecay")
      .action((x, c) => c.copy(weightDecay = x))
    opt[String]("logDir")
      .text("log directory")
      .action((x, c) => c.copy(logDir = x))
    opt[String]("appName")
      .text("app name")
      .action((x, c) => c.copy(appName = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("optim")
      .text("optim method location")
      .action((x, c) => c.copy(optimSnapshot = Some(x)))
  }

  case class TestParams(
                         numNodes: Int = 4,
                         numCores: Int = 4,
                         folder: String = "./",
                         model: String = "",
                         batchSize: Int = 16,
                         imageSize: Int = 224  // VGG16 pre-trained model requires the image size to be 224 * 224
                       )

  val testParser = new OptionParser[TestParams]("BigDL CNN FineTune Example") {
    head("Test pre-trained model")
    opt[Int]("numNodes")
      .text("num of nodes/ executors")
      .action((x, c) => c.copy(numNodes = x))
    opt[Int]("numCores")
      .text("num of cores")
      .action((x, c) => c.copy(numCores = x))
    opt[String]("folder")
      .text("hdfs location of image files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
    opt[Int]("batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]("imageSize")
      .text("image size for the model specified")
      .action((x, c) => c.copy(imageSize = x))
  }
}

