package com.cloudera.datascience.dl.bigdl

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, SGD, Top1Accuracy, Top5Accuracy, Trigger}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}

object FineTuneTrain {
  // Logging, redirecting all logs with info level
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    Utils.trainParser.parse(args, Utils.TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("BigDL: Training CNN utilizing pre-trained VGG16 model")
      val sc = new SparkContext(conf)
      sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")

      // Set environment variables and verify them
      Engine.init
      try {
        // Read training data images and transform them
        val trainSet = DataSet.array(Utils.readJpegs(sc, param.trainFolder, param.numNodes, param.numCores, param.imageSize), sc) ->
          BytesToBGRImg() ->
          BGRImgNormalizer(104, 117, 123, 1, 1, 1) ->
          BGRImgToBatch(param.batchSize)

        // Read validation data images and transform them
        val valSet = DataSet.array(Utils.readJpegs(sc, param.valFolder, param.numNodes, param.numCores, param.imageSize), sc) ->
          BytesToBGRImg() ->
          BGRImgNormalizer(104, 117, 123, 1, 1, 1) ->
          BGRImgToBatch(param.batchSize)

        // Create model
        val model = if (param.modelSnapshot.isDefined) {
          println("loading saved model snapshot...")
          Module.load[Float](param.modelSnapshot.get)
        } else if (param.modelPath != "") {
          param.modelName match {
            case "VGG16" =>
              Module.loadCaffe[Float](VGG16NetCaltech(param.classNum),
                param.modelDefPath, param.modelPath, false)
            case _ => throw new IllegalArgumentException(s"${param.modelName}")
          }
        } else {
          param.modelName match {
            case "VGG16" =>
              VGG16NetCaltech(param.classNum)
            case _ => throw new IllegalArgumentException(s"${param.modelName}")
          }
        }

        // set hyperparameters
        val optim = if (param.optimSnapshot.isDefined) {
          println("loading saved optim method snapshot...")
          OptimMethod.load[Float](param.optimSnapshot.get)
        } else {
          new SGD(
            learningRate = param.learningRate,
            learningRateDecay = param.learningRateDecay,
            weightDecay = param.weightDecay,
            momentum = param.momentum,
            learningRateSchedule = SGD.EpochStep(20, 0.1),
            dampening = param.dampening,
            nesterov = param.nesterov
          )
        }

        // Setup distributed training, provide model, training data and loss function
        RandomGenerator.RNG.setSeed(123)
        val optimizer = Optimizer(
          model = model,
          dataset = trainSet,
          criterion = new ClassNLLCriterion[Float]()
        )

        // Setup checkpoint to save the model after "param.iteration"s
        if (param.checkpoint.isDefined) {
          optimizer.setCheckpoint(param.checkpoint.get, Trigger.severalIteration(param.iteration))
        }

        // Enable visualization with TensorBoard
        val trainSummary = TrainSummary(param.logDir, param.appName)
        // As Collecting parameters will slow down the training, it's disabled by default.
        trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(20))
        val validationSummary = ValidationSummary(param.logDir, param.appName)
        optimizer.setTrainSummary(trainSummary)
        optimizer.setValidationSummary(validationSummary)

        // Optimization/ training process using the hyperparameters and other settings setup earlier
        optimizer
          .setOptimMethod(optim)
          // Perform validation after every epoch and output Top1 and Top5 Accuracy
          .setValidation(Trigger.everyEpoch, valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
          // Stop after "param.maxEpoch"
          .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
          .optimize()

        // Reading loss summary
        val trainLoss = trainSummary.readScalar("Loss")
        val validationLoss = validationSummary.readScalar("Loss")

      } finally {
        sc.stop()
      }
    })
  }
}


