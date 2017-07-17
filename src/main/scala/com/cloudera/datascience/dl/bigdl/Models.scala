package com.cloudera.datascience.dl.bigdl

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object VGG16NetCaltech {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)
      .setName("conv1_1"))
    model.add(ReLU(true).setName("relu1_1"))
    model.add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
      .setName("conv1_2"))
    model.add(ReLU(true).setName("relu1_2"))
    model.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool1"))
    model.add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)
      .setName("conv2_1"))
    model.add(ReLU(true).setName("relu2_1"))
    model.add(SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
      .setName("conv2_2"))
    model.add(ReLU(true).setName("relu2_2"))
    model.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool2"))
    model.add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)
      .setName("conv3_1"))
    model.add(ReLU(true).setName("relu3_1"))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
      .setName("conv3_2"))
    model.add(ReLU(true).setName("relu3_2"))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
      .setName("conv3_3"))
    model.add(ReLU(true).setName("relu3_3"))
    model.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3"))
    model.add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)
      .setName("conv4_1"))
    model.add(ReLU(true).setName("relu4_1"))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
      .setName("conv4_2"))
    model.add(ReLU(true).setName("relu4_2"))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
      .setName("conv4_3"))
    model.add(ReLU(true).setName("relu4_3"))
    model.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4"))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
      .setName("conv5_1"))
    model.add(ReLU(true).setName("relu5_1"))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
      .setName("conv5_2"))
    model.add(ReLU(true).setName("relu5_2"))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
      .setName("conv5_3"))
    model.add(ReLU(true).setName("relu5_3"))
    model.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool5"))
    model.add(View(512 * 7 * 7))
    model.add(Linear(512 * 7 * 7, 4096).setName("fc6"))
    model.add(ReLU(true).setName("relu6"))
    model.add(Dropout(0.5).setName("drop6"))
    model.add(Linear(4096, 4096).setName("fc7"))
    model.add(ReLU(true).setName("relu7"))
    model.add(Dropout(0.5).setName("drop7"))
    model.add(Linear(4096, classNum).setName("fc8_caltech256")) // Renaming the last FC layer
    model.add(LogSoftMax().setName("loss"))
    model
  }
}

