# bigdl-cnn-demo

## Setting up BigDL
This project uses an as-yet-unreleased version 0.3.0 of BigDL. To use it, you must first download and build BigDL.
NOTE: Once the 0.2.1 of BigDL is out this actually works with Spark 2.2 
```
git clone https://github.com/intel-analytics/BigDL
cd BigDL
mvn -DskipTests -Dspark.version=2.2.0 -Pspark_2.x clean install
```

## Build
```
cd [wherever the project directory is]
mvn -Dspark.version=2.2.0.cloudera1 clean package
```

## Setup 
### Copy image data on edge node
```
curl -L -O http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar
mkdir -p 256_ObjectCategories/train 256_ObjectCategories/test 256_ObjectCategories/valid
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | shuf | head -n 5000 | xargs -I {} mv {} ./256_ObjectCategories/valid/
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | shuf | head -n 6000 | xargs -I {} mv {} ./256_ObjectCategories/test/
find ./256_ObjectCategories/ -regextype posix-extended -type f -regex ".*/[0-9]{3}\..+" -print | xargs -I {} mv {} ./256_ObjectCategories/train/
find ./256_ObjectCategories/ -regextype posix-extended -type d -regex ".*/[0-9]{3}\..+" -delete
```
### Copy image data to HDFS
```
hadoop fs -put ./256_ObjectCategories
```
### Copy VGG16 model and definition file on edge node
Reference: https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
```
curl -L -O http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
curl -L -O https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
```
## Deploy
### Application parameters
- numNodes: number of executors
- numCores: number of cores / executor
- modelName: VGG16 
- modelPath: location of the pre-trained VGG16 model 
- modelDefPath: location of the VGG16 model definition file
- batchSize: The mini-batch size for training. Note that this value needs to be divisible by "numNodes * numCores"
- imageSize: The size of the image to which it needs to be re-scaled. When using VGG16 it should be 224
- trainFolder: training data location
- valFolder: validation data location
- classNum: number of class/ labels
- maxEpoch: number of passes over the training data
- model parameters associated with SGD optimization
  - learningRate: learning rate, default - 0.1
  - learningRateDecay: learning rate decay, default - 0.0
  - weightDecay: weight decay (L2 regularization) - 0.0
  - momentum: default 0.9
  - nesterov: enable/ disable nesterov momentum. default false, requires a momentum value and dampening = 0
  - dampening: default 0.0
- logDir: location to store the training and validation summary results
- appName: Run name, example - run1. It will create a sub-directory within "logDir" with the name specified
- checkpoint: location to be specified where you want to save the model & optimization snapshot
- iteration: the iteration number at which to capture the checkpoint, default 1000. In this case, the training sample data has 20,812 records and suppose we want to save the model after training all the observations, then this should be set to 1301 (num of records/ batchSize = 20812/16, round it to the next number
- model: specify the model snapshot file to be used as the starting point when re-training an existing model (available from prior runs when checkpoint and iteration is specified)
- optim: specify the optimization snapshot file to be used as the starting point when re-training an existing model (available from prior runs when checkpoint and iteration is specified)
````
spark2-submit \
 --master yarn --deploy-mode client \
 --executor-memory 27g \
 --driver-memory 20g \
 --conf spark.yarn.executor.memoryOverhead=12g \
 --conf spark.driver.extraJavaOptions=-Dbigdl.check.singleton=false \
 --conf spark.shuffle.reduceLocality.enabled=false \
 --conf spark.driver.maxResultSize=5g \
 --num-executors 4 --executor-cores 4 \
 --conf spark.dynamicAllocation.enabled=false \
 --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
 --conf spark.rpc.message.maxSize=800 \
 --conf "spark.kryoserializer.buffer.max=2047m" \
 --class "com.cloudera.datascience.dl.bigdl.FineTuneTrain" \
 bigdl-cnn-demo-1.0-SNAPSHOT-jar-with-dependencies.jar \
    --numNodes 4 \
    --numCores 4 \
    --imageSize 224 \
    --trainFolder hdfs:///path/to/256_ObjectCategories/train/ \
    --valFolder hdfs:///path/to/256_ObjectCategories/valid/ \
    --classNum 257 \
    --batchSize 16 \
    --maxEpoch 2 \
    --modelName VGG16 \
    --modelPath ./VGG_ILSVRC_16_layers.caffemodel \
    --modelDefPath ./VGG_ILSVRC_16_layers_deploy.prototxt \
    --optimTechnique SGD \
    --learningRate 0.0001 \
    --weightDecay 0.0001 \
    --dropPercentage 0.0 \
    --momentum 0.8 \
    --nesterov true \
    --dampening 0.0 \
    --logDir ./logs \
    --appName run1 \
    --iteration 1301 \
    --checkpoint ./bigdl-models
````

### Retreiving model and optimization snapshot files to retrain the data
````
spark2-submit \
 --master yarn --deploy-mode client \
 --executor-memory 27g \
 --driver-memory 20g \
 --conf spark.yarn.executor.memoryOverhead=12g \
 --conf spark.driver.extraJavaOptions=-Dbigdl.check.singleton=false \
 --conf spark.shuffle.reduceLocality.enabled=false \
 --conf spark.driver.maxResultSize=5g \
 --num-executors 4 --executor-cores 4 \
 --conf spark.dynamicAllocation.enabled=false \
 --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
 --conf spark.rpc.message.maxSize=800 \
 --conf "spark.kryoserializer.buffer.max=2047m" \
 --class "com.cloudera.datascience.dl.bigdl.FineTuneTrain" \
 bigdl-cnn-demo-1.0-SNAPSHOT-jar-with-dependencies.jar \
    --numNodes 4 \
    --numCores 4 \
    --imageSize 224 \
    --trainFolder hdfs:///path/to/256_ObjectCategories/train/ \
    --valFolder hdfs:///path/to/256_ObjectCategories/valid/ \
    --classNum 257 \
    --batchSize 16 \
    --maxEpoch 10 \
    --model ./bigdl-models/20170717_154249/model.2602 \
    --optim ./bigdl-models/20170717_154249/optimMethod.2602 \
    --logDir ./logs \
    --appName run1 \
    --iteration 1301 \
    --checkpoint ./bigdl-models
````

### Testing model performance on test/ validation set
````
spark2-submit \
 --master yarn --deploy-mode client \
 --executor-memory 16g \
 --driver-memory 20g \
 --conf spark.yarn.executor.memoryOverhead=12g \
 --conf spark.driver.extraJavaOptions=-Dbigdl.check.singleton=false \
 --conf spark.shuffle.reduceLocality.enabled=false \
 --conf spark.driver.maxResultSize=5g \
 --num-executors 4 --executor-cores 4 \
 --conf spark.dynamicAllocation.enabled=false \
 --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
 --conf spark.rpc.message.maxSize=800 \
 --conf "spark.kryoserializer.buffer.max=2047m" \
 --class "com.cloudera.datascience.dl.bigdl.FineTuneTest" \
 bigdl-cnn-1.0-SNAPSHOT-jar-with-dependencies.jar \
    --numNodes 4 \
    --numCores 4 \
    --imageSize 224 \
    --batchSize 16 \
    --folder hdfs:///path/to/256_ObjectCategories/test/ \
    --model ./bigdl-models/20170717_154249/model.2602
````
