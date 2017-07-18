# bigdl-cnn-demo

## Setting up BigDL

This project uses an as-yet-unreleased version 0.2.0 of BigDL. To use it, you must first download and build BigDL.

```
git clone https://github.com/intel-analytics/BigDL
cd BigDL
mvn -DskipTests -Pspark_2.x clean install
```

## Running models

Using a pre-trained VGG16 model to predict image classes in Caltech-256

Download VGG16 model and the model definition file from here: https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

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
- checkpoint: location to be specified when you want to save the model & optimization snapshot
- iteration: the iteration number at which to capture the checkpoint, default 1000
- model: specify the model snapshot file to be used as the starting point when re-training an existing model (available from prior runs when checkpoint and iteration is specified)
- optim: specify the optimization snapshot file to be used as the starting point when re-training an existing model (available from prior runs when checkpoint and iteration is specified)

### Using pre-trained VGG16 model on actual data
- Takes around 3 hours to finish one epoch 
- Training set result: Top1Accuracy 70%, Top5Accuracy 88%
- Validation result: Top1Accuracy 48%, Top5Accuracy 70%

#### Notes: 
- Source bigdl before the spark-submit command, this might go away in future
- Saving model and optimization snapshot files
 - If you want to save model snapshot for later use (example, run additional epochs), make sure you specify the checkpoint location along with the "iteration" parameter telling the application where to save the snapshot. In this case, the training sample data has 20,812 records and suppose we want to save the model after training all the observations, then this should be set to 1301 (num of records/ batchSize = 20812/16, round it to the next number)

````
. bigdl.sh 

spark2-submit \
 --master yarn --deploy-mode client \
 --executor-memory 25g \
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
    --trainFolder hdfs:///user/shendrickson/256_ObjectCategories_Stratified/train/ \
    --valFolder hdfs:///user/shendrickson/256_ObjectCategories_Stratified/valid/ \
    --classNum 257 \
    --batchSize 16 \
    --maxEpoch 5 \
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
    --logDir /home/nisha/Projects/bigdl-cnn-demo/logs \
    --appName epochs5 \
    --iteration 1301 \
    --checkpoint ./bigdl-models
````

### Retreiving model and optimization snapshot files to retrain the data
````
. bigdl.sh 

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
    --trainFolder hdfs:///user/shendrickson/256_ObjectCategories_Stratified/train/ \
    --valFolder hdfs:///user/shendrickson/256_ObjectCategories_Stratified/valid/ \
    --classNum 257 \
    --batchSize 16 \
    --maxEpoch 10 \
    --model ./bigdl-models/20170717_154249/model.2602 \
    --optim ./bigdl-models/20170717_154249/optimMethod.2602 \
    --logDir /home/nisha/Projects/bigdl-cnn-demo/logs \
    --appName epochs5 \
    --iteration 1301 \
    --checkpoint ./bigdl-models
````

### Testing model performance on test/ validation set
````
. bigdl.sh

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
    --folder hdfs:///user/shendrickson/256_ObjectCategories_Stratified/test/ \
    --model ./bigdl-models/20170717_154249/model.2602
````
