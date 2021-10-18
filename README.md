# KotlinDL-mnist

> KotlinDL: Deep Learning Framework written in Kotlin

This project trains a dense neural network using KotlinDL on the mnist dataset.

## Quick-Start

1. Import the project into Intellij Idea IDE

2. Run `main.kt` or build a JAR file:

```
$ ./gradlew shadowJar 
$ java -jar build/libs/hello-kotlindl-1.0-SNAPSHOT-all.jar 
```

Outputs:

```
Extracting 60000 images of 28x28 from /home/wuhanstudio/kotlndl-mnist/cache/datasets/mnist/train-images-idx3-ubyte.gz
Extracting 60000 labels from /home/wuhanstudio/kotlndl-mnist/cache/datasets/mnist/train-labels-idx1-ubyte.gz
Extracting 10000 images of 28x28 from /home/wuhanstudio/kotlndl-mnist/cache/datasets/mnist/t10k-images-idx3-ubyte.gz
Extracting 10000 labels from /home/wuhanstudio/kotlndl-mnist/cache/datasets/mnist/t10k-labels-idx1-ubyte.gz
2021-10-18 11:36:13.974410: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2021-10-18 11:36:13.991707: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 1190500000 Hz
2021-10-18 11:36:13.992207: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f1d7d234db0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-10-18 11:36:13.992244: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ===========================================================================
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - Model: Sequential
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ___________________________________________________________________________
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - Layer (type)                           Output Shape              Param #   
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ===========================================================================
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - input_1(Input)                         [None, 28, 28, 1]         0
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ___________________________________________________________________________
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - flatten_2(Flatten)                     [None, 784]               0
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ___________________________________________________________________________
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - dense_3(Dense)                         [None, 256]               200960
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ___________________________________________________________________________
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - dense_4(Dense)                         [None, 128]               32896
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ___________________________________________________________________________
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - dense_5(Dense)                         [None, 10]                1290
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ___________________________________________________________________________
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ===========================================================================
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - Total trainable params: 235146
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - Total frozen params: 0
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - Total params: 235146
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - ===========================================================================
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 1 loss: 1.5531934 metric: 0.91681665
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 2 loss: 1.5114808 metric: 0.95426667
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 3 loss: 1.5006521 metric: 0.96405
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 4 loss: 1.4932657 metric: 0.97105
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 5 loss: 1.4879782 metric: 0.97581667
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 6 loss: 1.4843937 metric: 0.97915
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 7 loss: 1.4815989 metric: 0.98175
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 8 loss: 1.479125 metric: 0.98373336
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 9 loss: 1.4773757 metric: 0.9853
[main] INFO org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel - epochs: 10 loss: 1.4757954 metric: 0.9867333
EvaluationResult(lossValue=1.4855552911758423, metrics={ACCURACY=0.97607421875})

```
