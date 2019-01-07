# keras_frcnn

This repo is an implementation of [Faster R-CNN](https://arxiv.org/abs/1506.01497) integrating both Keras and Tensorflow. We use a lot of [endernewton](https://github.com/endernewton)‘s tensorflow code, and the reference is that: https://github.com/endernewton/tf-faster-rcnn


## Introduction ##

Our implementation is aiming to build the Keras interface based on a Tensorflow Faster R-CNN code [1]. Most of the tensorflow functions are packed by using Keras Lambda function. We reconstruct the code based on the above goals.


## Benchmark ##
 - ##### Pascal VOC 2007

model     | #GPUs | batch size |lr        | max_epoch     | mem/GPU | mAP (%) 
---------|--------|-----|--------|-----|--------|-----
VGG-16     | 1 | 1    |1e-5| 7  | 8817 MB  | 66.0

  

  - ##### Pascal VOC 2007 + 2012

model     | #GPUs | batch size |lr        | max_epoch     | mem/GPU | mAP (%) 
---------|--------|-----|--------|-----|--------|-----
VGG-16     | 1 | 1    |1e-5| 7  | 8817 MB  | 72.2




  - ##### COCO 2014

model     | #GPUs | batch size |lr        | max_epoch     | mem/GPU | mAP (%) 
---------|--------|-----|--------|-----|--------|-----
VGG-16     | 1 | 1    |1e-5| 7  | 8817 MB  | 31.2


<!---   Trained only using Pascal VOC07 training set and tested on VOC07 testing set, after 7 epochs, the mAP is around 68% (+-1). The only data pre-processing is the left-right flipping of each image. The convolution feature from a VGG16 is used as the shared convolution feature.

   Tips: during training, the Adam optimizer is used. The lr is set as 1e-5, which influences the final result a lot. However, in [1], the SGD with a lr of 1e-3 is used. And I cannot get a convergence result by using the same setting. I have not figured out the reason why the two lr are different so much.
--->

## Required Environment ##

 - [Ubuntu](https://www.ubuntu.com/) 16.04

 - Python 2/3, in case you need the sufficient scientific computing packages, we recommend you to install [anaconda](https://www.anaconda.com/what-is-anaconda/).

 - [Tensorflow](https://www.tensorflow.org/) >= 1.5.0

 - [Keras](https://keras.io/) >= 2.2.0

 - Optional: if you need GPUs acceleration, please install [CUDA](https://developer.nvidia.com/cuda-toolkit) that the version requires >= 9.0


## Tutorial ##
- ##### Data preparation and setup

  Follow the ref: https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models. Download Pascal VOC 07 dataset and build soft link in folder ./data/, then name the link as "VOCdevkit2007"

  To compile the lib, move into the lib folder "cd ./lib". According to your hardware, change the "-arch" parameter in setup.py line: 130. Then run

  ```
  make
  ```

  Tips: to find your hardware compiling setting, you can refer to: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/. For me using the Titan XP, I set the "-arch" as "sm_61".

- ##### Training

  Build the model weight saving folder "../output/[NET]/"

  Download pre-trained models and weights:
  ```
  mkdir net_weights
  wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
  cd ..

  ```

  Run the following script:

  ```
  ./scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16} is the network arch to use,
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
  ```

- ##### Testing

  Build the test output saving folder "../output/[NET]/"

  Run the following script:

  ```
  ./scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16} is the network arch to use,
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./scripts/test_faster_rcnn.sh 0 pascal_voc vgg16

  ```


## Reference ##

* [1] tf-faster-rcnn: https://github.com/endernewton/tf-faster-rcnn

