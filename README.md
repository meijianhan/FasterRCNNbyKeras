# keras_frcnn

This repo is an implementation of faster r-cnn using Keras and Tensorflow. The Tensorflow part borrows the ref: https://github.com/endernewton/tf-faster-rcnn a lot.


## Introduction ##

The implementation is aiming to build the Keras interface based on a Tensorflow FRCNN code [1]. Most of the tf functions are packed by using Keras Lambda. And the structure of the code is changed.


## Testing Results and Some Notes ##
 - ##### Pascal VOC07

   Trained only using Pascal VOC07 training set and tested on VOC07 testing set, after 7 epochs, the mAP is around 68% (+-1). The only data pre-processing is the left-right flipping of each image. The convolution feature from a VGG16 is used as the shared convolution feature.

   Tips: during training, the Adam optimizer is used. The lr is set as 1e-5, which influences the final result a lot. However, in [1], the SGD with a lr of 1e-3 is used. And I cannot get a convergence result by using the same setting. I have not figured out the reason why the two lr are different so much.

## Required Environment ##

 - [Ubuntu](https://www.ubuntu.com/) 16.04

 - Python 2/3, in case you need the sufficient scientific computing packages, we recommend you to install [anaconda](https://www.anaconda.com/what-is-anaconda/).

 - [Tensorflow](https://www.tensorflow.org/) >= 1.5.0

 - [Keras](https://keras.io/) >= 2.2.0

 - Optional: if you need GPUs acceleration, please install [CUDA](https://developer.nvidia.com/cuda-toolkit) that the version requires >= 9.0


## Training and Testing ##
- ##### Data preparation and setup

  Follow the ref: https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models. Download Pascal VOC 07 dataset and build soft link in folder ./data/, then name the link as "VOCdevkit2007"

  To compile the lib, move into the lib folder "cd ./lib". According to your hardware, change the "-arch" parameter in setup.py line: 130. Then run

  ```
  make
  ```

  Tips: to find your hardware compiling setting, you can refer to: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/. For me using the Titan XP, I set the "-arch" as "sm_61".

- ##### Training

  Build the model weight saving folder "../model_save"

  Run the following script:

  ```
  python train.py
  ```

- ##### Testing

  Build the test output saving folder "../test_save"

  Run the following script:

  ```
  python test.py
  ```


## Reference ##

* [1] tf-faster-rcnn: https://github.com/endernewton/tf-faster-rcnn

