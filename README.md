# Training-in-Simulator-Inference-in-Real-World

### Motivation:
Deep learning in the research field of autonomous driving systems (ADS) always suffers from data acquiring and annotation. For novel sensing modalities such as Light Detection and Ranging (LiDAR), it takes even more effort to annotate since it is difficult for human beings to label on the point clouds. Tackling this issue, we explore the use of synthetic data to train a neural network for real world inference.

### Dataset used: 
  1. [Semantic KITTI](http://www.semantic-kitti.org/dataset.html)
  2. Self generated dataset using Carla Simulator

### Requirements:
  1. [Carla 0.9.12](https://carla.org/2021/08/02/release-0.9.12/): [Steps for installation and system requirements](https://carla.readthedocs.io/en/0.9.12/start_quickstart/) - Method which I used - Package installation from github
  2. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
  3. [Numpy](https://numpy.org/install/)
  4. [Tensorflow/Pytorch](https://towardsdatascience.com/guide-to-conda-for-tensorflow-and-pytorch-db69585e32b8)
