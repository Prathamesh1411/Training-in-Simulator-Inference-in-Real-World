# Training-in-Simulator-Inference-in-Real-World

## Motivation:
Deep learning in the research field of autonomous driving systems (ADS) always suffers from data acquiring and annotation. For novel sensing modalities such as Light Detection and Ranging (LiDAR), it takes even more effort to annotate since it is difficult for human beings to label on the point clouds. Tackling this issue, we explore the use of synthetic data to train a neural network for real world inference.

## Dataset used: 
  1. [Semantic KITTI](http://www.semantic-kitti.org/dataset.html)
  2. Self generated dataset using Carla Simulator

## Requirements:
  1. [Carla 0.9.12](https://carla.org/2021/08/02/release-0.9.12/): [Steps for installation and system requirements](https://carla.readthedocs.io/en/0.9.12/start_quickstart/) - Method which I used: Package installation from github
  2. [Conda 4.10.3](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
  3. [Numpy 1.21.2](https://numpy.org/install/)
  4. [Tensorflow 2.4.1/Pytorch 1.11.0](https://towardsdatascience.com/guide-to-conda-for-tensorflow-and-pytorch-db69585e32b8)
  5. Carla has maintained detailed documentation about its [core concepts](https://carla.readthedocs.io/en/0.9.12/core_concepts/) and [Python API reference](https://carla.readthedocs.io/en/0.9.12/python_api/)
  
## Method to collect data with carla:
  1. Open a terminal in the main installed `carla` package folder. Run the following command to execute package file and start the simulation:
  ``` 
  ./CarlaUE4.sh  
  ``` 
  2. Copy and paste the file `src/utils/datacollector.py`(in this repo) inside your installed carla package at location `PythonAPI/examples` folder.
  3. Open another terminal inside `PythonAPI/example` and run following command to start collecting data:
  ``` 
  python3 datacollector.py --sync -m Town01 -l  
  ```
  4. Optional- Run in parallel in new terminal 
  ```
  python3 generate_traffic.py -n 50 -w 50      # spawn 50 vehicles and pedestrians 
  python3 dynamic_weather.py                   # collect the dataset using varying weather conditions. 
  ```
## Folder structure:
```
/PythonAPI/examples/dataset/
          └── sequences/
                  ├── Town01/
                  │   ├── poses.txt
                  │   ├── camera_depth/
                  │   │     ├── images/
                  │   │     │     ├ WorldSnapshot(frame=26)_1.png
                  │   │     │     ├ WorldSnapshot(frame=27)_1.png
                  │   │     ├── raw_data/
                  │   │     │     ├── binary 
                  │   │     │     │    ├ WorldSnapshot(frame=26).npy
                  │   │     │     │    ├ WorldSnapshot(frame=27).npy
                  │   ├── camera_rgb/
                  │   │     ├── images/
                  │   │     │     ├ WorldSnapshot(frame=26)_0.png
                  │   │     │     ├ WorldSnapshot(frame=27)_0.png
                  │   │     ├── raw_data/
                  │   │     │     ├── binary 
                  │   │     │     │    ├ WorldSnapshot(frame=26).npy
                  │   │     │     │    ├ WorldSnapshot(frame=27).npy
                  │   ├── camera_semseg/
                  │   │     ├── images/
                  │   │     │     ├ WorldSnapshot(frame=26)_2.png
                  │   │     │     ├ WorldSnapshot(frame=27)_2.png
                  │   │     ├── raw_data/
                  │   │     │     ├── binary 
                  │   │     │     │    ├ WorldSnapshot(frame=26).npy
                  │   │     │     │    ├ WorldSnapshot(frame=27).npy
                  │   ├── lidar_semseg/
                  │   │     ├── images/
                  │   │     │     ├ 00000026.png
                  │   │     │     ├ 00000027.png
                  │   │     ├── raw_data/
                  │   │     │     ├── binary_files 
                  │   │     │     │    ├ WorldSnapshot(frame=26)_3.bin
                  │   │     │     │    ├ WorldSnapshot(frame=27)_3.bin
                  │   │     │     ├── ground_truth
                  │   │     │     │    ├ WorldSnapshot(frame=26)_3.label
                  │   │     │     │    ├ WorldSnapshot(frame=27)_3.label
                  │   │     │     ├── ply_files 
                  │   │     │     │    ├ WorldSnapshot(frame=26)_3.ply
                  │   │     │     │    ├ WorldSnapshot(frame=27)_3.ply
                  │   │     │     ├── updated_ground_truth                      # Label remapping to be in accordance with Semantic KITTI labels
                  │   │     │     │    ├ WorldSnapshot(frame=26)_3_new.label
                  │   │     │     │    ├ WorldSnapshot(frame=27)_3_new.label
                  ├── Town02/
                  ├── Town03/
```

`lidar_semseg/images` contains Spherically Projected images(Range-view). 
`lidar_semseg/raw_data/updated_ground_truth` contains label remapping for only 3 classes:
  a. 0 - background
  b. 1 - car
  c. 2 - pedestrians
  
  ## Semantic-KITTI modifier:
  1. Download dataset - [Semantic KITTI](http://www.semantic-kitti.org/dataset.html)
  2. Run `KITTI_data_modifer.py` to create projected bin files, label files and images of given sequence `data_odometry_velodyne_proj, data_odometry_labels_projimg, data_odometry_labels_proj`.
  3. Semantic KITTI labels are remapped corresponding to Carla labels.
  
  
  ## Our Results:

![Results](results/predicted_kitti.gif)
  
