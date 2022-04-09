from cProfile import label
import numpy as np
import os
import time

def load_train_carla():
    n_carla = len([fname for fname in os.scandir('lidar_semseg/raw_data/binary_files') if fname.is_file])
    n_kitti = len([fname for fname in os.scandir('semantic_kitti_modified_dataset/data_odometry_velodyne_proj/dataset/sequences/00/velodyne') if fname.is_file])
    data = np.zeros((n_carla+n_kitti,64,1024,5))
    data_label = np.zeros((n_carla+n_kitti,64,1024))
    count = 0
    count_2 = 0
    print(n_carla+n_kitti)
    for filename in os.scandir("lidar_semseg/raw_data/binary_files"):
        if str(filename.name).endswith('.bin'):
            scan = np.fromfile("lidar_semseg/raw_data/binary_files/"+filename.name,dtype=np.float32)
            scan = scan.reshape((64,1024,5))
            data[count] = scan
            print(count)
            count = count + 1
            # break
    for filename in os.scandir('semantic_kitti_modified_dataset/data_odometry_velodyne_proj/dataset/sequences/00/velodyne'):
        if str(filename.name).endswith('.bin'):
            scan = np.fromfile('semantic_kitti_modified_dataset/data_odometry_velodyne_proj/dataset/sequences/00/velodyne/'+filename.name,dtype=np.float32)
            scan = scan.reshape((64,1024,5))
            data[count] = scan
            print(count)
            count = count + 1
    for filename in os.scandir("lidar_semseg/raw_data/updated_ground_truth"):
        if str(filename.name).endswith('.label'):
            labels = np.fromfile("lidar_semseg/raw_data/updated_ground_truth/"+filename.name,dtype = np.uint32)
            labels = labels.reshape((64,1024))
            data_label[count_2] = labels
            print(count_2)
            count_2 += 1
    for filename in os.scandir('semantic_kitti_modified_dataset/data_odometry_labels_proj/dataset/sequences/00/labels'):
        if str(filename.name).endswith('.label'):
            labels = np.fromfile('semantic_kitti_modified_dataset/data_odometry_labels_proj/dataset/sequences/00/labels/'+filename.name,dtype = np.uint32)
            labels = labels.reshape((64,1024))
            data_label[count_2] = labels
            print(count_2)
            count_2 += 1
    # print(data[:,:,:,0].shape)
    print(data_label.shape)
    with open("carla_train.npy","wb") as f:
        np.save(f,data)
    with open("carla_train_label.npy","wb") as f:
        np.save(f,data_label)
    return (data,data_label)

if __name__ == '__main__':
    tic_2 = time.time()
    (X_train,X_label) = load_train_carla()
    toc_2 = time.time()
    print(toc_2-tic_2)
    # tic = time.time()
    # #data = np.load("carla_train.npy")
    # toc = time.time()
    # print("Horray Data load zhala")
    # print(data.shape)
    # print(toc-tic)