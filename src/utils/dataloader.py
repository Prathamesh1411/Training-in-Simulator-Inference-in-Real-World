import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler

def load_data(path="",sequences={"kitti":['05','06','07'],"carla":["Town01","Town02"]},normalized = True, shuffle = True):
    n = 0
    if ("kitti" in sequences) and ("carla" in sequences):
        num_seq = len(sequences["kitti"]) + len(sequences["carla"])
        for i in sequences["carla"]:
            temp_n = len([fname for fname in os.scandir(path+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/binary_files') if fname.is_file])
            n += temp_n
        for i in sequences["kitti"]:
            temp_n = len([fname for fname in os.scandir(path+'/kitti/data_odometry_velodyne_proj/dataset/sequences/'+str(i)+'/velodyne') if fname.is_file])
            n += temp_n
        data = np.zeros((n,64,1024,5))
        data_labels = np.zeros((n,64,1024))
        count = 0
        count_2 = 0
        for i in sequences["carla"]:
            for filename in os.scandir(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/binary_files'):
                if str(filename.name).endswith('.bin'):
                    scan = np.fromfile(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/binary_files/'+filename.name,dtype=np.float32)
                    scan = scan.reshape((64,1024,5))
                    if normalized:
                        scan_normalized = np.zeros(scan.shape)
                        scan_normalized[:,:,0] = MinMaxScaler().fit_transform(scan[:,:,0])
                        scan_normalized[:,:,1] = MinMaxScaler().fit_transform(scan[:,:,1]) 
                        scan_normalized[:,:,2] = MinMaxScaler().fit_transform(scan[:,:,2])
                        scan_normalized[:,:,3] = MinMaxScaler().fit_transform(scan[:,:,3])
                        scan_normalized[:,:,4] = MinMaxScaler().fit_transform(scan[:,:,4])
                        data[count] = scan_normalized
                        count = count + 1
                    else:
                        data[count] = scan
                        count = count + 1
            for filename in os.scandir(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/updated_ground_truth'):
                if str(filename.name).endswith('.label'):
                    labels = np.fromfile(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/updated_ground_truth/'+filename.name,dtype = np.uint32)
                    labels = labels.reshape((64,1024))
                    data_labels[count_2] = labels
                    count_2 += 1
        for i in sequences["kitti"]:
            for filename in os.scandir(str(path)+'/kitti/data_odometry_velodyne_proj/dataset/sequences/'+str(i)+'/velodyne'):
                if str(filename.name).endswith('.bin'):
                    scan = np.fromfile(str(path)+'/kitti/data_odometry_velodyne_proj/dataset/sequences/'+str(i)+'/velodyne/'+filename.name,dtype=np.float32)
                    scan = scan.reshape((64,1024,5))
                    if normalized:
                        scan_normalized = np.zeros(scan.shape)
                        scan_normalized[:,:,0] = MinMaxScaler().fit_transform(scan[:,:,0])
                        scan_normalized[:,:,1] = MinMaxScaler().fit_transform(scan[:,:,1]) 
                        scan_normalized[:,:,2] = MinMaxScaler().fit_transform(scan[:,:,2])
                        scan_normalized[:,:,3] = MinMaxScaler().fit_transform(scan[:,:,3])
                        scan_normalized[:,:,4] = MinMaxScaler().fit_transform(scan[:,:,4])
                        data[count] = scan_normalized
                        count = count + 1
                    else:
                        data[count] = scan
                        count = count + 1
            for filename in os.scandir(str(path)+'/kitti/data_odometry_labels_proj/dataset/sequences/'+str(i)+'/labels'):
                if str(filename.name).endswith('.label'):
                    labels = np.fromfile(str(path)+'/kitti/data_odometry_labels_proj/dataset/sequences/'+str(i)+'/labels/'+filename.name,dtype = np.uint32)
                    labels = labels.reshape((64,1024))
                    data_labels[count_2] = labels
                    count_2 += 1
    elif ("kitti" in sequences) and ("carla" not in sequences):
        for i in sequences["kitti"]:
            temp_n = len([fname for fname in os.scandir(path+'/kitti/data_odometry_velodyne_proj/dataset/sequences/'+str(i)+'/velodyne') if fname.is_file])
            n += temp_n 
        data = np.zeros((n,64,1024,5))
        data_labels = np.zeros((n,64,1024))
        count = 0
        count_2 = 0
        for i in sequences["kitti"]:
            for filename in os.scandir(str(path)+'/kitti/data_odometry_velodyne_proj/dataset/sequences/'+str(i)+'/velodyne'):
                if str(filename.name).endswith('.bin'):
                    scan = np.fromfile(str(path)+'/kitti/data_odometry_velodyne_proj/dataset/sequences/'+str(i)+'/velodyne/'+filename.name,dtype=np.float32)
                    scan = scan.reshape((64,1024,5))
                    if normalized:
                        scan_normalized = np.zeros(scan.shape)
                        scan_normalized[:,:,0] = MinMaxScaler().fit_transform(scan[:,:,0])
                        scan_normalized[:,:,1] = MinMaxScaler().fit_transform(scan[:,:,1]) 
                        scan_normalized[:,:,2] = MinMaxScaler().fit_transform(scan[:,:,2])
                        scan_normalized[:,:,3] = MinMaxScaler().fit_transform(scan[:,:,3])
                        scan_normalized[:,:,4] = MinMaxScaler().fit_transform(scan[:,:,4])
                        data[count] = scan_normalized
                        count = count + 1
                    else:
                        data[count] = scan
                        count = count + 1
            for filename in os.scandir(str(path)+'/kitti/data_odometry_labels_proj/dataset/sequences/'+str(i)+'/labels'):
                if str(filename.name).endswith('.label'):
                    labels = np.fromfile(str(path)+'/kitti/data_odometry_labels_proj/dataset/sequences/'+str(i)+'/labels/'+filename.name,dtype = np.uint32)
                    labels = labels.reshape((64,1024))
                    data_labels[count_2] = labels
                    count_2 += 1
    elif ("kitti" not in sequences) and ("carla" in sequences):
        for i in sequences["carla"]:
            temp_n = len([fname for fname in os.scandir(path+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/binary_files') if fname.is_file])
            n += temp_n
        data = np.zeros((n,64,1024,5))
        data_labels = np.zeros((n,64,1024))
        count = 0
        count_2 = 0
        for i in sequences["carla"]:
            for filename in os.scandir(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/binary_files'):
                if str(filename.name).endswith('.bin'):
                    scan = np.fromfile(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/binary_files/'+filename.name,dtype=np.float32)
                    scan = scan.reshape((64,1024,5))
                    if normalized:
                        scan_normalized = np.zeros(scan.shape)
                        scan_normalized[:,:,0] = MinMaxScaler().fit_transform(scan[:,:,0])
                        scan_normalized[:,:,1] = MinMaxScaler().fit_transform(scan[:,:,1]) 
                        scan_normalized[:,:,2] = MinMaxScaler().fit_transform(scan[:,:,2])
                        scan_normalized[:,:,3] = MinMaxScaler().fit_transform(scan[:,:,3])
                        scan_normalized[:,:,4] = MinMaxScaler().fit_transform(scan[:,:,4])
                        data[count] = scan_normalized
                        count = count + 1
                    else:
                        data[count] = scan
                        count = count + 1
            for filename in os.scandir(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/updated_ground_truth'):
                if str(filename.name).endswith('.label'):
                    labels = np.fromfile(str(path)+'/carla/dataset/sequences/'+str(i)+'/lidar_semseg/raw_data/updated_ground_truth/'+filename.name,dtype = np.uint32)
                    labels = labels.reshape((64,1024))
                    data_labels[count_2] = labels
                    count_2 += 1 
    if shuffle:
        indices_list = [int(i) for i in range(data.shape[0])]
        np.random.shuffle(indices_list)
        data = data[indices_list,:]
        data_labels = data_labels[indices_list,:]
    
    return (data,data_labels)
