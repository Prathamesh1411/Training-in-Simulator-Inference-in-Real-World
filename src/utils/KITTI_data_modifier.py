import numpy as np
import os
from PIL import Image

KITTI_LABEL_COLORS = np.array([
    (0, 0, 0),
    (0, 0, 142),
    (220, 20, 60)
])
# KITTI_LABEL_COLORS = {
#         0: [0, 0, 0],
#         1: [255, 0, 0],
#         10: [100, 150, 245],
#         11: [100, 230, 245],
#         13: [100, 80, 250],
#         15: [30, 60, 150],
#         16: [0, 0, 255],
#         18: [80, 30, 180],
#         20: [0, 0, 255],
#         30: [255, 30, 30],
#         31: [255, 40, 200],
#         32: [150, 30, 90],
#         40: [255, 0, 255],
#         44: [255, 150, 255],
#         48: [75, 0, 75],
#         49: [175, 0, 75],
#         50: [255, 200, 0],
#         51: [255, 120, 50],
#         52: [255, 150, 0],
#         60: [150, 255, 170],
#         70: [0, 175, 0],
#         71: [135, 60, 0],
#         72: [150, 240, 80],
#         80: [255, 240, 150],
#         81: [255, 0, 0],
#         99: [50, 255, 255],
#         252: [100, 150, 245],
#         256: [0, 0, 255],
#         253: [255, 40, 200],
#         254: [255, 30, 30],
#         255: [150, 30, 90],
#         257: [100, 80, 250],
#         258: [80, 30, 180],
#         259: [0, 0, 255]}

def do_projection(velo_path, proj_velo_path, label_path, proj_label_path, proj_labelimg_path):
    bin_num = 0
    label_num = 0
    # img_num = 0
    for file in sorted(os.listdir(velo_path)):
        H = 64
        W = 1024
        fov_up = 3.0
        fov_down = -25.0
        try:
            if file.endswith(".bin"):
                original_bin = np.fromfile(os.path.join(velo_path, file), np.float32)
                original_bin = original_bin.reshape((-1, 4))
                point_cloud_xyz = original_bin[:, 0:3]
                fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
                fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
                fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
                proj_W = W
                proj_H = H

                #range information
                depth = np.linalg.norm(point_cloud_xyz, 2, axis=1)

                ####Setting up the shape containers
                proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
                unproj_range = np.zeros((0, 1), dtype=np.float32)
                proj_xyz = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
                proj_idx = np.full((proj_H, proj_W), -1, dtype=np.uint32)
                proj_sem_label = np.full((proj_H, proj_W), 0, dtype=np.uint32)  #Changed from -1
                # proj_sem_color = np.full((proj_H, proj_W, 3), 0, dtype=np.float32)
                proj_remission = np.full((proj_H, proj_W), -1, dtype=np.float32)

                # Extracting each column of lidar data for x, y, z, remission and labels
                scan_x = point_cloud_xyz[:, 0]
                scan_y = point_cloud_xyz[:, 1]
                scan_z = point_cloud_xyz[:, 2]
                remission = original_bin[:, 3]
                semantic_label = get_semantic_label(label_path, label_num)
                # sem_label_color = color_map[semantic_label]

                # get angles of all points
                yaw = -np.arctan2(scan_y, scan_x)
                pitch = np.arcsin(scan_z / depth)

                # get projections in image coords
                proj_x = 0.5 * ((yaw / np.pi) + 1.0)  # in [0.0, 1.0] "v"
                proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0] "u"

                # scale to image size using angular resolution
                proj_x *= proj_W  # in [0.0, W]
                proj_y *= proj_H  # in [0.0, H]

                # round and clamp for use as index
                # v
                proj_x = np.floor(proj_x)
                proj_x = np.minimum(proj_W - 1, proj_x)
                proj_x = np.maximum(0, proj_x).astype(np.uint32)  # in [0,W-1]
                proj_x = np.copy(proj_x)  # store a copy in orig order
                # print(np.shape(np.unique(proj_x)))

                # u
                proj_y = np.floor(proj_y)
                proj_y = np.minimum(proj_H - 1, proj_y)
                proj_y = np.maximum(0, proj_y).astype(np.uint32)  # in [0,H-1]
                proj_y = np.copy(proj_y)  # store a copy in original order
                # print(np.unique(proj_y))

                # copy of depth in original order
                unproj_range = np.copy(depth)
                indices = np.arange(depth.shape[0])

                # order in decreasing depth
                # order = np.argsort(depth)[::-1]
                # depth = depth[order]
                # indices = indices[order]
                # point_cloud_xyz = point_cloud_xyz[order]
                # remission = remission[order]
                # proj_y = proj_y[order]
                # proj_x = proj_x[order]

                proj_range[proj_y, proj_x] = depth
                proj_xyz[proj_y, proj_x] = point_cloud_xyz
                proj_idx[proj_y, proj_x] = indices
                proj_sem_label[proj_y, proj_x] = semantic_label
                # proj_sem_color[proj_y, proj_x] = sem_label_color
                proj_remission[proj_y, proj_x] = remission

                # creating input tensor
                create_KITTI_input_tensor(proj_velo_path, bin_num, proj_xyz, proj_remission, proj_range)
                proj_label = modify_semantic_label(proj_label_path, proj_sem_label, label_num)
                create_proj_image(proj_labelimg_path, proj_label, label_num)
                # create_proj_image(proj_labelimg_path, proj_sem_label, label_num)

                bin_num += 1
                label_num += 1


        except Exception as e:
            raise e

def create_KITTI_input_tensor(proj_path, bin_num, proj_xyz, proj_intensity, proj_range):
    if not os.path.exists(proj_path):
        os.makedirs(proj_path)
    proj_filename = "new_%06d.bin" % bin_num
    filepath_second = os.path.join(proj_path, proj_filename)
    print(proj_filename)
    file = open(os.path.join(proj_path, proj_filename), "w")
    proj_intensity = np.expand_dims(proj_intensity, axis=2)
    proj_range = np.expand_dims(proj_range, axis=2)
    input_tensor = np.concatenate((proj_xyz, proj_intensity, proj_range), axis=2)
    # print(np.shape(input_tensor))
    input_tensor.tofile(file)
    file.close()

def get_semantic_label(path, label_num):
    filename = "%06d.label" % label_num
    semantic_label = np.fromfile(os.path.join(path, filename), np.uint32)
    semantic_label = semantic_label & 0xFFFF
    return semantic_label


def modify_semantic_label(proj_path, proj_label, label_num):
    vehicle = [10, 11, 13, 15, 18, 20, 252, 257, 258, 259]
    pedestrian = [30, 31, 32, 253, 254, 255]
    if not os.path.exists(proj_path):
        os.makedirs(proj_path)
    proj_filename = "new_%06d.label" % label_num
    filepath_second = os.path.join(proj_path, proj_filename)
    print(proj_filename)
    proj_label = proj_label.reshape(-1)
    for i in range(len(proj_label)):
        if proj_label[i] & 0xFFFF in vehicle:
            proj_label[i] = 1

        elif proj_label[i] & 0xFFFF in pedestrian:
            proj_label[i] = 2

        else:
            proj_label[i] = 0

    file = open(os.path.join(proj_path, proj_filename), "w")
    # proj_label = proj_label.reshape(64, 1024)
    proj_label.tofile(file)
    # file.close

    return proj_label


def create_proj_image(projimg_path, proj_label, label_num):
    if not os.path.exists(projimg_path):
        os.makedirs(projimg_path)
    # dtype = np.uint32
    # label = np.fromfile("new_000003.label", dtype)
    pixel_color = KITTI_LABEL_COLORS[proj_label]
    pixel_color_image = np.reshape(pixel_color, (64, 1024, 3))

    # # Save the image using Pillow module.
    image = (np.asarray(pixel_color_image)).astype(np.uint8)
    image = Image.fromarray(image)
    projimg_filename = "%06d.png" % label_num
    image_path = os.path.join(projimg_path, projimg_filename)
    image.save(image_path)

# def create_proj_image(projimg_path, proj_label, label_num):
#     # proj_label = np.reshape(proj_label, (65536, 1))
#     if not os.path.exists(projimg_path):
#         os.makedirs(projimg_path)
#     proj_label = proj_label.reshape(-1)
#     # print(len(proj_label))
#     proj_label = proj_label.astype(np.uint32)
#     sem_label_color = []
#     for i in range(len(proj_label)):
#         sem_label_color.append(KITTI_LABEL_COLORS[int(proj_label[i] & 0xFFFF)])
#
#     sem_label_color = np.reshape(sem_label_color, (64, 1024, 3))
#     image = (np.asarray(sem_label_color)).astype(np.uint8)
#     image = Image.fromarray(image)
#     projimg_filename = "%06d.png" % label_num
#     image_path = os.path.join(projimg_path, projimg_filename)
#     image.save(image_path)



if __name__ == "__main__":
    velo_path = "/home/prathamesh/Downloads/data_odometry_velodyne/dataset/sequences/08/velodyne/"
    label_path = "/home/prathamesh/Downloads/data_odometry_labels/dataset/sequences/08/labels/"
    proj_velo_path = "/home/prathamesh/Downloads/data_odometry_velodyne_proj/dataset/sequences/08/velodyne/"
    proj_label_path = "/home/prathamesh/Downloads/data_odometry_labels_proj/dataset/sequences/08/labels/"
    proj_labelimage_path = "/home/prathamesh/Downloads/data_odometry_labels_projimg/dataset/sequences/08/labels/"
    do_projection(velo_path, proj_velo_path, label_path, proj_label_path, proj_labelimage_path)