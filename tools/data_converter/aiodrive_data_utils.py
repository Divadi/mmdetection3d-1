import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
import os
from os import path as osp
from pathlib import Path
from skimage import io

from tqdm import tqdm

from .kitti_data_utils import _extend_matrix, add_difficulty_to_annos

# TODO: Make this an import, if AIODrive dev becomes like a pip package
def get_split():
	# Total: 40
	train = ['Town01_seq0000', 'Town01_seq0001', 'Town01_seq0002', 'Town01_seq0003', 'Town01_seq0004', 'Town01_seq0005', \
			 'Town01_seq0006', 'Town01_seq0007', 'Town01_seq0008', 'Town01_seq0009', 'Town01_seq0014', 'Town02_seq0000', \
			 'Town02_seq0001', 'Town02_seq0002', 'Town02_seq0003', 'Town02_seq0004', 'Town02_seq0005', 'Town02_seq0006', \
			 'Town02_seq0007', 'Town02_seq0009', 'Town02_seq0016', 'Town02_seq0019', 'Town03_seq0000', 'Town03_seq0001', \
			 'Town03_seq0002', 'Town03_seq0003', 'Town03_seq0004', 'Town03_seq0005', 'Town03_seq0006', 'Town03_seq0007', \
			 'Town03_seq0008', 'Town03_seq0009', 'Town03_seq0015', 'Town03_seq0100', 'Town04_seq0000', 'Town04_seq0001', \
			 'Town04_seq0002', 'Town04_seq0003', 'Town04_seq0004', 'Town04_seq0005']

	# Total: 30
	val   = ['Town04_seq0006', 'Town04_seq0007', 'Town04_seq0008', 'Town04_seq0009', 'Town04_seq0014', 'Town04_seq0100', \
			 'Town05_seq0000', 'Town05_seq0001', 'Town05_seq0002', 'Town05_seq0003', 'Town05_seq0004', 'Town05_seq0005', \
			 'Town05_seq0006', 'Town05_seq0007', 'Town05_seq0008', 'Town05_seq0009', 'Town05_seq0014', 'Town05_seq0100', \
			 'Town06_seq0000', 'Town06_seq0001', 'Town06_seq0002', 'Town06_seq0003', 'Town06_seq0004', 'Town06_seq0005', \
			 'Town06_seq0006', 'Town06_seq0007', 'Town06_seq0008', 'Town06_seq0009', 'Town06_seq0015', 'Town06_seq0100']

	# Total: 30
	test  = ['Town07_seq0000', 'Town07_seq0001', 'Town07_seq0002', 'Town07_seq0003', 'Town07_seq0004', 'Town07_seq0005', \
			 'Town07_seq0006', 'Town07_seq0009', 'Town07_seq0010', 'Town07_seq0011', 'Town07_seq0100', 'Town07_seq0102', \
			 'Town07_seq0103', 'Town10HD_seq0002', 'Town10HD_seq0003', 'Town10HD_seq0007', 'Town10HD_seq0009', \
			 'Town10HD_seq0010', 'Town10HD_seq0011', 'Town10HD_seq0012', 'Town10HD_seq0013', 'Town10HD_seq0014', \
			 'Town10HD_seq0016', 'Town10HD_seq0017', 'Town10HD_seq0100', 'Town10HD_seq0101', 'Town10HD_seq0104', \
			 'Town10HD_seq0105', 'Town10HD_seq0106', 'Town10HD_seq0108']

	return {'train': train, 'val': val, 'test': test}

# def get_split():
# 	# Total: 40
# 	train = ['Town01_seq0000']

# 	# Total: 30
# 	val   = ['Town01_seq0000']

# 	# Total: 30
# 	test  = ['Town07_seq0000']

# 	return {'train': train, 'val': val, 'test': test}


def get_seq_idx_pairs(path, seq_ids):
    res = []
    for seq_id in seq_ids:
        for idx in sorted(os.listdir(Path(path) / seq_id)):
            res.append((seq_id, idx.split(".")[0]))

    return res

def get_aiodrive_info_path(seq_id,
                           idx,
                           prefix,
                           info_type="image_2",
                           file_tail=".png",
                           training=True,
                           relative_path=True,
                           exist_check=True):
    info_type = "{}/{}".format(info_type, seq_id)
    img_idx_str = idx
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('trainval') / info_type / img_idx_str
    else:
        file_path = Path('test') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_image_path(seq_id,
                   idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2'):
    return get_aiodrive_info_path(seq_id, idx, prefix, info_type, '.png', training,
                                  relative_path, exist_check)

def get_label_path(seq_id,
                   idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2'):
    return get_aiodrive_info_path(seq_id, idx, prefix, info_type, '.txt', training,
                                  relative_path, exist_check)

def get_calib_path(seq_id,
                   idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_aiodrive_info_path(seq_id, idx, prefix, 'calib', '.txt', training,
                                  relative_path, exist_check)


'''
Note: There is no great way to represent truncated, occluded, and bbox because
they are w.r.t different images. For now, setting default values for them.
Also, "alpha" is not available.
'''
def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'num_points_in_gt': {
            'velodyne': [],
            'denselv1': [],
            'denselv2': []
        }
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]

    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])

    annotations['truncated'] = np.array([0 for x in content]) # Filler
    annotations['occluded'] = np.array([0 for x in content]) # Filler
    annotations['alpha'] = np.array([-10 for x in content]) # Filler
    annotations['bbox'] = np.array([[-10, -10, -10, -10] # Filler
                                    for x in content]).reshape(-1, 4)

    annotations['num_points_in_gt']['velodyne'] = \
                            np.array([int(x[3]) for x in content], dtype=np.int32)
    annotations['num_points_in_gt']['denselv1'] = \
                            np.array([int(x[4]) for x in content], dtype=np.int32)
    annotations['num_points_in_gt']['denselv2'] = \
                            np.array([int(x[5]) for x in content], dtype=np.int32)

    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[15:18]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[18:21]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[21])
                                          for x in content]).reshape(-1)

    if len(content) != 0 and len(content[0]) == 27:  # have score
        annotations['score'] = np.array([float(x[26]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))

    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def get_aiodrive_info(path,
                      training=True,
                      label_info=True,
                      seq_idx_pairs=[],
                      extend_matrix=True,
                      num_worker=8,
                      relative_path=True,
                      imageshape=(720, 1920)):
    """
    Following KITTI annotation format version 2, with some changes.
    Label files for aiodrive includes num points in gt, so will load them in this function.
    TODO: Currently does not handle multiple image paths (because multiple images).
    TODO: Support sweeps
    TODO: Support depth sensor (currently not supported because need to generate point cloud from depth images)
    {
        image: {
            seq_id: ...
            image_idx: ...
            image_path: ... # image_2
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
            denselv1_path: ...
            denselv2_path: ...
        }
        calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            Tr_imu_to_velo: ...
            Tr_cam_to_velo: ...
            P0: ...
            P1: ...
            P2: ...
            P3: ...
            P4: ...
            P5: ...
        }
        annos: {
            Reference get_label_anno
            [optional]difficulty: kitti difficulty
        }
        num_points_in_gt: {
            velodyne: ...
            denselv1: ...
            denselv2: ...
        }
    }
    """
    root_path = Path(path)
    assert isinstance(seq_idx_pairs, list)

    def map_func(seq_idx_pair):
        seq_id, idx = seq_idx_pair
        
        info = {}
        image_info = {'seq_id': seq_id, 'image_idx': idx, 'image_shape': imageshape}
        pc_info = {'num_features': 4}
        annotations = None
        calib_info = {}

        # Load image infos
        image_info['image_path'] = get_image_path(seq_id, idx, path, training,
                                                  relative_path)
        info['image'] = image_info

        # Load point paths
        pc_info['velodyne_path'] = get_aiodrive_info_path(seq_id, idx, path, "lidar_velodyne", 
                                                          ".bin", training, relative_path)
        pc_info['denselv1_path'] = get_aiodrive_info_path(seq_id, idx, path, "lidar_denselv1", 
                                                          ".bin", training, relative_path)
        pc_info['denselv2_path'] = get_aiodrive_info_path(seq_id, idx, path, "lidar_denselv2", 
                                                          ".bin", training, relative_path)
        info['point_cloud'] = pc_info
        
        # Load annotations
        if label_info:
            label_path = get_label_path(seq_id, idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
            info['annos'] = annotations
            add_difficulty_to_annos(info)
    
        # Load calib
        calib_path = get_calib_path(seq_id, idx, path, training, relative_path=False)
        calib_file_dict = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                k, v = line.split(":")
                calib_file_dict[k.strip()] = v.strip().split(" ")
        P0 = np.array([float(info) for info in calib_file_dict['P0']]).reshape(3, 4)
        P1 = np.array([float(info) for info in calib_file_dict['P1']]).reshape(3, 4)
        P2 = np.array([float(info) for info in calib_file_dict['P2']]).reshape(3, 4)
        P3 = np.array([float(info) for info in calib_file_dict['P3']]).reshape(3, 4)
        P4 = np.array([float(info) for info in calib_file_dict['P4']]).reshape(3, 4)
        P5 = np.array([float(info) for info in calib_file_dict['P5']]).reshape(3, 4)
        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)
            P4 = _extend_matrix(P4)
            P5 = _extend_matrix(P5)

        R0_rect = np.array([float(info) for info in calib_file_dict['R0_rect']]).reshape(3, 3)
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect
        
        Tr_velo_to_cam = np.array([float(info) for info in calib_file_dict['Tr_velo_to_p2']]).reshape(3, 4)
        Tr_imu_to_velo = np.array([float(info) for info in calib_file_dict['imu_to_world']]).reshape(3, 4)
        Tr_cam_to_velo = np.array([float(info) for info in calib_file_dict['p2_to_world']]).reshape(3, 4)
        if extend_matrix:
            Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            Tr_cam_to_velo = _extend_matrix(Tr_cam_to_velo)
    
        calib_info['P0'] = P0
        calib_info['P1'] = P1
        calib_info['P2'] = P2
        calib_info['P3'] = P3
        calib_info['P4'] = P4
        calib_info['P5'] = P5
        calib_info['R0_rect'] = rect_4x4
        calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
        calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
        calib_info['Tr_cam_to_velo'] = Tr_cam_to_velo
        info['calib'] = calib_info

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = list(tqdm(executor.map(map_func, seq_idx_pairs), total=len(seq_idx_pairs)))

    return list(image_infos)
