import mmcv
import numpy as np
import os.path as osp
from collections import OrderedDict
from pathlib import Path

from mmdet3d.core.bbox import box_np_ops
from .kitti_data_utils import get_kitti_image_info, get_waymo_image_info

from .aiodrive_data_utils import get_split, get_aiodrive_info, get_seq_idx_pairs

aiodrive_categories = ('Car', 'Pedestrian', 'Cyclist', 'Motorcycle', 'Undefined')


def create_aiodrive_info_file(data_path,
                              pkl_prefix='aiodrive',
                              save_path=None,
                              relative_path=True):
    """Create info file of AIODrive dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    splits = get_split()

    print('Generating info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    # Save Train
    seq_idx_pairs = get_seq_idx_pairs(osp.join(data_path, "trainval/calib"), splits['train'])
    aiodrive_infos_train = get_aiodrive_info(
        data_path,
        training=True,
        seq_idx_pairs=seq_idx_pairs,
        relative_path=relative_path
    )
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'AIODrive info train file is saved to {filename}')
    mmcv.dump(aiodrive_infos_train, filename)

    # Save Val
    seq_idx_pairs = get_seq_idx_pairs(osp.join(data_path, "trainval/calib"), splits['val'])
    aiodrive_infos_val = get_aiodrive_info(
        data_path,
        training=True,
        seq_idx_pairs=seq_idx_pairs,
        relative_path=relative_path
    )
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'AIODrive info val file is saved to {filename}')
    mmcv.dump(aiodrive_infos_val, filename)

    # Save TrainVal
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'AIODrive info trainval file is saved to {filename}')
    mmcv.dump(aiodrive_infos_train + aiodrive_infos_val, filename)

    # Save Test
    seq_idx_pairs = get_seq_idx_pairs(osp.join(data_path, "test/calib"), splits['test'])
    aiodrive_infos_test = get_aiodrive_info(
        data_path,
        training=False,
        label_info=False,
        seq_idx_pairs=seq_idx_pairs,
        relative_path=relative_path
    )
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'AIODrive info test file is saved to {filename}')
    mmcv.dump(aiodrive_infos_test, filename)