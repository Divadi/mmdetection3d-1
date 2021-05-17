import copy
import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp

from numpy import linalg as LA

from mmdet.datasets import DATASETS
from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .custom_3d import Custom3DDataset
from .kitti_dataset import KittiDataset

@DATASETS.register_module()
class AIODriveDataset(KittiDataset):
    """AIODrive Dataset.
    """
    CLASSES = ('Car', 'Pedestrian', 'Cyclist', 'Motorcycle')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='lidar_velodyne',
                 pipeline=None,
                 classes=('Car', 'Pedestrian', 'Cyclist', 'Motorcycle'),
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

    def _get_pts_filename(self, seq_id, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                seq_id, f'{idx}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        seq_id = info['image']['seq_id']
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(seq_id, sample_idx)
        input_dict = dict(
            sample_idx=f'{seq_id}_{sample_idx}', # uncertain if this breaks anything
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_boxes_with_no_points(annos)
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    '''
    Many GT boxes in info file has no points inside.
    Remove those according to num_points_in_gt of corresponding self.pts_prefix
    Note that this modifies in-place the self.data_infos dict that's kept throughout training.
    '''
    def remove_boxes_with_no_points(self, annos):
        if isinstance(annos['num_points_in_gt'], dict): # has not been filtered before
            pts_type = self.pts_prefix.split("_")[-1]
            # pts_type = "velodyne"
            boxes_with_points = annos['num_points_in_gt'][pts_type] != 0

            annos['num_points_in_gt'] = annos['num_points_in_gt'][pts_type]
            for key in annos.keys():
                annos[key] = annos[key][boxes_with_points]
        else: # actually, this is not necessary, but just do for completeness
            boxes_with_points = annos['num_points_in_gt'] != 0
            for key in annos.keys():
                annos[key] = annos[key][boxes_with_points]

        return annos

    '''
    Our evaluation metric relies on distance to DT and GT boxes.
    Input just needs to be a list of dicts with "location" key.
    '''
    def add_distances(self, annos):
        for anno in annos:
            # print(anno)
            anno['distance'] = LA.norm(anno['location'], axis=1)
        return annos

    # TODO: Work in progress
    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import aiodrive_eval
        gt_annos = [info['annos'] for info in self.data_infos]
        # making sure num_points_in_gt only has the point cloud we care about
        gt_annos = [self.remove_boxes_with_no_points(annos) for annos in gt_annos] 
        # Now, add distance calculations
        gt_annos = self.add_distances(gt_annos)
        if isinstance(result_files, dict):
            for _, result_files_ in result_files.items():
                result_files_ = self.add_distances(result_files_)
        else:
            result_files = self.add_distances(result_files)


        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                eval_types = ['bev', '3d']
                if 'img' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = aiodrive_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            if metric == 'img_bbox':
                ap_result_str, ap_dict = aiodrive_eval(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                ap_result_str, ap_dict = aiodrive_eval(gt_annos, result_files,
                                                    self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show:
            self.show(results, out_dir)
        return ap_dict