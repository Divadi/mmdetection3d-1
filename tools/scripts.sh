CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh \
    configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_aiodrive-3d-3class.py 2 \
    --gpus 2 \
    --deterministic \
    --autoscale-lr

CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_test.sh \
    outputs/04_17/day/0/0.py \
    outputs/04_17/day/0/epoch_30.pth 2 \
    --deterministic \
    --eval mAP

CUDA_VISIBLE_DEVICES=1,2,3 ./tools/dist_train.sh \
    configs/_custom/04_17/day/0_no_eval.py 3 \
    --gpus 3 \
    --deterministic \
    --autoscale-lr
numba.cuda.cudadrv.driver.CudaAPIError: [2] Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY   
ERROR:numba.cuda.cudadrv.driver:Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY     

CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh \
    configs/_custom/04_18/day/0.py 2 \
    --gpus 2 \
    --deterministic \
    --autoscale-lr

CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh \
    configs/_custom/04_18/day/0_no_eval.py 2 \
    --gpus 2 \
    --deterministic \
    --autoscale-lr

CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh \
    configs/_custom/04_18/day/0_no_eval.py 2 \
    --gpus 2 \
    --deterministic \
    --autoscale-lr

CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh \
    configs/_custom/04_18/day/0_no_eval.py 2 \
    --gpus 2 \
    --deterministic \
    --autoscale-lr


CUDA_VISIBLE_DEVICES=0,1 PORT=49889 ./tools/dist_train.sh \
    configs/_custom/04_19/overnight/0.py 2 \
    --gpus 2 \
    --deterministic \
    --autoscale-lr

CUDA_VISIBLE_DEVICES=0,1 PORT=49889 ./tools/dist_test.sh \
    configs/_custom/04_19/overnight/0.py \
    outputs/04_19/overnight/0/latest.pth 2 \
    --deterministic \
    --eval mAP

# configs/_custom/04_18/day/0_kitti.py

CUDA_VISIBLE_DEVICES=0,1 PORT=49889 ./tools/dist_test.sh \
    configs/_custom/04_18/day/0_changed_for_kitti_val.py \
    outputs/04_18/day/0/epoch_30.pth 2 \
    --deterministic \
    --eval mAP \
    # --cfg-options model.voxel_layer.max_voxels.1=80000




CUDA_VISIBLE_DEVICES=0,1 PORT=50001 ./tools/dist_train.sh \
    configs/_custom/04_21/overnight/1.py 2 \
    --gpus 2 \
    --deterministic \
    --autoscale-lr

CUDA_VISIBLE_DEVICES=0,1 PORT=49889 ./tools/dist_test.sh \
    configs/_custom/04_18/day/0_changed_for_kitti_val.py \
    outputs/04_18/day/0/epoch_30.pth 2 \
    --deterministic \
    --eval mAP

CUDA_VISIBLE_DEVICES=0,1 PORT=49889 ./tools/dist_test.sh \
    configs/_custom/04_18/day/0_changed_for_kitti_val.py \
    outputs/04_18/day/0/epoch_30.pth 2 \
    --deterministic \
    --eval mAP



CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=49889 ./tools/dist_test.sh \
    configs/_custom/04_18/day/0_full_eval_range.py \
    outputs/04_18/day/1/epoch_6.pth 4 \
    --deterministic \
    --eval mAP \
    --out outputs/04_18/day/1/val_full_eval_range_all_frames.pkl


CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=49889 ./tools/dist_test.sh \
    configs/_custom/04_18/day/0_full_eval_range.py \
    outputs/04_18/day/0/latest.pth 4 \
    --deterministic \
    --eval mAP 



CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=49889 ./tools/dist_test.sh \
    outputs/04_22/overnight/1/1.py \
    outputs/04_22/overnight/1/latest.pth 4 \
    --deterministic \
    --eval mAP 