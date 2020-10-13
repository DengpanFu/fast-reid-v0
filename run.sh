#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
    --config-file ./configs/CMDM/mgn_R50.yml --num-gpus 4 \
    MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/imgnet_sup_resnet50.pth" \
    MODEL.PIXEL_MEAN [89.89,79.20,80.07] \
    DATASETS.KWARGS 'data_name:duke+split_mode:id+split_ratio:1.0' \
    OUTPUT_DIR "logs/CMDM/duke_R50_imgsup" # && \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
#     --config-file ./configs/Market1501/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/imgnet_sup_resnet50.pth" \
#     OUTPUT_DIR "logs/Market1501/bagtricks_R50_imgsup" && \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
#     --config-file ./configs/MSMT17/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/imgnet_sup_resnet50.pth" \
#     OUTPUT_DIR "logs/MSMT17/bagtricks_R50_imgsup" && \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
#     --config-file ./configs/MSMT17/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/resnet50_n02_AugTemp_200ep.pth" \
#     OUTPUT_DIR "logs/MSMT17/bagtricks_R50_n02_AugTemp_200ep"


# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
#     --config-file ./configs/DukeMTMC/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/resnet50_n02_200ep.pth" \
#     OUTPUT_DIR "logs/DukeMTMC/bagtricks_R50_n02_200ep" && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
#     --config-file ./configs/Market1501/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/resnet50_n02_200ep.pth" \
#     OUTPUT_DIR "logs/Market1501/bagtricks_R50_n02_200ep" && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
#     --config-file ./configs/MSMT17/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/resnet50_n02_200ep.pth" \
#     OUTPUT_DIR "logs/MSMT17/bagtricks_R50_n02_200ep" && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
#     --config-file ./configs/DukeMTMC/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/resnet50_n02_AugTemp_200ep.pth" \
#     OUTPUT_DIR "logs/DukeMTMC/bagtricks_R50_n02_AugTemp_200ep" && \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
#     --config-file ./configs/Market1501/bagtricks_R50.yml --num-gpus 4 \
#     MODEL.BACKBONE.PRETRAIN_PATH "pre_models/models/resnet50_n02_AugTemp_200ep.pth" \
#     OUTPUT_DIR "logs/Market1501/bagtricks_R50_n02_AugTemp_200ep"
