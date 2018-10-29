#https://blog.csdn.net/w5688414/article/details/78395177

#cd checkpoints 
#unzip ssd_300_vgg.ckpt.zip

#================create Dataset===================#
DATASET_DIR='/lustre/dataset/VOC/VOCdevkit/VOC2007/'
#python tf_convert_data.py \
#    --dataset_name=pascalvoc \
#    --dataset_dir=/lustre/dataset/VOC/VOCdevkit/VOC2007/ \
#    --output_name=voc_2007_test \
#    --output_dir=./tfrecords/
#OUTPUT_DIR='./tfrecords'
#python tf_convert_data.py \
#    --dataset_name=pascalvoc \
#    --dataset_dir=${DATASET_DIR} \
#    --output_name=voc_2007_train \
#    --output_dir=${OUTPUT_DIR}

#DATASET_DIR=/lustre/dataset/VOC/VOCdevkit/VOC2007/
#OUTPUT_DIR=./tfrecords/
#python tf_convert_data.py \
#    --dataset_name=pascalvoc \
#    --dataset_dir=${DATASET_DIR} \
#    --output_name=voc_2007_train \
#    --output_dir=${OUTPUT_DIR}


#================finetune SSD model===================#
#DATASET_DIR=./tfrecords/
DATASET_NAME='coco'
DATASET_SPLIT_NAME='train'
DATASET_DIR='/home/huiwu1/dataset/coco-text/tfrecords/train/'
TRAIN_DIR='logs/'
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
#CHECKPOINT_PATH=./logs/model.ckpt-856
#CHECKPOINT_PATH=./logs/model.ckpt-4721
#CHECKPOINT_PATH=./logs/model.ckpt-4786

python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME}\
    --dataset_split_name=${DATASET_SPLIT_NAME} \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=16

#EVAL_DIR=${TRAIN_DIR}/eval
#DATASET_DIR='/home/huiwu1/dataset/coco-text/tfrecords/val/'
#python eval_ssd_network.py \
#    --eval_dir=${EVAL_DIR} \
#    --dataset_dir=${DATASET_DIR} \
#    --dataset_name=${DATASET_NAME} \
#    --dataset_split_name=${DATASET_SPLIT_NAME} \
#    --model_name=ssd_300_vgg \
#    --checkpoint_path=${TRAIN_DIR} \
#    --wait_for_checkpoints=True \
#    --batch_size=1 \
#    --max_num_batches=500

:<<!
#================build new SSD model===================#
DATASET_DIR=./tfrecords
TRAIN_DIR=./log/
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32

DATASET_DIR=./tfrecords
TRAIN_DIR=./log_finetune/
CHECKPOINT_PATH=./log/model.ckpt-N
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.00001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
!

