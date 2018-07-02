#cd checkpoints 
#unzip ssd_300_vgg.ckpt.zip

python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=/lustre/dataset/VOC/VOCdevkit/VOC2007/ \
    --output_name=voc_2007_test \
    --output_dir=./tfrecords/

#CHECKPOINT_PATH = ./checkpoints/ssd_300_vgg.ckpt.data-00000-of-00001
python eval_ssd_network.py \
    --eval_dir=./logs/ \
    --dataset_dir=./tfrecords/ \
    --dataset_name=pascalvoc_2007 \
    --model_name=ssd_300_vgg \
    --checkoutpoint_path=./checkpoints/ \
    --batch_size=1


