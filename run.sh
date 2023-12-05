DATASET_NAME="MORE"
BERT_NAME='bert-base-uncased'
VIT_NAME='clip-vit-base-patch32'

CUDA_VISIBLE_DEVICES=1  python run.py \
        --dataset_name=${DATASET_NAME} \
        --vit_name=${VIT_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=20 \
        --batch_size=16 \
        --lr=1e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --use_dep \
        --use_box \
        --use_cap \
        --max_seq=96 \
        --save_path="ckpt"
