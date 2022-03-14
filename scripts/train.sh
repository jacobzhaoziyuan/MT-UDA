#!/usr/bin/bash
cd ..
SOURCE_PATH=data/mr
TARGET_PATH=data/ct
VAL_PATH=data/mr/mr_like_test/fake_mr
VAL_SOURCE_PATH=data/mr/mr_like_test/org_mr
TEST_PATH=data/mr/mr_like_test_npz/fake_mr
LABELED_ORG_SOURCE_PATH=data/mr/mr_labeled/datalist/org_4labeled.txt
LABELED_CYC_SOURCE_PATH=data/mr/mr_labeled/datalist/cyc_4labeled.txt


DIR='exp/mtuda_mr2ct'
EPOCH=2
BATCH_SIZE=16
NUM_CLASS=5
# warm up epochs for adam warm up with teacher training
WARM_EPOCH=30
# pretrain student only
ITER_PRE=30
LR=0.005
gpu=0
# constant lr after warm up, always using Adam
LR_MODE=None
OPTIMIZER=Adam
SEED=42
python main.py --gpu $gpu --lr_mode $LR_MODE --sup_aug --tea_aug\
                --out $DIR --lr $LR --batch-size $BATCH_SIZE --optimizer $OPTIMIZER\
               --iter-pretrain $ITER_PRE \
               --consistency 0.1 --consistency_rampup $EPOCH  --consistency_type mel \
               --kd 1 --kd_rampup $EPOCH \
               --epochs $EPOCH --num-class $NUM_CLASS -Tmax $EPOCH --warmup_epochs $WARM_EPOCH \
               --source-path $SOURCE_PATH --target-path $TARGET_PATH --supervised_org_s_dir $LABELED_ORG_SOURCE_PATH --supervised_cyc_s_dir $LABELED_CYC_SOURCE_PATH  \
               --val-path $VAL_PATH --val-source-path $VAL_SOURCE_PATH --seed $SEED --ema-decay 0.99