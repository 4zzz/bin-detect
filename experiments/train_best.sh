#!/bin/bash

set -x

epochs=1000
width=512
height=384
bb=resnet34

train_dataset="datasets/VISIGRAPP_TRAIN/dataset.json"
test_dataset="datasets/VISIGRAPP_TEST/dataset.json"
test_dir="$(dirname "$test_dataset")"

outdir="$1"

train() {
    test_name="$1"
    shift
    if [ -e "${test_name}.pth" ] ; then
        echo "Skipping $test_name"
    else
        python train.py -e $epochs -bb $bb -ti "$outdir/${test_name}_training.json" -iw ${width} -ih ${height} -smw "$outdir/${test_name}.pth" $@ "$train_dataset" 2>&1 | tee "$outdir/${test_name}_trainlog.log"
    fi
}


test_name="best_cutout_crop_noise"
train "$test_name" -ts 2 -crp 0.8 -crmax 0.1 -cp 0.8 -cmaxs 0.4
test_name="best_cutout_noise"
train "$test_name" -ts 2 -crp 0.8 -crmax 0.1
