#!/bin/bash

set -x

epochs=1000
width=512
height=384
bb=resnet34

train_dataset="datasets/VISIGRAPP_TRAIN/dataset.json"
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

for sigma in 1 2 4 8 16 32 ; do
    test_name="noise_sigma_$sigma"
    train "$test_name" -ts $sigma
done

for crp in 0.2 0.4 0.6 0.8 1 ; do
    for crmaxs in 0.1 0.2 0.3 0.4 0.6 0.8 1 ; do
        test_name="crop_crp_${crp}_crmaxs_${crmaxs}"
        train "$test_name" -crp $crp -crmax $crmaxs
    done
done

for cp in 0.2 0.4 0.6 0.8 1 ; do
    for cmaxs in 0.3 0.4 0.6 0.8 1 ; do
        test_name="cutout_cp_${cp}_cmaxs_${cmaxs}"
        train "$test_name" -cp $cp -cmaxs $cmaxs
    done
done
