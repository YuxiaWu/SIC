#!/usr/bin bash

# CUDA_VISIBLE_DEVICES=3 bash run_script_slot/run_ours.sh


for dataset in 'camrest'  'woz-attr' 'carslu' 'woz-hotel' 'atis'
do
    for known_cls_ratio in 0.75
    do
        for seed in 0 1 2 3 4 5 6 7 8 9
        do 
            python run.py \
            --dataset $dataset \
            --method 'ours' \
            --setting 'semi_supervised' \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --backbone 'bert' \
            --config_file_name 'ours' \
            --gpu_id '0' \
            --train \
            --pre_train \
            --thr 0.9 \
            --save_results \
            --save_model \
            --results_file_name 'results_ours.csv' 
        done
    done
done


#--pre_train \
#--predict_k \