#!/bin/bash

# TỰ ĐỘNG NHẢY RA THƯ MỤC GỐC (Chứa file run_longExp.py và folder dataset)
cd "$(dirname "$0")/.." || exit 1

# Tạo mới hoặc xóa file cũ để ghi log mới
> FRLinear_output/ETTm2_1.txt
> FRLinear_output/ETTm2_2.txt
> FRLinear_output/ETTm2_3.txt
> FRLinear_output/ETTm2_4.txt

echo "=== START RUNNING ETTm2_336_96 ===" >> FRLinear_output/ETTm2_1.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_96_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> FRLinear_output/ETTm2_1.txt 2>&1

echo "=== START RUNNING ETTm2_336_192 ===" >> FRLinear_output/ETTm2_2.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_192_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 192 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> FRLinear_output/ETTm2_2.txt 2>&1

echo "=== START RUNNING ETTm2_336_336 ===" >> FRLinear_output/ETTm2_3.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_336_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 336 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> FRLinear_output/ETTm2_3.txt 2>&1

echo "=== START RUNNING ETTm2_336_720 ===" >> FRLinear_output/ETTm2_4.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_720_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 720 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> FRLinear_output/ETTm2_4.txt 2>&1

echo "=== ALL RUNS COMPLETED ===" >> FRLinear_output/ETTm2.txt
