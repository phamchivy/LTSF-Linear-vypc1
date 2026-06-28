#!/bin/bash

# Tạo mới hoặc xóa file cũ để ghi log mới
> ETTm1.txt

echo "=== START RUNNING ETTm1_336_96 ===" >> ETTm1.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv --model_id ETTm1_336_96_1 --model FRLinear --data ETTm1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm1.txt 2>&1

echo "=== START RUNNING ETTm1_336_192 ===" >> ETTm1.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv --model_id ETTm1_336_192_1 --model FRLinear --data ETTm1 --features M --seq_len 336 --pred_len 192 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm1.txt 2>&1

echo "=== START RUNNING ETTm1_336_336 ===" >> ETTm1.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv --model_id ETTm1_336_336_1 --model FRLinear --data ETTm1 --features M --seq_len 336 --pred_len 336 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm1.txt 2>&1

echo "=== START RUNNING ETTm1_336_720 ===" >> ETTm1.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv --model_id ETTm1_336_720_1 --model FRLinear --data ETTm1 --features M --seq_len 336 --pred_len 720 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm1.txt 2>&1

echo "=== ALL RUNS COMPLETED ===" >> ETTm1.txt
