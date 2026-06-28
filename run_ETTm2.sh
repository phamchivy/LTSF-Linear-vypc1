#!/bin/bash

# Tạo mới hoặc xóa file cũ để ghi log mới
> ETTm2.txt

echo "=== START RUNNING ETTm2_336_96 ===" >> ETTm2.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_96_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm2.txt 2>&1

echo "=== START RUNNING ETTm2_336_192 ===" >> ETTm2.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_96_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 192 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm2.txt 2>&1

echo "=== START RUNNING ETTm2_336_336 ===" >> ETTm2.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_96_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 336 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm2.txt 2>&1

echo "=== START RUNNING ETTm2_336_720 ===" >> ETTm2.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_336_96_1 --model FRLinear --data ETTm2 --features M --seq_len 336 --pred_len 720 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTm2.txt 2>&1

echo "=== ALL RUNS COMPLETED ===" >> ETTm2.txt
