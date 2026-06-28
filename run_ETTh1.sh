#!/bin/bash

# Tạo mới hoặc xóa file log cũ
> ETTh1_1.txt
> ETTh1_2.txt
> ETTh1_3.txt
> ETTh1_4.txt

# echo "=== START RUNNING ETTh1_336_96 ==="
# echo "=== START RUNNING ETTh1_336_96 ===" >> ETTh1_1.txt
# python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96_1 --model FRLinear --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh1_1.txt 2>&1

echo "=== START RUNNING ETTh1_336_192 ==="
echo "=== START RUNNING ETTh1_336_192 ===" >> ETTh1_2.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_192_1 --model FRLinear --data ETTh1 --features M --seq_len 336 --pred_len 192 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh1_2.txt 2>&1

echo "=== START RUNNING ETTh1_336_336 ==="
echo "=== START RUNNING ETTh1_336_336 ===" >> ETTh1_3.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_336_1 --model FRLinear --data ETTh1 --features M --seq_len 336 --pred_len 336 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh1_3.txt 2>&1

echo "=== START RUNNING ETTh1_336_720 ==="
echo "=== START RUNNING ETTh1_336_720 ===" >> ETTh1_4.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_720_1 --model FRLinear --data ETTh1 --features M --seq_len 336 --pred_len 720 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh1_4.txt 2>&1

echo "=== ALL ETTh1 RUNS COMPLETED ==="
