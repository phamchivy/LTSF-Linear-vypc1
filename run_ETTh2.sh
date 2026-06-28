#!/bin/bash

# Tạo mới hoặc xóa file log cũ
> ETTh2_1.txt
> ETTh2_2.txt
> ETTh2_3.txt
> ETTh2_4.txt

# echo "=== START RUNNING ETTh2_336_96 ==="
# echo "=== START RUNNING ETTh2_336_96 ===" >> ETTh2_1.txt
# python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv --model_id ETTh2_336_96_1 --model FRLinear --data ETTh2 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh2_1.txt 2>&1

echo "=== START RUNNING ETTh2_336_192 ==="
echo "=== START RUNNING ETTh2_336_192 ===" >> ETTh2_2.txt
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv --model_id ETTh2_336_192_1 --model FRLinear --data ETTh2 --features M --seq_len 336 --pred_len 192 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh2_2.txt 2>&1

# echo "=== START RUNNING ETTh2_336_336 ==="
# echo "=== START RUNNING ETTh2_336_336 ===" >> ETTh2_3.txt
# python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv --model_id ETTh2_336_336_1 --model FRLinear --data ETTh2 --features M --seq_len 336 --pred_len 336 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh2_3.txt 2>&1

# echo "=== START RUNNING ETTh2_336_720 ==="
# echo "=== START RUNNING ETTh2_336_720 ===" >> ETTh2_4.txt
# python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv --model_id ETTh2_336_720_1 --model FRLinear --data ETTh2 --features M --seq_len 336 --pred_len 720 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 5 --train_epochs 20 >> ETTh2_4.txt 2>&1

echo "=== ALL ETTh2 RUNS COMPLETED ==="
