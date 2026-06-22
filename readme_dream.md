python -m dream_research.scripts.run_decomposition \
--is_training 1 \
--root_path ./dataset/ \
--data_path ETTh1.csv \
--model_id ETTh1_336_96_P1_gaussian \
--model DLinear \
--data ETTh1 \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 7 \
--batch_size 32 \
--learning_rate 0.005 \
--des Exp \
--itr 1 \
--decomposition gaussian

python run_longExp.py \
--is_training 1 \
--root_path ./dataset/ \
--data_path ETTh1.csv \
--model_id ETTh1_336_96 \
--model DLinear \
--data ETTh1 \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 7 \
--batch_size 32 \
--learning_rate 0.005 \
--des Exp \
--itr 1

python run_longExp.py \
--is_training 1 \
--root_path ./dataset/ \
--data_path ETTh1.csv \
--model_id ETTh1_336_96 \
--model DREAM \
--data ETTh1 \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 7 \
--batch_size 32 \
--learning_rate 0.005 \
--des Exp \
--itr 1 \
--decomposition gaussian

---

Benchmark 22/06/2026

python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96_1 --model DREAM --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 1 --decomposition ma

python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model DLinear --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 1

python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96_1 --model DREAM --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 1 --decomposition fft

python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96_1 --model DREAM --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --batch_size 32 --learning_rate 0.005 --des Exp --itr 1 --decomposition gaussian

