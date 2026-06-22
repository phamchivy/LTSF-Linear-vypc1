# dream_research/scripts/train_phase2.py

import os

experiments = [
    ("ma", 2),
    ("fft", 2),
    ("fft", 4),
    ("fft", 8),
    ("wavelet", 2),
    ("wavelet", 4)
]

for decomp, rank in experiments:

    model_id = f"ETTh1_336_96_P2_{decomp}_r{rank}"

    cmd = f"""
python run_longExp.py \
--is_training 1 \
--root_path ./dataset/ \
--data_path ETTh1.csv \
--model_id {model_id} \
--model DREAM \
--data ETTh1 \
--features M \
--seq_len 336 \
--pred_len 96 \
--enc_in 7 \
--batch_size 32 \
--learning_rate 0.005 \
--itr 1 \
--des Exp \
--decomposition {decomp} \
--svd_rank {rank}
"""

    print("=" * 80)
    print(f"Running {model_id}")

    os.system(cmd)