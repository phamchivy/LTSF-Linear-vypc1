# dream_research/scripts/train_phase1.py

import os

decompositions = [
    "ma",
    "fft",
    "wavelet"
]

for decomp in decompositions:

    cmd = f"""
python run_longExp.py \
--is_training 1 \
--root_path ./dataset/ \
--data_path ETTh1.csv \
--model_id ETTh1_336_96_P1_{decomp} \
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
--decomposition {decomp}
"""

    print("=" * 80)
    print(f"Running {decomp}")
    os.system(cmd)