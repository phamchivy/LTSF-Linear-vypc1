import os
import re
import glob
import collections
from datetime import datetime

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(script_dir, ".."))
TARGET_DIR = "FRLinear_output"

def main():
    if not os.path.isdir(TARGET_DIR):
        print(f"Lỗi: Thư mục {TARGET_DIR} không tồn tại ở thư mục gốc!")
        return

    current_date = datetime.now().strftime("%d_%m_%Y")

    # Tìm tên file output chưa tồn tại (result_1_final_..., result_2_final_..., ...)
    counter = 1
    while True:
        output_file = os.path.join(
            TARGET_DIR, f"result_{counter}_final_{current_date}.txt"
        )
        if not os.path.exists(output_file):
            break
        counter += 1

    print("=== ĐANG TRÍCH XUẤT KẾT QUẢ VÀ TÍNH TOÁN MEAN/STD ===")

    lines_out = []

    # --- BƯỚC 1: TRÍCH XUẤT LOG SẠCH ---
    txt_files = glob.glob(os.path.join(TARGET_DIR, "*.txt"))

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        if "final" in filename:
            continue

        lines_out.append(f"--- File: {filename} ---")

        current_start = None
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                clean_line = raw_line.strip()

                if "START RUNNING" in clean_line:
                    current_start = clean_line
                elif "mse:" in clean_line and "mae:" in clean_line:
                    if current_start:
                        lines_out.append(current_start)
                    lines_out.append(clean_line)
                    lines_out.append("")

        lines_out.append("-" * 40)

    # --- BƯỚC 2: THỐNG KÊ MEAN/STD ---
    lines_out.append("")
    lines_out.append("=" * 40)
    lines_out.append("=== THỐNG KÊ TRUNG BÌNH (MEAN & STD) ===")
    lines_out.append("=" * 40)
    lines_out.append("")

    metrics_map = collections.defaultdict(lambda: {"mse": [], "mae": []})
    current_config = None

    for line in lines_out:
        line = line.strip()
        if not line:
            continue

        if "START RUNNING" in line:
            current_config = line
            continue

        if "mse:" in line and "mae:" in line and current_config:
            try:
                parts = line.split(",")
                mse_part = parts[0].split("mse:")[1].strip()
                mae_part = parts[1].split("mae:")[1].strip()

                metrics_map[current_config]["mse"].append(float(mse_part))
                metrics_map[current_config]["mae"].append(float(mae_part))
            except Exception:
                continue

    def sort_key(config_string):
        parts = config_string.split()
        if len(parts) >= 4:
            core_name = parts[3]
            sub_parts = core_name.split("_")
            dataset_name = sub_parts[0]
            try:
                pred_len = int(sub_parts[-1])
                return (dataset_name, pred_len)
            except ValueError:
                return (config_string, 0)
        return (config_string, 0)

    for config in sorted(metrics_map.keys(), key=sort_key):
        mse_list = metrics_map[config]["mse"]
        mae_list = metrics_map[config]["mae"]

        if mse_list and mae_list:
            mean_mse = np.mean(mse_list)
            std_mse = np.std(mse_list)
            mean_mae = np.mean(mae_list)
            std_mae = np.std(mae_list)

            lines_out.append(f"=== {config} ===")
            lines_out.append(f"MSE -> mean = {mean_mse:.3f}, std = {std_mse:.3f}")
            lines_out.append(f"MAE -> mean = {mean_mae:.3f}, std = {std_mae:.3f}")
            lines_out.append("-" * 40)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print("=== HOÀN THÀNH TẤT CẢ! ===")
    print(f"Kiểm tra kết quả tổng hợp chính xác tại: {output_file}")


if __name__ == "__main__":
    main()