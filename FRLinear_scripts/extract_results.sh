#!/bin/bash

# TỰ ĐỘNG NHẢY RA THƯ MỤC GỐC
cd "$(dirname "$0")/.." || exit 1

CURRENT_DATE=$(date +"%d_%m_%Y")
TARGET_DIR="FRLinear_output"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Lỗi: Thư mục $TARGET_DIR không tồn tại ở thư mục gốc!"
    exit 1
fi

COUNTER=1
while [ -f "$TARGET_DIR/result_${COUNTER}_final_${CURRENT_DATE}.txt" ]; do
    COUNTER=$((COUNTER + 1))
done

OUTPUT_FILE="$TARGET_DIR/result_${COUNTER}_final_${CURRENT_DATE}.txt"

echo "=== ĐANG TRÍCH XUẤT KẾT QUẢ VÀ TÍNH TOÁN MEAN/STD ==="

# --- BƯỚC 1: TRÍCH XUẤT LOG SẠCH RA FILE FINAL ---
for file in "$TARGET_DIR"/*.txt; do
    [ -e "$file" ] || continue
    if [[ "$(basename "$file")" =~ "final" ]]; then
        continue
    fi
    
    echo "--- File: $(basename "$file") ---" >> "$OUTPUT_FILE"
    
    current_start=""
    while IFS= read -r line; do
        clean_line=$(echo "$line" | xargs)
        
        if [[ "$clean_line" =~ "START RUNNING" ]]; then
            current_start="$clean_line"
        elif [[ "$clean_line" =~ "mse:" && "$clean_line" =~ "mae:" ]]; then
            if [ -n "$current_start" ]; then
                echo "$current_start" >> "$OUTPUT_FILE"
            fi
            echo "$clean_line" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE" 
        fi
    done < "$file"
    
    echo "----------------------------------------" >> "$OUTPUT_FILE"
done


# --- BƯỚC 2: CHẠY PYTHON ĐỌC TRỰC TIẾP FILE FINAL ĐỂ THỐNG KÊ ---
echo -e "\n========================================" >> "$OUTPUT_FILE"
echo "=== THỐNG KÊ TRUNG BÌNH (MEAN & STD) ===" >> "$OUTPUT_FILE"
echo -e "========================================\n" >> "$OUTPUT_FILE"

python3 - "$OUTPUT_FILE" >> "$OUTPUT_FILE" << 'EOF'
import sys
import collections
import numpy as np

final_file_path = sys.argv[1]
metrics_map = collections.defaultdict(lambda: {'mse': [], 'mae': []})

current_config = None

with open(final_file_path, 'r') as f:
    for line in f:
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
                
                metrics_map[current_config]['mse'].append(float(mse_part))
                metrics_map[current_config]['mae'].append(float(mae_part))
            except Exception:
                continue

# Hàm bổ trợ tách ký tự để sắp xếp thông minh (Tên_Mô_Hình trước, Số_Pred_Len sau)
def sort_key(config_string):
    # config_string dạng: "=== START RUNNING ETTh1_336_96 ==="
    # Tách chuỗi theo khoảng trắng để lấy phần lõi "ETTh1_336_96"
    parts = config_string.split()
    if len(parts) >= 4:
        core_name = parts[3] # Lấy phần "ETTh1_336_96"
        # Tách tiếp theo dấu gạch dưới để lấy tên tập dữ liệu và con số pred_len cuối cùng
        sub_parts = core_name.split('_')
        dataset_name = sub_parts[0] # "ETTh1"
        try:
            pred_len = int(sub_parts[-1]) # Số 96 hoặc 192 hoặc 336...
            return (dataset_name, pred_len)
        except ValueError:
            return (config_string, 0)
    return (config_string, 0)

# In kết quả thống kê theo thứ tự đã được chuẩn hóa số học
for config in sorted(metrics_map.keys(), key=sort_key):
    mse_list = metrics_map[config]['mse']
    mae_list = metrics_map[config]['mae']
    
    if mse_list and mae_list:
        mean_mse = np.mean(mse_list)
        std_mse = np.std(mse_list)
        mean_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
        
        print(f"=== {config} ===")
        print(f"MSE -> mean = {mean_mse:.3f}, std = {std_mse:.3f}")
        print(f"MAE -> mean = {mean_mae:.3f}, std = {std_mae:.3f}")
        print("-" * 40)
EOF

echo "=== HOÀN THÀNH TẤT CẢ! ==="
echo "Kiểm tra kết quả tổng hợp chính xác tại: $OUTPUT_FILE"
