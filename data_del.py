import os

# 資料路徑
data_path = 'data/PTT_Gossiping_Sex.txt'

# 初始化變數
valid_data = []

# 資料準備
with open(data_path, 'r', encoding='utf-8') as fp:
    all_lines = fp.readlines()
    for line in all_lines:
        line = line.strip()
        if "  " in line:
            question, answer = line.split("  ", 1)
            if question and answer:
                valid_data.append(line)
            else:
                print(f"Skipping line due to incomplete data: {line}")
        else:
            print(f"Skipping line due to missing separator: {line}")

# # 保存有效的數據回文件中
# with open(data_path, 'w', encoding='utf-8') as fp:
#     for line in valid_data:
#         fp.write(line + '\n')

print(f'Total valid lines: {len(valid_data)}')

# 資料總數: 832271
