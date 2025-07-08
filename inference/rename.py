import os

# 设置目录路径
folder = 'real_test'
files = [f for f in os.listdir(folder) if f.endswith('.wav')]
files.sort()  # 可选：按文件名排序

# 执行重命名
for idx, filename in enumerate(files, 1):
    new_name = f"real_{idx:04d}.wav"
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)

print("✅ 所有文件已重命名完成。")
