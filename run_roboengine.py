import subprocess
from pathlib import Path
import os
# 获取文件夹路径
folder_path = Path("/mnt/c/Users/26370/Desktop/mask")
pics_path = Path("/mnt/c/Users/26370/Desktop/waterline2/images/train")
# 获取所有文件并按修改时间排序（从旧到新）
files = sorted([f for f in folder_path.iterdir() if f.is_file()],
               key=lambda x: x.stat().st_mtime)

# 获取文件名列表
file_names = [f.name for f in files]

for name in file_names:
    pic_name = os.path.splitext(name)[0]
    pic_path = os.path.join(pics_path, pic_name)
    mask_path = os.path.join(folder_path, name)
    subprocess.run(["python", "/mnt/c/Users/26370/Desktop/roboengine-main/roboengine-main/dealbottle_with_mask.py",
                    '--image', f'{pic_path}.jpg','--mask',mask_path,'--num-aug',"3",'--no-robot'])












