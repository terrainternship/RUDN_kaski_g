import os
import cv2
import numpy as np
import shutil
import random
from datetime import datetime
startTime = datetime.now()


dir_img = r'C:\Users\Lenovo\Desktop\glass_all\train_combi'
dir_lab = r'C:\Users\Lenovo\Desktop\glass_all\labels'

start = r'C:\Users\Lenovo\Desktop\glass_all\glasses_db'
str_dir = ['train', 'valid']

# if not os.path.exists(start):
#     print('Для работы нужно загрузить папку ' + start 
#           + ' с подпапками images и labels')

# if os.path.exists(start):
#     shutil.rmtree(start)

# os.mkdir(start)
for dir_name in str_dir:
    os.mkdir(os.path.join(start, dir_name))
    os.mkdir(os.path.join(start, dir_name, 'images'))
    os.mkdir(os.path.join(start, dir_name, 'labels'))

files = os.listdir(dir_img)
random.shuffle(files)
f_len = len(files)
val_len = int(round(f_len * 0.85, 0))
# tr_len = int(round(f_len * 0.93, 0))

for f in files:
    f1 = '_'.join(f.split('_')[1:])
    print(f1)
    
    if f in files[:val_len]:
        ds_d = str_dir[0]
        shutil.copy2(os.path.join(dir_img, f), os.path.join(start, ds_d, 'images', f1))
        if os.path.exists(os.path.join(dir_lab, f1.replace(".jpg", ".txt"))):
            shutil.copy2(os.path.join(dir_lab, f1.replace(".jpg", ".txt")), os.path.join(start, ds_d, 'labels', f1.replace(".jpg", ".txt")))
#     elif f in files[:tr_len]:
#         ds_d = str_dir[2]
#         shutil.copy2(os.path.join(dir_img, f), os.path.join(start, ds_d, 'images', f))
#         if os.path.exists(os.path.join(dir_lab, f.replace(".jpg", ".txt"))):
#             shutil.copy2(os.path.join(dir_lab, f.replace(".jpg", ".txt")), os.path.join(start, ds_d, 'labels', f.replace(".jpg", ".txt")))
    else:
        ds_d = str_dir[1]
        shutil.copy2(os.path.join(dir_img, f), os.path.join(start, ds_d, 'images', f1))
        if os.path.exists(os.path.join(dir_lab, f1.replace(".jpg", ".txt"))):
            shutil.copy2(os.path.join(dir_lab, f1.replace(".jpg", ".txt")), os.path.join(start, ds_d, 'labels', f1.replace(".jpg", ".txt")))
 
with open(os.path.join(start, 'data.yaml'), 'w') as f:
    path = 'content/drive/MyDrive/Glasses/glasses_seg/glasses_db'
    train = '/content/drive/MyDrive/Glasses/glasses_seg/glasses_db/train'
    val = '/content/drive/MyDrive/Glasses/glasses_seg/glasses_db/valid'
    
    f.write(f"names: \n  0: glasses\npath: {path}\ntrain: {train}\nval: {val}")
    #("train: 'train/images'\nval: 'valid/images'\ntest: 'test/images'\nnc: 1\nnames: ['glasses']")
    
with open(os.path.join(start, 'data.yaml'), 'r') as f:
    print(f.read())

print(f'Время работы - ', datetime.now() - startTime)
