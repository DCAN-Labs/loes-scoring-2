import os
import shutil
from os import listdir
from os.path import isfile, join

base_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/helper/'
for file in os.listdir(base_dir):
    d = os.path.join(base_dir, file)
    if os.path.isdir(d):
        other_dir = os.path.join(d, 'tmp_dcm2bids/helper/other')
        only_files = [f for f in listdir(other_dir) if isfile(join(other_dir, f))]
        file_group_names = []
        for f in only_files:
            start = -1
            for i in range(0, len(f)):
                if f[i].isupper():
                    start = i
                    break
            end = -1
            for j in range(len(f) - 1, start + 1, -1):
                if f[j].isupper():
                    end = j
                    break
            if start != -1 and end != -1:
                file_group_name = f[start:end + 1]
                if file_group_name not in file_group_names:
                    file_group_names.append(file_group_name)
            else:
                print(f'Could not find group: {f}')
        for f in only_files:
            for file_group_name in file_group_names:
                file_group_path = os.path.join(other_dir, file_group_name)
                isExist = os.path.exists(file_group_path)
                if not isExist:
                    os.makedirs(file_group_path)
                if file_group_name in f:
                    shutil.move(join(other_dir, f), join(other_dir, file_group_name, f))
                    break
