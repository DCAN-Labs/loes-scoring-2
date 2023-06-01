import subprocess

in_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/FINAL'
out_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/atlas_reg'

# Sample command
# ./affine_registration_wrapper.sh sub-02 20090924

import os

rootdir = in_dir
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        print(file)
        for file2 in os.listdir(d):
            d2 = os.path.join(d, file2)
            if os.path.isdir(d2):
                print(f'\t{file2}')
                output = subprocess.check_output(['/home/miran045/reine097/projects/loes-scoring-2/bin/affine_registration_wrapper.sh', str(file), str(file2)])
                print(f'\t\toutput: {output}')
