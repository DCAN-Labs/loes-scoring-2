import os
import sys


def normalize_file(orig_anat, out, tx):
    # TODO the template we’ll use will be the averaged model training data image
    os.chdir('/home/feczk001/shared/projects/nnunet_predict/task550_testing/hypernorm/')

    if tx=='T1':
        template='templates/0mo_T1_average.nii.gz'
    else:
        template='templates/0mo_T2_average.nii.gz'

    # TODO this is really the only important part of the code
    cmd='fslmaths {} -sub `fslstats {} -M` -div `fslstats {} -S` -mul `fslstats {} -S` -add `fslstats {} -M` {}'.format(orig_anat,orig_anat,orig_anat,template,template,out)
    os.system(cmd)

if __name__ == '__main__':
    orig_anat = sys.argv[1]
    out = sys.argv[2]
    tx = sys.argv[3]
    normalize_file(orig_anat, out, tx)
