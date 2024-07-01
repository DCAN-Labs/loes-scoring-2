from fastMONAI.vision_all import *
import pandas as pd

path = Path('/home/feczk001/shared/data/loes_scoring/')

df = pd.read_csv(path/'Nascene_deID_files.csv')
df['loes_score'] = np.around(df.loes_score.tolist(), decimals=0)

df.head()

print([df.loes_score.min(), df.loes_score.max()])

med_dataset = MedDataset(path='/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/', max_workers=12)
data_info_df = med_dataset.summary()
print(data_info_df.head())
resample, reorder = med_dataset.suggestion()

bs=4
in_shape = [1, 197, 233, 189]

item_tfms = [ZNormalization(), PadOrCrop(in_shape[1:]), RandomAffine(scales=0, degrees=5, isotropic=False)] 

def add_folder(row):
   return os.path.join('/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/', row['FILE'])

df['full_path'] = df.apply(add_folder, axis=1)

dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), RegressionBlock), 
                      splitter=RandomSplitter(seed=1),
                      get_x=ColReader('full_path'),
                      get_y=ColReader('loes_score'),
                      item_tfms=item_tfms,
                      reorder=reorder,
                      resample=resample) 

dls = dblock.dataloaders(df, bs=bs)

print([len(dls.train_ds.items), len(dls.valid_ds.items)])

from monai.networks.nets import Regressor
model = Regressor(in_shape=[1, 197, 233, 189], out_shape=1, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), kernel_size=3, num_res_units=2)

loss_func = L1LossFlat()

learn = Learner(dls, model, loss_func=loss_func, metrics=[mae])

learn.lr_find()

