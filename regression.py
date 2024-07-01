#!/usr/bin/env python
# coding: utf-8

# # Regression
# > We will use the same data used in the classification tutorial for this task (the IXI Dataset). The approach for regression tasks is nearly identical. Therefore, take a look at the classification tutorial for explanations of the various cells.
# ---
# skip_showdoc: true
# skip_exec: true
# ---
# [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMIV-ML/fastMONAI/blob/master/nbs/10b_tutorial_regression.ipynb)

# In[ ]:


print('Running locally')


# In[ ]:


from fastMONAI.vision_all import *


# In[ ]:


path = Path('../data')
path.mkdir(exist_ok=True)


# In[ ]:


STUDY_DIR = download_ixi_data(path=path)


# ### Looking at the data

# In[ ]:


df = pd.read_csv(STUDY_DIR/'dataset.csv')
df['age'] = np.around(df.age_at_scan.tolist(), decimals=0)


# In[ ]:


df.head()


# In[ ]:


df.age.min(), df.age.max()


# In[ ]:


med_dataset = MedDataset(path=STUDY_DIR/'T1_images', max_workers=12)


# In[ ]:


data_info_df = med_dataset.summary()


# In[ ]:


data_info_df.head()


# In[ ]:


resample, reorder = med_dataset.suggestion()


# In[ ]:


bs=4
in_shape = [1, 256, 256, 160]


# In[ ]:


item_tfms = [ZNormalization(), PadOrCrop(in_shape[1:]), RandomAffine(scales=0, degrees=5, isotropic=False)] 


# In[ ]:


dblock = MedDataBlock(blocks=(ImageBlock(cls=MedImage), RegressionBlock), 
                      splitter=RandomSplitter(seed=32),
                      get_x=ColReader('t1_path'),
                      get_y=ColReader('age'),
                      item_tfms=item_tfms,
                      reorder=reorder,
                      resample=resample) 


# In[ ]:


dls = dblock.dataloaders(df, bs=bs)


# In[ ]:


len(dls.train_ds.items), len(dls.valid_ds.items)


# In[ ]:


dls.show_batch(anatomical_plane=2)


# ### Create and train a 3D model

# Import a network from MONAI that can be used for regression tasks, and define the input image size, the output size, channels, etc.  

# In[ ]:


from monai.networks.nets import Regressor
model = Regressor(in_shape=[1, 256, 256, 160], out_shape=1, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), kernel_size=3, num_res_units=2)


# In[ ]:


loss_func = L1LossFlat()


# In[ ]:


learn = Learner(dls, model, loss_func=loss_func, metrics=[mae])


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find()


# In[ ]:


lr = 1e-4


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('model-brainage'); 


# ### Inference

# In[ ]:


learn.load('model-brainage'); 


# In[ ]:


interp = Interpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(k=9, anatomical_plane=2)

