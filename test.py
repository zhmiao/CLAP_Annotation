# %%
import os
import numpy as np
import pandas as pd
from app.utils import *

# %%
ann_df = pd.read_csv("./demo_data/annotations/det_Miao_id_0_seed_0_Miao_ann.csv")

# %%
unique_cat = ann_df["species"].unique()

# %%
current_cat = unique_cat[0]

# %%
ann_df_cat = ann_df.loc[ann_df["species"] == current_cat]

# %%
aud_files_cat = ann_df_cat["filename"].unique()

# %%
for i in range(len(aud_files_cat)):
    aud_f = aud_files_cat[i]
    ann_f = ann_df_cat.loc[ann_df_cat["filename"] == aud_f]
    generate_segs(aud_f, ann_f,
                  prefix=aud_f.split('/')[-1].replace(".wav", ''),
                  clean_seg_folder=(i == 0), save_full=False)

# %%

# %%
