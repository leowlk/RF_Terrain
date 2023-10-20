import pandas as pd
import numpy as np
import json
import interpolation
from interpolation import Interp
import random_forest
import pyproj
import rasterio
import multiprocessing, sys, csv, os, time

import matplotlib.pyplot as plt
import seaborn as sns

def sampleFromTIF(sample_from, sample_pts_df, filterna=False):
    t = rasterio.open(sample_from, driver="GTiff")
    all_pt_coords = [
        (lon, lat) for lon, lat in zip(sample_pts_df["lon"], sample_pts_df["lat"])
    ]
    sampled_value = [sample[0] for sample in t.sample(all_pt_coords)]
    sampled_df = pd.DataFrame(all_pt_coords, columns=["lon", "lat"])
    sampled_df['g_height'] = sampled_value
    if filterna:
        df_filtered = sampled_df[sampled_df.iloc[:, 2] != -9999]
        return df_filtered.reset_index(drop=True)
    else:
        return sampled_df

def main():
    # pts = pd.read_csv("ICESAT/tasmania/tasmania.csv")
    # G_tasmania = rasterio.open("ICESAT/tasmania/gnd_dem_01.tif")
    # p = sampleFromTIF("ICESAT/tasmania/gnd_dem_01.tif", pts, filterna=True)
    # p.to_csv("ICESAT/tasmania/tasmania_g.csv",sep=',')
    orig = pd.read_csv("ICESAT/tasmania/tasmania.csv")
    newg = pd.read_csv("ICESAT/tasmania/tasmania_g.csv")
    origg = orig[['h_te_interp']]
    newgg = newg[['g_height']]
    diff = pd.DataFrame(origg.values - newgg.values)
    
    

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    spent = t1 - t0
    print(f"==> used {round(spent,2)} seconds")
