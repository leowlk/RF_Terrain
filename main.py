# from fetch_ice_data import IceSatDownload
# from prep_data import Prep, Grid
# from random_forest import RF
# from interpolation import Interp
import csv
import json
import multiprocessing
import os
import pprint
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio

import interpolation
import random_forest
from interpolation import Interp

def main():
    try:
        jparams = json.load(open("params_au.json"))
    except:
        print("ERROR: something is wrong with the params.json file.")
        sys.exit()

    # Filtering for based on uncertainty of < 25 meters
    data = pd.read_csv(jparams["icesat_csv"])
    icepts = data[data["h_te_uncertainty"] < 25].reset_index(drop=True)
    # Points in 3D pandas dataframe
    icepts_LLH = icepts[["lat", "lon", "h_te_interp"]]

    res = jparams["interp"]["resolution"]

    # Access to EPSG according to the middle of the csv string
    parts = jparams["icesat_csv"].split("/")
    # Access the desired element (index 1 in this case)
    if parts[1] == "nzealand":
        epsg = 2193  # New Zealand Transverse Mercator 2000
    if parts[1] == "zlimburg":
        epsg = 28992  # Amersfoort / RD New
    if parts[1] == "ganyon":
        epsg = 32612  # UTM Zone 12

    # convert distances....100m in respective epsg to ?? in
    def transform_coordinates(src_epsg, dst_epsg, x, y):
        src_proj = pyproj.Proj(init=f"epsg:{src_epsg}")
        dst_proj = pyproj.Proj(init=f"epsg:{dst_epsg}")
        lon, lat = pyproj.transform(src_proj, dst_proj, x, y)
        return lon, lat

    # ----- [1] Interpolation of ICESAT-2 Points -> save as features ----- #
    interp_pipeline = ["laplace", "aidw", "idw"]  # 'laplace','idw','tin','nni'
    icepts_NP = icepts_LLH.to_numpy()  # Convert pts from pandas to numpy

    for interp_method in interp_pipeline:
        interp_outtif = jparams["interp"][interp_method]["outfile"]
        if interp_method == "laplace":
            interpolation.laplace_interp(icepts_NP, res, interp_outtif)
        if interp_method == "nni":
            interpolation.nni_interp(icepts_NP, res, interp_outtif)
        if interp_method == "tin":
            interpolation.tin_interp(icepts_NP, res, interp_outtif)
        if interp_method == "idw":
            power = jparams["interp"][interp_method]["power"]
            n_neighbours = jparams["interp"][interp_method]["n_neighbours"]
            interpolation.idw_interp(icepts_NP, res, interp_outtif, power, n_neighbours)
        if interp_method == "aidw":
            n_neighbours = jparams["interp"][interp_method]["n_neighbours"]
            interpolation.aidw_interp(icepts_NP, res, interp_outtif, n_neighbours)

    # ----- [2] Gather GEE features ----- #
    def change_projection():
        pass
        # proj = pyproj.Transformer.from_crs(crs_origin, crs, always_xy=True)
        # xmin, ymin = proj.transform(xmin_ori,ymin_ori)
        # xmax, ymax = proj.transform(xmax_ori,ymax_ori)

    # ----- [3] Prep features data ----- #
    grid = Interp(icepts_NP, res=100)
    grid = grid.make_grid(icepts_NP)

    icepts_LL = icepts_LLH[["lat", "lon"]]
    gridpts_LL = pd.DataFrame(grid, columns=["lat", "lon"])

    icepts_RF = pd.DataFrame()  # Icepoints Lat-Lon
    gridpts_RF = pd.DataFrame()  # Gridpoints Lat-Lon

    features = random_forest.get_features_path(jparams["features-path"])

    # One Hot & Normalise Encoding for ALL PTS
    def process_data(df, feat_label, feat_path, location_data):
        if feat_label.endswith("_c"):
            sampled_df = random_forest.sampleFromTIF(
                feat_label, feat_path, location_data
            )
            oneHotEncoded_df = random_forest.oneHotEncoding(sampled_df, feat_label)
            processed_df = pd.concat([df, oneHotEncoded_df], axis=1)
            print(processed_df)
        else:
            sampled_df = random_forest.sampleFromTIF(
                feat_label, feat_path, location_data
            )
            nsampled_df = sampled_df.drop(columns=["lon", "lat"])
            processed_df = pd.concat([df, nsampled_df], axis=1)
        return processed_df

    for feat_label, feat_path in zip(features.keys(), features.values()):
        # Process data for ICEPTS
        icepts_RF = process_data(icepts_RF, feat_label, feat_path, icepts_LL)
        # Process data for GRIDPTS
        gridpts_RF = process_data(gridpts_RF, feat_label, feat_path, gridpts_LL)

    icepts_RF = pd.concat([icepts_LLH, icepts_RF], axis=1)
    gridpts_RF = pd.concat([gridpts_LL, gridpts_RF], axis=1)

    # ----- Correlation -----
    # print(icepts_RF)
    # correlation = icepts_RF.corr(method='pearson')
    # plt.matshow(correlation)
    # plt.savefig("ice_corr.png")
    # plt.show()

    # Distance to ICEPTS
    distance_to_ice = random_forest.dist_to_pts(icepts_LL, icepts_LL)
    distance_to_grid = random_forest.dist_to_pts(icepts_LL, gridpts_LL)

    # Normalise Interp_h column and concat into gridpts_RF
    # interp_h = random_forest.normaliseScaling(icepts_LLH, "h_te_interp")
    icepts_RF = pd.concat([icepts_RF, distance_to_ice], axis=1)
    gridpts_RF = pd.concat([gridpts_RF, distance_to_grid], axis=1)

    # ----- [4] Random Forest Mahcine Learning ----- #
    results_tif = jparams["results"]["outfile"]
    results_tif = results_tif.removesuffix(".tif")

    models = {
        "nl": "ICESAT/zlimburg/results/zlimburg_pred_sklearn_model.joblib",
        "nz": "ICESAT/nzealand/results/taranaki_pred_sklearn_model.joblib",
        "gc": "ICESAT/gcanyon/results/gcanyon_pred_sklearn_model.joblib",
    }

    optimal_params = {
        "nl": {
            "max_depth": 50,
            "min_samples_split": 2,
            "n_estimators": 500,
            "n_jobs": -1,
        },
        "nz": {
            "max_depth": 10,
            "min_samples_split": 2,
            "n_estimators": 500,
            "n_jobs": -1,
        },
        "gc": {
            "max_depth": 10,
            "min_samples_split": 2,
            "n_estimators": 200,
            "n_jobs": -1,
        },
    }

    random_forest.regression(
        icepts_RF,
        gridpts_RF,
        mode="sklearn",  # mode: "ranger", "sklearn", "xgboost"
        outname=results_tif + "_sklearn.tif",
        save_rf_model=False,
        params={},
    )

    # random_forest.use_rf_model(
    #     icepts_RF,
    #     gridpts_RF,
    #     use_model=None,
    #     outname=results_tif + "_modelNL.tif",
    # )


# ----- [3] Prep features data ----- #

# features = []
# isatgid = random_forest.make_grid(icepts_NP, res=0.0009166)
# k = random_forest.preprocess(icepts_NP, isatgrid, features)


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    spent = t1 - t0
    print(f"==> used {round(spent, 2)} seconds")
