from interpolation import Interp
import rasterio
import pandas as pd
import json
import pyproj
import joblib

# Xarray
import xarray as xr
import rioxarray


# Scipy Spatial
from scipy.spatial import distance_matrix, KDTree, distance
from sklearn.neighbors import NearestNeighbors

# SK-learn functions
# from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Random Forest Regressors
from sklearn.ensemble import RandomForestRegressor
from skranger.ensemble import RangerForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, max_error, accuracy_score
from sklearn.inspection import permutation_importance

import os, sys


class Feature:
    def __init__(self, icepts, gridpts) -> None:
        self.icepts = icepts
        self.gridpts = gridpts

    def one_hot_feature(self):
        pass


def get_features_path(directory):
    tif_files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(directory, filename)
            key = os.path.splitext(filename)[0]  # Extract filename without extension
            tif_files[key] = file_path
    return tif_files


def sampleFromTIF(feat_labl, sample_from, sample_pts_df):
    """_summary_

    Args:
        feat_labl (_type_): _description_
        sample_from (_type_): _description_
        sample_pts_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    t = rasterio.open(sample_from, driver="GTiff")
    all_pt_coords = [
        (lon, lat) for lon, lat in zip(sample_pts_df["lon"], sample_pts_df["lat"])
    ]
    sampled_value = [sample[0] for sample in t.sample(all_pt_coords)]
    sampled_df = pd.DataFrame(all_pt_coords, columns=["lon", "lat"])
    sampled_df[feat_labl] = sampled_value
    return sampled_df


def oneHotEncoding(feat_df, feat_labl):
    """_summary_

    Args:
        feat_labl (_type_): _description_
        feat_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    encoder = OneHotEncoder()
    encoder.fit(feat_df[[feat_labl]])
    encoded_labl = encoder.transform(feat_df[[feat_labl]]).toarray()
    encoded_feat_df = pd.DataFrame(
        encoded_labl, columns=encoder.get_feature_names_out()
    )
    return encoded_feat_df


def normaliseScaling(feat_df, feat_labl):
    """_summary_

    Args:
        feat_labl (_type_): _description_
        feat_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    scaler = MinMaxScaler()
    scaled_feat_df = feat_df.copy()
    scaled_feat_num = scaled_feat_df[feat_labl].values.reshape(-1, 1)

    # Fit and transform the selected columns with the StandardScaler
    scaled_feat_df[feat_labl] = scaler.fit_transform(scaled_feat_num)
    scaled_features = scaled_feat_df.drop(columns=["lat", "lon"]).fillna(0)
    scaled_features.fillna(0)
    return scaled_features


class Distances:
    def __init__(self, icepts, gridpts) -> None:
        self.icepts = icepts
        self.gridpts = gridpts

    def convert_coordinates(lon, lat, from_epsg=4326, to_epsg=None):
        from_crs = pyproj.CRS.from_epsg(from_epsg)
        if to_epsg is None:
            # If no "to_epsg" is provided, it will convert to the default CRS of EPSG:3857 (Web Mercator)
            to_epsg = 3857
        to_crs = pyproj.CRS.from_epsg(to_epsg)
        transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
        # Transform the coordinates
        x, y = transformer.transform(lon, lat)
        return x, y

    def dist_to_icepts(self, pts='icepts'):
        knn = NearestNeighbors(n_neighbors=100, algorithm="ball_tree").fit(self.icepts)
        if pts == 'icepts':
            d, i = knn.kneighbors(self.icepts)
        else:
            d, i = knn.kneighbors(self.gridpts)
        dist_df = pd.DataFrame(d)
        dist_df.columns = dist_df.columns.astype(str)
        return dist_df
        
        # else:
        #     d, i = knn.kneighbors(self.gridpts)
        #     dist_df = pd.DataFrame(d)
        #     dist_df.columns = dist_df.columns.astype(str)
        #     return dist_df


class Regression:
    def __init__(self, pts_dataframe, pts_grid) -> None:
        self.df = pts_dataframe
        self.grid_raster = pts_grid

    def test_train(self):
        x = self.df.drop(columns=["h_te_interp"])
        y = self.df["h_te_interp"]  # Target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2
        )

        # VarianceThreshold from sklearn provides a simple baseline approach to feature selection
        return self.X_train, self.X_test, self.y_train, self.y_test

    def sklearn_RFregression(self):
        # Random Forest Regressor
        sklearn_rf_model = RandomForestRegressor(n_jobs=-1)
        # Fitting the Random Forest Regression model to the data
        sklearn_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(sklearn_rf_model)
        self.rf_model = sklearn_rf_model

    def ranger_RFregression(self):
        # Ranger Regressor
        ranger_rf_model = RangerForestRegressor(n_jobs=-1)
        # Fitting the Random Forest Regression model to the data
        ranger_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(ranger_rf_model)
        self.rf_model = ranger_rf_model

    def rf_evaluation(self, rf_model):
        # Use the model to make predictions on the testing data
        y_pred = rf_model.predict(self.X_test)
        # Evaluate the performance of the model
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        max_err = max_error(self.y_test, y_pred)

        print("--------[RF Tests]--------")
        print("MSE:", mse)
        print("R-squared:", r2)
        print("Max error:", max_err)
        print("--------------------------")

    def save(self, outname):
        print(self.grid_raster.columns)
        print(self.X_train.columns)

        # pred_h = pd.DataFrame(self.grid_raster, columns=self.X_train.columns)
        
        # pred_h.dropna(axis=1, inplace=True)
        
        # rf_pred_target = self.rf_model.predict(pred_h)
        # pred_h["pred_h"] = rf_pred_target        
        # latlon_h = pred_h[["lat", "lon", "pred_h"]]

        # # export to Gtiff with 'lat' 'lon' and 'predicted h'
        # xr_pred_h = xr.Dataset.from_dataframe(latlon_h.set_index(["lat", "lon"]))
        # xr_pred_h.rio.set_crs("EPSG:4326")
        # # xr_pred_h.rio.write_nodata(-9999)
        # xr_pred_h.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        # xr_pred_h.rio.to_raster(outname, driver="GTiff")


def dist_to_pts(icepts, gridpts):
    d = Distances(icepts, gridpts)
    if gridpts is icepts:
        return d.dist_to_icepts()
    else:
        return d.dist_to_gridpts()


def regression(iceDF, gridDF, mode="sklearn", outname='outname.tif'):
    i = Regression(iceDF, gridDF)
    X_train, X_test, y_train, y_test = i.test_train()
    if mode == "sklearn":
        i.sklearn_RFregression()
    elif mode == "ranger":
        i.ranger_RFregression()
    i.save(outname)
    
    

def _test():
    pass


if __name__ == "__main__":
    _test()
