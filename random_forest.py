import rasterio
import pandas as pd
import numpy as np
import json
import pyproj
import joblib
import matplotlib.pyplot as plt

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
from sklearn.ensemble import RandomForestRegressor  # SKLearn
from skranger.ensemble import RangerForestRegressor  # Ranger
import xgboost as xgb  # XGBoost

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection

from sklearn.metrics import mean_squared_error, r2_score, max_error, accuracy_score
from sklearn.inspection import permutation_importance

import os, sys


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


class Geometry:
    def __init__(self, icepts, gridpts) -> None:
        self.icepts = icepts[["lat", "lon"]]
        self.iceptsH = icepts[["lat", "lon", "h_te_interp"]]
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

    def dist_to(self, pts="icepts"):
        knn = NearestNeighbors(n_neighbors=100, algorithm="ball_tree").fit(self.icepts)
        if pts == "icepts":
            d, i = knn.kneighbors(self.icepts)
        elif pts == "gridpts":
            d, i = knn.kneighbors(self.gridpts)
        else:
            print("Invalid points specified.")
            return None
        dist_df = pd.DataFrame(d)
        dist_df.columns = "dist_buffer_" + dist_df.columns.astype(str)
        return dist_df

    def height_to(self, pts="icepts"):
        """Calculate Nearest Neighbour Height

        Args:
            pts (str, optional): _description_. Defaults to "icepts".

        Returns:
            _type_: _description_
        """
        knn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(self.icepts)
        if pts == "icepts":
            d, i = knn.kneighbors(self.icepts)
            near_h = [
                [self.iceptsH["h_te_interp"][h_index] for h_index in row[1:]]
                for row in i
            ]
            # near_h = []
            # for row in i:
            #     tmp = []
            #     for h_index in row:
            #         ht = self.iceptsH['h_te_interp'][h_index]
            #         tmp.append(ht)
            #     near_h.append(tmp)

        elif pts == "gridpts":
            d, i = knn.kneighbors(self.gridpts)
            near_h = [
                [self.iceptsH["h_te_interp"][h_index] for h_index in row[1:]]
                for row in i
            ]
            # near_h = []
            # for row in i:
            #     tmp = []
            #     for h_index in row:
            #         ht = self.iceptsH['h_te_interp'][h_index]
            #         tmp.append(ht)
            #     near_h.append(tmp)
        else:
            print("Invalid points specified.")
            return None

        nearest_h_df = pd.DataFrame(near_h)
        nearest_h_df.columns = "nearest_height_" + nearest_h_df.columns.astype(str)
        return nearest_h_df

    def angle_to(self, pts="icepts"):
        """Calculate the Angle Relative to the Nearest Neighbour

        Args:
            pts (str, optional): _description_. Defaults to "icepts".

        Returns:
            _type_: _description_
        """
        knn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(self.icepts)
        if pts == "icepts":
            d, i = knn.kneighbors(self.icepts)
            near_angl = []
            for row in i:
                pt_0 = self.iceptsH.iloc[row[0]].to_numpy()
                magO = np.linalg.norm(pt_0)
                pt_rest = row[1:]
                _tmp = []
                for r in pt_rest:
                    vecA = self.iceptsH.iloc[r].to_numpy()
                    # Dot product of vector vs origin
                    dot_product = np.dot(pt_0, vecA)
                    magA = np.linalg.norm(vecA)
                    cosine_theta = dot_product / (magO * magA)
                    theta_rad = np.arccos(cosine_theta)
                    theta_deg = np.degrees(theta_rad)
                    _tmp.append(theta_deg)
                near_angl.append(_tmp)

        elif pts == "gridpts":
            d, i = knn.kneighbors(self.gridpts)
            near_angl = []
            for row in i:
                pt_0 = self.iceptsH.iloc[row[0]].to_numpy()
                magO = np.linalg.norm(pt_0)
                pt_rest = row[1:]
                _tmp = []
                for r in pt_rest:
                    vecA = self.iceptsH.iloc[r].to_numpy()
                    # Dot product of vector vs origin
                    dot_product = np.dot(pt_0, vecA)
                    magA = np.linalg.norm(vecA)
                    cosine_theta = dot_product / (magO * magA)
                    theta_rad = np.arccos(cosine_theta)
                    theta_deg = np.degrees(theta_rad)
                    _tmp.append(theta_deg)
                near_angl.append(_tmp)

        else:
            print("Invalid points specified.")
            return None

        angle_df = pd.DataFrame(near_angl)
        angle_df.columns = "angle_btwn_" + angle_df.columns.astype(str)
        return angle_df

    def relativeh_to(self, pts="icepts"):
        """Calculate Relative Height In Relation to Nearest Neighbour

        Args:
            pts (str, optional): _description_. Defaults to "icepts".

        Returns:
            _type_: _description_
        """
        knn = NearestNeighbors(n_neighbors=100, algorithm="ball_tree").fit(self.icepts)
        if pts == "icepts":
            d, i = knn.kneighbors(self.icepts)
            relative_h_list = []
            for row in i:
                pt_0 = self.iceptsH.iloc[row[0]].to_numpy()
                pt_rest = row[1:]
                _tmp = []
                for r in pt_rest:
                    pt_r = self.iceptsH.iloc[r].to_numpy()
                    diff = pt_r - pt_0
                    _tmp.append(diff[2])
                relative_h_list.append(_tmp)

        elif pts == "gridpts":
            d, i = knn.kneighbors(self.gridpts)
            relative_h_list = []
            for row in i:
                pt_0 = self.iceptsH.iloc[row[0]].to_numpy()
                pt_rest = row[1:]
                _tmp = []
                for r in pt_rest:
                    pt_r = self.iceptsH.iloc[r].to_numpy()
                    diff = pt_r - pt_0
                    _tmp.append(diff[2])
                relative_h_list.append(_tmp)
        else:
            print("Invalid points specified.")
            return None

        relative_h_df = pd.DataFrame(relative_h_list)
        relative_h_df.columns = "relative_h_" + relative_h_df.columns.astype(str)
        return relative_h_df   
    
    def slope_to(self, pts="icepts"):
        """Calculate Slope to nearest neighbour

        Args:
            pts (str, optional): _description_. Defaults to "icepts".

        Returns:
            _type_: _description_
        """
        knn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(self.icepts)
        if pts == "icepts":
            d, i = knn.kneighbors(self.icepts)
            slope_list = []
            for row in i:
                pt_0 = self.iceptsH.iloc[row[0]].to_numpy()
                pt_rest = row[1:]
                _tmp = []
                for r in pt_rest:
                    pt_r = self.iceptsH.iloc[r].to_numpy()
                    diff = pt_r - pt_0
                    
                    
                    
                    _tmp.append(diff[2])
                slope_list.append(_tmp)

        elif pts == "gridpts":
            d, i = knn.kneighbors(self.gridpts)
            slope_list = []
            for row in i:
                pt_0 = self.iceptsH.iloc[row[0]].to_numpy()
                pt_rest = row[1:]
                _tmp = []
                for r in pt_rest:
                    pt_r = self.iceptsH.iloc[r].to_numpy()
                    diff = pt_r - pt_0
                    _tmp.append(diff[2])
                slope_list.append(_tmp)
        else:
            print("Invalid points specified.")
            return None

        slope_df = pd.DataFrame(slope_list)
        slope_df.columns = "slope_h_" + slope_df.columns.astype(str)
        return slope_df

class Regression:
    def __init__(self, pts_dataframe, pts_grid) -> None:
        self.df = pts_dataframe
        self.grid_raster = pts_grid
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def test_train(self):
        """Test-Train Split on Random Forest Regression

        Returns:
            _type_: _description_
        """
        self.x_ml = self.df.drop(columns=["h_te_interp"])
        self.y_ml = self.df["h_te_interp"]  # Target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x_ml, self.y_ml, test_size=0.2, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    # VarianceThreshold from sklearn provides a simple baseline approach to feature selection
    def perm_importance(self, rf_model):
        print("Permutation Importance:")
        perm_importance = permutation_importance(
            rf_model, self.X_test, self.y_test, n_repeats=30, random_state=42
        )
        print("Results:")
        importance_mean = perm_importance.importances_mean
        importance_std = perm_importance.importances_std

        sorted_idx = importance_mean.argsort()  # Sort indices in descending order
        features = self.X_test.columns.tolist()
        sorted_list = []
        for i in sorted_idx:
            sorted_list.append([features[i], round(importance_mean[i], 5)])
            # sorted_list.append([features[i], round(importance_mean[i], 5), round(importance_std[i], 5)])
        I = pd.DataFrame(sorted_list)

        I.columns = ["features", "importance"]

        I["features"] = I["features"].str.extract(r"(\w+)_")
        I = I.groupby("features")["importance"].sum().reset_index()
        I.to_csv("perm_importance.csv")
        print(I)

    def mdi_importance(self, rf_model):
        print("MDI Importance:")
        mdi_importance = rf_model.feature_importances_

        sorted_idx = mdi_importance.argsort()  # Sort indices in descending order
        features = self.X_test.columns.tolist()
        sorted_list = []
        for i in sorted_idx:
            sorted_list.append([features[i], round(mdi_importance[i], 5)])
            # sorted_list.append([features[i], round(importance_mean[i], 5), round(importance_std[i], 5)])
        I = pd.DataFrame(sorted_list)

        I.columns = ["features", "importance"]

        I["features"] = I["features"].str.extract(r"(\w+)_")
        I = I.groupby("features")["importance"].sum().reset_index()
        I.to_csv("mdi_importance.csv")
        print(I)

    def sklearn_RFregression(self, use_model=None, params={}):
        self.test_train()
        # ----- Random Forest Regressor -----
        if use_model == None:
            sklearn_rf_model = RandomForestRegressor(**params)

        # use model (if any) && use params (if any)
        else:
            print("use model is present")
            sklearn_rf_model = joblib.load(use_model)
        sklearn_rf_model.fit(self.X_train, self.y_train)

        # ----- Print RF Tree -----
        from sklearn import tree

        # print(len(sklearn_rf_model.estimators_))
        # # plt.figure(figsize=(10,10))
        # _ = tree.plot_tree(
        #     sklearn_rf_model.estimators_[0],
        #     feature_names=list(self.X_train.columns),
        #     filled=True,
        #     node_ids=True,
        #     fontsize=4,
        #     max_depth=3,
        # )
        # # plt.savefig('plotted_tree.png')
        # plt.show()

        # ----- Feature Importances -----
        # self.perm_importance(sklearn_rf_model)
        self.mdi_importance(sklearn_rf_model)
        # self.sfs_selection(sklearn_rf_model)
        self.rf_evaluation(sklearn_rf_model)
        self.rf_model = sklearn_rf_model
        return sklearn_rf_model

    # -------- Grid Search CV --------
    def grid_search(self):
        search_space = {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "n_jobfs": [-1],
        }

        # make a GridSearchCV object
        GS = GridSearchCV(
            estimator=self.rf_model,
            param_grid=search_space,
            # sklearn.metrics.SCORERS.keys()
            scoring=["r2", "neg_root_mean_squared_error"],
            refit="r2",
            cv=5,
            verbose=4,
        )
        GS.fit(self.X_train, self.y_train)
        print(GS.best_estimator_)
        print(GS.best_params_)
        print(GS.best_score_)

    def ranger_RFregression(self):
        self.test_train()
        # ----- Ranger Regressor -----
        ranger_rf_model = RangerForestRegressor(n_jobs=-1)
        # Fitting the Random Forest Regression model to the data
        ranger_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(ranger_rf_model)
        self.rf_model = ranger_rf_model
        return ranger_rf_model

    def xgboost_RFregression(self):
        self.test_train()
        # ----- XGBoost Regressor -----
        xgb_rf_model = xgb.XGBRegressor(n_jobs=-1, objective='reg:quantileerror')

        # [XGB] Fitting the RF Regression model to the data
        xgb_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(xgb_rf_model)
        self.rf_model = xgb_rf_model
        return xgb_rf_model

    def rf_evaluation(self, rf_model):
        # Use the model to make predictions on the testing data
        y_pred = rf_model.predict(self.X_test)
        # Evaluate the performance of the model
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        max_err = max_error(self.y_test, y_pred)
        rf_train_acc = rf_model.score(self.X_train, self.y_train)
        rf_test_acc = rf_model.score(self.X_test, self.y_test)

        print("--------[RF Tests]--------")
        print(f"MSE: {mse:.3f}")
        print(f"R-squared: {r2:.3f}")
        print(f"Max error: {max_err:.3f}")
        print(f"RF train accuracy: {rf_train_acc:.3f}")
        print(f"RF test accuracy: {rf_test_acc:.3f}")
        print("--------------------------")

        # residuals = self.y_test - y_pred
        # plt.figure(figsize=(8, 6))
        # plt.scatter(y_pred, residuals, c="blue", marker="o", label="Residuals")
        # plt.axhline(y=0, color="red", linestyle="-", label="Zero Residual Line")
        # plt.xlabel("Predicted Values")
        # plt.ylabel("Residuals")
        # plt.title("Residual Plot for Random Forest Regression")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.scatter(self.y_ml, y_pred, alpha=0.5)
        # plt.xlabel("Observed Values")
        # plt.ylabel("Predicted Values")
        # plt.title("Scatter Plot of Predicted vs Observed Values for Random Forest Regression")
        # plt.grid(True)
        # plt.show()

        # if val_method == 'kfold'

        # print(rf_model.feature_importances_)
        # print(rf_model.n_features_in_)
        # print(rf_model.feature_names_in_)

    def save_rfmodel(self, rf_modelname):
        # save the model to disk
        joblib.dump(self.rf_model, rf_modelname)

    def sfs_selection(self, rf_model):
        print("SFS Features Selection:")
        sfs = SFS(
            rf_model,
            k_features=10,
            forward=True,
            floating=False,
            scoring="neg_mean_squared_error",
            cv=None,
        )
        sfs = sfs.fit(self.x_ml, self.y_ml)
        selected_feature_indices = sfs.k_feature_idx_
        elected_features = self.X_train.columns[list(selected_feature_indices)]
        print(elected_features)

        fig = plot_sequential_feature_selection(sfs.get_metric_dict(), kind="std_dev")
        # Customize the plot (optional)
        plt.title("Sequential Forward Selection (SFS)")
        plt.grid()
        plt.show()
        return None

    def output_tif(self, outname):
        pred_h = pd.DataFrame(self.grid_raster, columns=self.X_train.columns)
        pred_h.dropna(axis=1, inplace=True)

        rf_pred_target = self.rf_model.predict(pred_h)
        pred_h["pred_h"] = rf_pred_target
        latlon_h = pred_h[["lat", "lon", "pred_h"]]

        # export to Gtiff with 'lat' 'lon' and 'predicted h'
        xr_pred_h = xr.Dataset.from_dataframe(latlon_h.set_index(["lat", "lon"]))
        xr_pred_h.rio.set_crs("EPSG:4326")
        # xr_pred_h.rio.write_nodata(-9999)
        xr_pred_h.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        xr_pred_h.rio.to_raster(outname, driver="GTiff")

    # def cross_validation(self, rf_model, method='kfold'):
    #     if method=='kfold':
    #         scores=[]
    #         kFold=KFold(n_splits=10,random_state=42,shuffle=False)
    #         for train_index,test_index in kFold.split(X):
    #             print("Train Index: ", train_index, "\n")
    #             print("Test Index: ", test_index)

    #             X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    #             knn.fit(X_train, y_train)
    #             scores.append(knn.score(X_test, y_test))
    #             knn.fit(X_train, y_train)
    #             scores.append(knn.score(X_test,y_test))
    #             print(np.mean(scores))
    #             0.9393939393939394
    #             cross_val_score(knn, X, y, cv=10)


def dist_to_pts(icepts, gridpts):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.dist_to("icepts")
    else:
        return d.dist_to("gridpts")


def height_to_pts(icepts, gridpts):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.height_to("icepts")
    else:
        return d.height_to("gridpts")


def angle_to_pts(icepts, gridpts):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.angle_to("icepts")
    else:
        return d.angle_to("gridpts")


def relativeh_to_pts(icepts, gridpts):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.relativeh_to("icepts")
    else:
        return d.relativeh_to("gridpts")

def slope_to_pts(icepts, gridpts):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.slope_to("icepts")
    else:
        return d.slope_to("gridpts")



def regression(
    iceDF,
    gridDF,
    mode="sklearn",
    outname="outname.tif",
    save_rf_model=False,
    params=None,
):
    save_modelname = outname.removesuffix(".tif") + "_model.joblib"

    r = Regression(iceDF, gridDF)
    if mode == "sklearn":
        r.sklearn_RFregression(params=params)
        # if save_model
        if save_rf_model == True:
            joblib.dump(r.rf_model, save_modelname)

    elif mode == "ranger":
        r.ranger_RFregression()
        if save_rf_model == True:
            r.save_rfmodel(save_modelname)

    elif mode == "xgboost":
        r.xgboost_RFregression()
        if save_rf_model == True:
            r.save_rfmodel(save_modelname)
    else:
        print("got nothing.")

    # Output the predicted height to TIF file
    r.output_tif(outname)


def use_rf_model(iceDF, gridDF, use_model=None, outname="outname.tif"):
    r = Regression(iceDF, gridDF)
    r.sklearn_RFregression(use_model=use_model)

    print("outputting...")
    r.output_tif(outname)

    # r.rf_evaluation(Kfold)


def _test():
    pass


if __name__ == "__main__":
    _test()
