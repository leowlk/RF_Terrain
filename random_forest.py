import rasterio
import pandas as pd
import numpy as np
import json
import pyproj
import joblib
import math
import matplotlib.pyplot as plt
import seaborn as sns
import startinpy

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
    LearningCurveDisplay,
    learning_curve,
    validation_curve
)

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.feature_selection import SelectFromModel

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
        self.iceptsH = icepts[["lat", "lon", "g_height"]]
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

    def dist_to(self, pts="icepts", where=None):
        epsg_mapping = {"au": 5551, "nz": 2193, "nl": 28992, "us": 32612}
        epsg = epsg_mapping.get(where, 4326)
        
        icepts_LLH_T = self.transform_datafr(4326, epsg, self.iceptsH)
        icepts_LL_T = self.transform_datafr(4326, epsg, self.icepts)
        gridpts_LL_T = self.transform_datafr(4326, epsg, self.gridpts)
        
        gridpts_T = gridpts_LL_T[["x", "y"]]
        icepts_T = icepts_LL_T[["x", "y"]]
        iceptsH_T = icepts_LLH_T[["x", "y", "g_height"]]

        knn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(icepts_T)
        if pts == "icepts":
            d, i = knn.kneighbors(icepts_T)
            d = d[:, 1:]
        elif pts == "gridpts":
            d, i = knn.kneighbors(gridpts_T)
            d = d[:, 1:]
        else:
            print("Invalid points specified.")
            return None
        dist_df = pd.DataFrame(d)
        dist_df.columns = "distance_to_" + dist_df.columns.astype(str)
        return dist_df

    def height_to(self, pts="icepts"):
        """Calculate Nearest Neighbour Height

        Args:
            pts (str, optional): _description_. Defaults to "icepts".

        Returns:
            _type_: _description_
        """
        knn = NearestNeighbors(n_neighbors=11, algorithm="ball_tree").fit(self.icepts)
        if pts == "icepts":
            d, i = knn.kneighbors(self.icepts)
            near_h = [
                [self.iceptsH["g_height"][h_index] for h_index in row[1:]]
                for row in i
            ]
            # near_h = []
            # for row in i:
            #     tmp = []
            #     for h_index in row:
            #         ht = self.iceptsH['g_height'][h_index]
            #         tmp.append(ht)
            #     near_h.append(tmp)

        elif pts == "gridpts":
            d, i = knn.kneighbors(self.gridpts)
            near_h = [
                [self.iceptsH["g_height"][h_index] for h_index in row[1:]]
                for row in i
            ]
            # near_h = []
            # for row in i:
            #     tmp = []
            #     for h_index in row:
            #         ht = self.iceptsH['g_height'][h_index]
            #         tmp.append(ht)
            #     near_h.append(tmp)
        else:
            print("Invalid points specified.")
            return None

        nearest_h_df = pd.DataFrame(near_h)
        nearest_h_df.columns = "nearest_height_" + nearest_h_df.columns.astype(str)
        return nearest_h_df

    def angle_to(self, pts="icepts"):
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

    def transform_datafr(self, from_epsg, to_epsg, df):
        src_crs = pyproj.CRS.from_epsg(from_epsg)  # WGS84
        target_crs = pyproj.CRS.from_epsg(to_epsg)  # Web Mercator
        # Create a transformer
        transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=False)
        df["x"], df["y"] = transformer.transform(df["lat"].values, df["lon"].values)
        return df

    def relativeh_to(self, pts="icepts"):
        knn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(self.icepts)
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
    
    
    def slope_to(self, pts="icepts", where=None):
        epsg_mapping = {"au": 5551, "nz": 2193, "nl": 28992, "us": 32612}
        epsg = epsg_mapping.get(where, 4326)
            
        icepts_LLH_T = self.transform_datafr(4326, epsg, self.iceptsH)
        icepts_LL_T = self.transform_datafr(4326, epsg, self.icepts)
        gridpts_LL_T = self.transform_datafr(4326, epsg, self.gridpts)
        
        gridpts_T = gridpts_LL_T[["x", "y"]]
        icepts_T = icepts_LL_T[["x", "y"]]
        iceptsH_T = icepts_LLH_T[["x", "y", "g_height"]]
        
        knn = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(icepts_T)
        
        if pts == "icepts":
            d, i = knn.kneighbors(icepts_T)
            slope_list = []
            for row in i:
                pt_0 = iceptsH_T.iloc[row[0]].to_numpy()
                pt_rest = row[1:]
                _tmp = []
                slope_vec = []
                                
                for r in pt_rest:
                    pt_r = iceptsH_T.iloc[r].to_numpy()          
                    slope_vector = pt_r - pt_0
                    slope_vec.append(slope_vector)
                slope_vec = np.array(slope_vec)
                
                N = len(slope_vec)
                sum_of_slopes = 0
                for i in range(N - 1):
                    delta_x = slope_vec[i + 1][0] - pt_0[0]
                    delta_y = slope_vec[i + 1][1] - pt_0[1]
                    delta_z = slope_vec[i + 1][2] - pt_0[2]
                    slope_i = np.sqrt((delta_z / delta_x)**2 + (delta_z / delta_y)**2)
                    sum_of_slopes += slope_i
                average_slope = sum_of_slopes / (N - 1)
                slope_list.append(average_slope)

            
            # slope_list = []
            # for row in i:
            #     pt_0 = iceptsH_T.iloc[row[0]].to_numpy()
            #     pt_rest = row[1:]
            #     print(pt_rest)
                                
            #     _tmp = []
            #     for r in pt_rest:
            #         pt_r = iceptsH_T.iloc[r].to_numpy()
            #         slope_vector = pt_r - pt_0
            #         d_x, d_y, d_z = slope_vector
            #         overall_slope = d_z / math.sqrt(d_x**2 + d_y**2 + d_z**2)
            #         _tmp.append(overall_slope)
            #     slope_list.append(_tmp)

        elif pts == "gridpts":
            d, i = knn.kneighbors(gridpts_T)
            slope_list = []
            for row in i:
                pt_0 = iceptsH_T.iloc[row[0]].to_numpy()
                pt_rest = row[1:]
                _tmp = []
                slope_vec = []
                                
                for r in pt_rest:
                    pt_r = iceptsH_T.iloc[r].to_numpy()          
                    slope_vector = pt_r - pt_0
                    slope_vec.append(slope_vector)
                slope_vec = np.array(slope_vec)
                
                N = len(slope_vec)
                sum_of_slopes = 0
                for i in range(N - 1):
                    delta_x = slope_vec[i + 1][0] - pt_0[0]
                    delta_y = slope_vec[i + 1][1] - pt_0[1]
                    delta_z = slope_vec[i + 1][2] - pt_0[2]
                    slope_i = np.sqrt((delta_z / delta_x)**2 + (delta_z / delta_y)**2)
                    sum_of_slopes += slope_i
                average_slope = sum_of_slopes / (N - 1)
                slope_list.append(average_slope)
        else:
            print("Invalid points specified.")
            return None
        slope_df = pd.DataFrame(slope_list)
        slope_df.columns = "slope_h_" + slope_df.columns.astype(str)
        return slope_df
    
    def centroid_to(self, pts="icepts", where=None):
        epsg_mapping = {"au": 5551, "nz": 2193, "nl": 28992, "us": 32612}
        epsg = epsg_mapping.get(where, 4326)

        icepts_LLH_T = self.transform_datafr(4326, epsg, self.iceptsH)
        icepts_LL_T = self.transform_datafr(4326, epsg, self.icepts)
        gridpts_LL_T = self.transform_datafr(4326, epsg, self.gridpts)

        gridpts_T = gridpts_LL_T[["x", "y"]]
        icepts_T = icepts_LL_T[["x", "y"]]
        iceptsH_T = icepts_LLH_T[["x", "y", "g_height"]]

        lat = icepts_T[['y']].to_numpy()
        lon = icepts_T[['x']].to_numpy()
        c_lat = np.mean(lat)
        c_lon = np.mean(lon)
        centroid = np.array([c_lon, c_lat])
        
        if pts == "icepts":
            cent_list=[]
            for p in icepts_T.values:
                d = distance.euclidean(p, centroid)
                cent_list.append(d)

        elif pts == "gridpts":
            cent_list=[]
            for p in gridpts_T.values:
                d = distance.euclidean(p, centroid)
                cent_list.append(d)
        else:
            print("Invalid points specified.")
            return None
        
        cent_list = pd.DataFrame(cent_list)
        cent_list.columns = "centroid_dist_" + cent_list.columns.astype(str)
        return cent_list

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
        self.x_ml = self.df.drop(columns=["g_height", "lat", "lon"])
        self.y_ml = self.df["g_height"]  # Target
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
        I2 = I.groupby("features")["importance"].sum().reset_index()
        I.to_csv("mdi_importance.csv")
        I2.to_csv("mdi_importance2.csv")
        
        fig, ax = plt.subplots()
        I.plot.bar(ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

        print(I)
        
    # -------- Grid Search CV --------
    def grid_search(self):
        search_space = {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "n_jobfs": [-1]
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
    
    # ------ Validation Score -------
    def validation_score(self):
        sns.set_context("notebook")
        param_range = [10, 50, 100, 150, 200]
        train_scores, valid_scores = validation_curve(
            self.rf_model,
            self.X_train, self.y_train,
            param_name="n_estimators",
            param_range=param_range,
            cv=5,  # Cross-validation folds
            scoring=None,  # Use an appropriate scoring metric
            n_jobs=-1
        )
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = -np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)
                
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, label="Training score", color="red", marker="o")
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")

        plt.plot(param_range, valid_mean, label="Cross-validation score", color="green", marker="o")
        plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2, color="green")

        plt.title("Validation Curve for Random Forest")
        plt.xlabel("Number of Estimators")
        plt.ylabel("Negative Mean Squared Error")
        plt.legend(loc="best")
        plt.grid()
        plt.show()
    
    # def learning_score(self):
    #     train_sizes = np.linspace(0.1, 1.0, 10)
    #     train_sizes, train_scores, valid_scores = learning_curve(
    #         RandomForestRegressor(),
    #         self.X_train, self.y_train,
    #         train_sizes=train_sizes,
    #         cv=5,  # Number of cross-validation folds
    #         scoring="neg_mean_squared_error"  # Use an appropriate scoring metric
    #     )
    #     train_mean = -np.mean(train_scores, axis=1)
    #     train_std = np.std(train_scores, axis=1)
    #     valid_mean = -np.mean(valid_scores, axis=1)
    #     valid_std = np.std(valid_scores, axis=1)
        
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(train_sizes, train_mean, label="Training score", color="blue", marker="o")
    #     plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")

    #     plt.plot(train_sizes, valid_mean, label="Validation score", color="green", marker="o")
    #     plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2, color="green")

    #     plt.title("Learning Curve for Random Forest Regressor")
    #     plt.xlabel("Training Set Size")
    #     plt.ylabel("Negative Mean Squared Error")
    #     plt.legend(loc="best")
    #     plt.grid()
    #     plt.show()
    def permutation_importances(self, rf, X_train, y_train, metric):
        baseline = metric(rf, X_train, y_train)
        imp = []
        for col in X_train.columns:
            save = X_train[col].copy()
            X_train[col] = np.random.permutation(X_train[col])
            m = metric(rf, X_train, y_train)
            X_train[col] = save
            imp.append(baseline - m)
        return np.array(imp)

    def sklearn_RFregression(self, use_model=None, params={}):
        self.test_train()
        # ----- Random Forest Regressor -----
        if use_model == None:
            sklearn_rf_model = RandomForestRegressor(oob_score=True, **params)
            rfe = RFE(estimator=sklearn_rf_model, n_features_to_select=5)
            # pipeline = Pipeline(steps=[('s',rfe),('m',sklearn_rf_model)])

        # use model (if any) && use params (if any)
        else:
            print("use model is present")
            sklearn_rf_model = joblib.load(use_model)
        
        # print x_train and y_train

        
        sklearn_rf_model.fit(self.X_train, self.y_train)
        mdi_importance = sklearn_rf_model.feature_importances_

        # Sort feature importance scores in descending order
        sorted_indices = np.argsort(mdi_importance)
        sorted_importance = mdi_importance[sorted_indices]
        
        feature_names = self.X_test.columns  # If you have column names for your features

        # Horizontal bar plot to visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_importance)), sorted_importance)
        plt.yticks(range(len(sorted_importance)), feature_names[sorted_indices])  # Set y-axis labels
        plt.xlabel('MDI Importance')
        plt.ylabel('Features')  # Reversed the labels for horizontal bar chart
        plt.title('Feature Importance in Random Forest Regression')
        
        # Add importance labels to the bars
        for i, v in enumerate(sorted_importance):
            plt.text(v, i, f'{v:.4f}', color='black', va='center', ha='left', fontweight='normal')

        plt.tight_layout()
        plt.show()

        # mdi_importance = sklearn_rf_model.feature_importances_
        
        # # Sort feature importance scores in descending order
        # sorted_indices = np.argsort(mdi_importance)[::-1]
        # sorted_importance = mdi_importance[sorted_indices]
        # feature_names = self.X_test.columns  # If you have column names for your features

        # # Bar plot to visualize feature importance
        # sns.set()
        # plt.figure(figsize=(10, 6))
        # plt.barh(range(len(sorted_importance)), sorted_importance)
        # plt.xticks(range(len(sorted_importance)), feature_names[sorted_indices], rotation=90)
        # plt.xlabel('Features')
        # plt.ylabel('MDI Importance')
        # plt.title('Feature Importance in Random Forest Regression')
        # plt.tight_layout()
        # plt.show()
        

        # imp = self.permutation_importances(sklearn_rf_model, self.X_train, self.y_train, oob_regression_r2_score)

        
        # ----- Print RF Tree -----
        # from sklearn import tree
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
        # plt.savefig('plotted_tree.pdf')
        # plt.show()

        # ----- Feature Importances -----



        # self.perm_importance(sklearn_rf_model)
        self.mdi_importance(sklearn_rf_model)
        # self.sfs_selection(sklearn_rf_model)
        self.rf_evaluation(sklearn_rf_model)
        # self.validation_score()
        # self.learning_curv(sklearn_rf_model)
        # self.oob_curv(sklearn_rf_model)
        self.rf_model = sklearn_rf_model
        return sklearn_rf_model
    
    # def learning_curv(self, the_model):
        # train_sizes, train_scores, test_scores = learning_curve(
        #     the_model, self.X_train, self.y_train)
        # display = LearningCurveDisplay(train_sizes=train_sizes,
        #                                train_scores=train_scores, 
        #                                test_scores=test_scores, 
        #                                score_name="Score")
        # display.plot()
        # plt.show()
    
    def oob_curv(self, rf_model):
        oob_errors = 1 - rf_model.oob_score_
        print(oob_errors)
        # Create lists to store the number of trees and corresponding OOB errors
        n_estimators_range = range(1, 99)
        oob_errors = []

        # Iterate through different numbers of trees and calculate OOB errors
        for n_estimators in n_estimators_range:
            rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, random_state=42)
            rf.fit(self.X_train, self.y_train)
            oob_error = 1 - rf.oob_score_
            oob_errors.append(oob_error)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_range, oob_errors, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of Trees in the Forest')
        plt.ylabel('OOB Error')
        plt.title('OOB Error vs. Number of Trees in Random Forest')
        plt.grid(True)
        plt.savefig('ICESAT/plot/oob_plot.png')
        plt.show()


    def ranger_RFregression(self):
        self.test_train()
        # ----- Ranger Regressor -----
        ranger_rf_model = RangerForestRegressor(n_jobs=-1, importance='impurity')
        # Fitting the Random Forest Regression model to the data
        ranger_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(ranger_rf_model)
        self.rf_model = ranger_rf_model
        
        return ranger_rf_model

    def xgboost_RFregression(self):
        self.test_train()
        # ----- XGBoost Regressor -----
        xgb_rf_model = xgb.XGBRegressor(n_jobs=-1, objective="reg:quantileerror")

        # [XGB] Fitting the RF Regression model to the data
        xgb_rf_model.fit(self.X_train, self.y_train)
        self.rf_evaluation(xgb_rf_model)
        self.rf_model = xgb_rf_model
        return xgb_rf_model

    def combine_rf_models(model1, model2, model3, X_test, y_test):
        predictions1 = model1.predict(X_test)
        predictions2 = model2.predict(X_test)
        predictions3 = model3.predict(X_test)

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

        # --- Plot Residuals ----
        sns.set()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) 
        
        # --- Plot Pred v Ground ----
        sns.set_context("notebook")
        ax1.scatter(y_pred, self.y_test, c='b', marker='.')
        # Add labels and title
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Ground Values')
        ax1.set_title('Scatter Plot of Ground vs. Predicted')
        # Add a 45-degree reference line (optional)
        ax1.plot([np.min(self.y_test), np.max(self.y_test)], [np.min(self.y_test), np.max(self.y_test)], 'r--')
        ax1.grid(True)
        
        # --- Plot Residuals ----
        residuals = self.y_test - y_pred
        ax2.scatter(y_pred, residuals, c="b", marker=".", label="Residuals")
        ax2.axhline(y=0, color="red", linestyle="-", label="Zero Residual Line")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot for Random Forest Regression")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig('ICESAT/plot/residuals_au3.png')
        plt.show()
        plt.close()


    def save_rfmodel(self, rf_modelname):
        # save the model to disk
        joblib.dump(self.rf_model, rf_modelname)

    def sfm_selection(self, rf_model):
        """Select From Model

        Args:
            rf_model (_type_): _description_
        """
        sfm = SelectFromModel(rf_model, threshold="mean")
        pass

    def sfs_selection(self, rf_model):
        print("SFS Features Selection:")
        sfs = SFS(
            rf_model,
            k_features="best",
            forward=False,
            floating=False,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        sfs = sfs.fit(self.x_ml, self.y_ml)
        selected_feature_indices = sfs.k_feature_idx_
        elected_features = self.X_train.columns[list(selected_feature_indices)]

        print(elected_features)

        fig = plot_sfs(sfs.get_metric_dict(confidence_interval=0.95), kind="std_dev")

        # Customize the plot (optional)
        plt.title("Sequential Forward Selection (SFS)")
        plt.grid()
        plt.savefig("ICESAT/sfs_chart.png")
        plt.show()

        print(sfs.k_feature_idx_)
        print(sfs.k_feature_names_)

        return None

    def output_tif(self, outname):
        pred_h = pd.DataFrame(self.grid_raster, columns=self.X_train.columns)
        pred_h.dropna(axis=1, inplace=True)

        rf_pred_target = self.rf_model.predict(pred_h)
        pred_h["pred_h"] = rf_pred_target
        
        latlon_id = self.grid_raster[["lat", "lon"]]
        latlon_pred = pred_h[["pred_h"]]
        
        print(latlon_id)
        print(latlon_pred)
        
        latlon_h = pd.concat([latlon_id, latlon_pred], axis=1, join='inner')
        
        print(latlon_h)

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
    #             cross_val_score(knn, X, y, cv=10)


def dist_to_pts(icepts, gridpts, where=None):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.dist_to("icepts", where)
    else:
        return d.dist_to("gridpts", where)

def height_to_pts(icepts, gridpts):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.height_to("icepts")
    else:
        return d.height_to("gridpts")


def angle_to_pts(icepts, gridpts, where=None):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.angle_to("icepts")
    else:
        return d.angle_to("gridpts")


def relativeh_to_pts(icepts, gridpts, where=None):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.relativeh_to("icepts")
    else:
        return d.relativeh_to("gridpts")


def slope_to_pts(icepts, gridpts, where=None):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.slope_to("icepts", where)
    else:
        return d.slope_to("gridpts", where)

def centroid_to_pts(icepts, gridpts, where=None):
    d = Geometry(icepts, gridpts)
    if gridpts is icepts:
        return d.centroid_to("icepts", where)
    else:
        return d.centroid_to("gridpts", where)


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
