import numpy as np
import pandas as pd
import startinpy
from pyproj import CRS, Transformer
from itertools import product, permutations
import xarray as xr
import rioxarray
import rasterio
import math
from pyinterpolate import inverse_distance_weighting

# --
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree

def EPSG_transformer(in_epsg, out_epsg, pts_list):
    trans = Transformer.from_crs(in_epsg, out_epsg, always_xy=True)
    pts_list = [trans.transform(lon, lat) for lat, lon in pts_list]
    return pts_list

class Interp:
    def __init__(self, pts_in_3d, res) -> None:
        """_summary_

        Args:
            pts_in_3d (numpy.ndarray): _description_
            res (int): _description_
        """
        self.pts_in_3d = pts_in_3d
        self.dt = startinpy.DT()
        self.dt.insert(self.pts_in_3d, insertionstrategy="BBox")

        match res:
            case 30:
                self.resolution = 0.0002778
            case 100:
                self.resolution = 0.0009166
            case 200:
                self.resolution = 0.0019444
            case 300:
                self.resolution = 0.0027777
            case 400:
                self.resolution = 0.0036111
            case 500:
                self.resolution = 0.0044444
            case _:
                self.resolution = 0.0009166

    def projected_res(self, epsg=4326):
        trans = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

    def make_grid(self, data_in_3d):
        """Create raster grid from 3D coordinate points

        Args:
            data_in_3d (np.ndarray): input 3D points in np.ndarray with columns=[lat, lon]
            res (int): resolution based on EPSG:4326

        Returns:
            np.ndarray:
        """
        x_min, x_max = data_in_3d[:, 0].min(), data_in_3d[:, 0].max()
        y_min, y_max = data_in_3d[:, 1].min(), data_in_3d[:, 1].max()

        x_grid = np.arange(x_min, x_max, self.resolution)
        y_grid = np.arange(y_min, y_max, self.resolution)
        d = product(x_grid, y_grid)
        
        grid_raster = pd.DataFrame(data=d, columns=["lat", "lon"]).to_numpy()

        return grid_raster

    def startin(self):
        self.dt = startinpy.DT()
        self.dt.insert(self.pts_in_3d, insertionstrategy="BBox")

    def tin(self):
        self.startin()
        grid_raster = self.make_grid(self.pts_in_3d)

        tin_data = []

        for n in grid_raster:
            if self.dt.is_inside_convex_hull(n[0], n[1]) == True:
                data = [n[0], n[1], self.dt.interpolate_tin_linear(n[0], n[1])]
                tin_data.append(data)
            else:
                data = [n[0], n[1], np.nan]
                tin_data.append(data)

        self.tin_p = pd.DataFrame(tin_data, columns=["lat", "lon", "interp_h"])
        tin_h = xr.Dataset.from_dataframe(self.tin_p.set_index(["lat", "lon"]))
        return tin_h

    def laplace(self):
        self.startin()
        grid_raster = self.make_grid(self.pts_in_3d)

        lp_data = []
        for n in grid_raster:
            if self.dt.is_inside_convex_hull(n[0], n[1]) == True:
                data = [n[0], n[1], self.dt.interpolate_laplace(n[0], n[1])]
                lp_data.append(data)
            else:
                data = [n[0], n[1], np.nan]
                lp_data.append(data)
        self.lp_p = pd.DataFrame(lp_data, columns=["lat", "lon", "interp_h"])
        lp_h = xr.Dataset.from_dataframe(self.lp_p.set_index(["lat", "lon"]))
        return lp_h

    def nni(self):
        self.startin()
        grid_raster = self.make_grid(self.pts_in_3d)

        nni_data = []

        for n in grid_raster:
            if self.dt.is_inside_convex_hull(n[0], n[1]) == True:
                try:
                    data = [n[0], n[1], self.dt.interpolate_nni(n[0], n[1])]
                except:
                    data = [n[0], n[1], np.nan]
                # except:
                #     print(f'{n[0]}, {n[1]}')
                #     print(self.dt.is_inside_convex_hull(n[0], n[1]))
                nni_data.append(data)
            else:
                data = [n[0], n[1], np.nan]
                nni_data.append(data)

        self.nni_p = pd.DataFrame(nni_data, columns=["lat", "lon", "interp_h"])
        nni_h = xr.Dataset.from_dataframe(self.nni_p.set_index(["lat", "lon"]))
        return nni_h

    def idw(self, power, n_neighbours):
        grid_raster = self.make_grid(self.pts_in_3d)
        idw_data = []
        for n in grid_raster:
            data = [
                n[0],
                n[1],
                inverse_distance_weighting(
                    known_points=self.pts_in_3d,
                    unknown_location=n,
                    number_of_neighbours=n_neighbours,
                    power=power,
                ),
            ]
            idw_data.append(data)

        self.idw_p = pd.DataFrame(idw_data, columns=["lat", "lon", "interp_h"])
        idw_h = xr.Dataset.from_dataframe(self.idw_p.set_index(["lat", "lon"]))
        return idw_h
    
    def calc_angle_between(self, P, p1, p2):            
        edge1 = p1-P
        edge2 = p2-P
                        
        mag1 = np.sqrt(edge1.dot(edge1))
        mag2 = np.sqrt(edge2.dot(edge2))
        
        # dot product to find angle between two vectors
        cos_T = math.acos(edge1.dot(edge2) / (mag1 * mag2))
        return cos_T
    
    def calc_bisect_angle(self, P, p1, p2):
        mid_point = [(p1[0]+p2[0]) /2, (p1[1]+p2[1]) /2]        
        theta = self.calc_angle_between(mid_point, P, p1)
        return theta

        
    def aidw(self, n_neighbours):
        """_summary_

        Args:
            n_neighbours (_type_): _description_
        """
        pts_in_3d = self.pts_in_3d[:,:2] 
        pts_in_3d = np.unique(pts_in_3d, axis=0)

        grid_raster = self.make_grid(self.pts_in_3d)

        # pts_in_2d = EPSG_transformer('EPSG:4326', 'EPSG:5551', pts_in_2d)
        # grid_raster = EPSG_transformer('EPSG:4326', 'EPSG:5551', grid_raster)
        # pts_in_2d = np.array(pts_in_2d)
        # grid_raster = np.array(grid_raster)
        
        power = 2

        kd = KDTree(pts_in_3d) # [[x y z]...[]]

        aidw_data = []
        
        for grid_point in grid_raster:
            dist, ind = kd.query(grid_point, k=n_neighbours, p=2) # n_neighbour
            neighbour_icepts_list = pts_in_3d[ind] # [[...]  [...]  [...]]
            
            heights = self.pts_in_3d[ind][:,2]
            
            # print(grid_point)
            
            # print(heights)
            
            # print(dist) # [1204.93815075 1269.83783281 2249.61346113 2255.46108436 2328.58885744]
            
                                    
            # Finding pairs of points in the icepts list
            sin_theta = [1]
            for i in range(len(neighbour_icepts_list) - 1):
                p0 = neighbour_icepts_list[i]
                p1 = neighbour_icepts_list[i+1] 
                              
                A = self.calc_angle_between(grid_point, p0, p1)
                T = self.calc_bisect_angle(grid_point, p0, p1)
                                
                if A < 360/n_neighbours:
                    sin_T = (1 - np.cos(T)**2)** (power/2)
                    sin_theta.append(sin_T)
                else:
                    sin_T = 1
                    sin_theta.append(sin_T)
                    
            
            sum_height = [k / d ** power * h for d, h, k in zip(dist, heights, sin_theta)]
            
            sum_weight = [k * d ** (-power) for d, k in zip(dist, sin_theta)]
            
            
            s = sum(sum_height) / sum(sum_weight)
            
            aidw_data.append([grid_point[0], grid_point[1], s])
            
        self.aidw_p = pd.DataFrame(aidw_data, columns=["lat", "lon", "interp_h"])
        aidw_h = xr.Dataset.from_dataframe(self.aidw_p.set_index(["lat", "lon"]))
        return aidw_h
            
        

    def output(self, x_arr, *output_file):
        # +++[ Drop np.nan columns/rows and inputunate data (filling nan values) ]+++
        x_arr = x_arr.dropna("lon", how="all")
        imputer = KNNImputer(n_neighbors=10)
        tin_arr = imputer.fit_transform(x_arr["interp_h"].values)
        tin_xarr = xr.DataArray(tin_arr, dims=("lat", "lon"))
        x_arr["interp_h"] = tin_xarr
        # ++++++++++++++++++++++++++++
        x_arr.rio.set_crs("epsg:4326")
        x_arr.rio.set_spatial_dims("lon", "lat")
        x_arr.rio.to_raster(*output_file, driver="GTiff")


def laplace_interp(data, res, outname):
    i = Interp(data, res)
    arr = i.laplace()
    i.output(arr, outname)


def nni_interp(data, res, outname):
    i = Interp(data, res)
    xarr = i.nni()
    i.output(xarr, outname)


def tin_interp(data, res, outname):
    i = Interp(data, res)
    xarr = i.tin()
    i.output(xarr, outname)


def idw_interp(data, res, outname, power=2, n_neighbours=10):
    i = Interp(data, res)
    arr = i.idw(power, n_neighbours)
    i.output(arr, outname)

def aidw_interp(data, res, outname, n_neighbours=10):
    i = Interp(data, res)
    arr = i.aidw(n_neighbours)
    i.output(arr, outname)
    breakpoint()

def _test():
    pass


if __name__ == "__main__":
    _test()
