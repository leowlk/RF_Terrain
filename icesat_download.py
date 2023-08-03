import numpy as np
import icepyx as ipx
from icepyx.core.visualization import Visualize
import os
import shutil
import matplotlib.pyplot as plt
from pyproj import Transformer
import h5py
import pandas as pd
from geopy.distance import geodesic

import glob
import sys


class DownloadICESAT:
    def __init__(self, short_name="ATL08", date_range=["2015-01-01", "2023-07-01"]):
        self.short_name = short_name
        self.date_range = date_range

    def input_bbox(self, bbox):
        self.minx, self.miny, self.maxx, self.maxy = bbox
        region = ipx.Query(self.short_name, bbox, self.date_range)
        self.region = region
        self.product = region.product
        self.dates = region.dates
        self.start_time = region.start_time
        self.end_time = region.end_time
        self.cycles = region.cycles
        self.tracks = region.tracks
        self.product_version = region.product_version
        self.spatial_extent = region.spatial_extent
        self.visualise = region.visualize_spatial_extent()
        self.viz = Visualize(region)
        self.CMRparams = region.CMRparams

    def download(self, location, *args):
        mapping = {
            "zlimburg": "pass",
            "nzealand": "pass",
            "gcanyon": "pass",
        }
        mapping[location](*args)

    def earthdata_download(self, fpath, *args, **kwargs):
        earthdata_uid = "lkan03"
        email = "leo.kan01@gmail.com"
        self.region.earthdata_login(earthdata_uid, email)
        self.all_info = self.region.product_all_info

        print("----------------\nAvailable granules to download:")
        print(self.region.avail_granules())
        user_input = input("----------------\nOrder and download graduals? (y/n) ")
        if user_input.lower() == "y":
            self.region.order_granules()
            self.region.download_granules(fpath)
        else:
            sys.exit()

    def read_h5(fname, bbox=None):
        """Read 1 ATL08 file and output 6 reduced files.
        Extract variables of interest and separate the ATL08 file
        into each beam (ground track) and ascending/descending orbits.
        """

        # Each beam is a group
        group = ["/gt1l", "/gt1r", "/gt2l", "/gt2r", "/gt3l", "/gt3r"]

        # Loop trough each beams
        for k, g in enumerate(group):
            try:
                # -----------------------------------#
                # 1) Read in data for a single beam #
                # -----------------------------------#
                # Load variables into memory (more can be added!)
                with h5py.File(fname, "r") as fi:
                    lat = fi[g + "/land_segments/latitude"][:]
                    lon = fi[g + "/land_segments/longitude"][:]
                    h_te_interp = fi[g + "/land_segments/terrain/h_te_interp"][:]
                    h_te_best_fit = fi[g + "/land_segments/terrain/h_te_best_fit"][:]
                    h_uncertainty = fi[g + "/land_segments/terrain/h_te_uncertainty"][:]
                    # terrain_slope = fi[g+'/land_segments/terrain/terrain_slope'][:]
                    # terrain_flg = fi[g+'/land_segments/terrain_flg'][:]
                    # photon_rate_te = fi[g+'/land_segments/terrain/photon_rate_te'][:]
                    # canopy_h_metrics = fi[g+'/land_segments/canopy/canopy_h_metrics'][:]
                    # canopy_openness = fi[g+'/land_segments/canopy/canopy_openness'][:]
                    # h_canopy_quad = fi[g+'/land_segments/canopy/h_canopy_quad'][:]

                    # ---------------------------------------------#
                    # 2) Filter data according region and quality #
                    # ---------------------------------------------#
                    # Select a region of interest
                    if bbox:
                        lonmin, lonmax, latmin, latmax = bbox
                        bbox_mask = (
                            (lon >= lonmin)
                            & (lon <= lonmax)
                            & (lat >= latmin)
                            & (lat <= latmax)
                        )
                    else:
                        bbox_mask = np.ones_like(lat, dtype=bool)  # get all
                    # save as csv
                    ofilecsv = fname.replace(".h5", "_" + g[1:] + ".csv")
                    result = pd.DataFrame()
                    result["lat"] = lat
                    result["lon"] = lon
                    result["h_te_interp"] = h_te_interp
                    result["h_te_best_fit"] = h_te_best_fit
                    result["h_te_uncertainty"] = h_uncertainty

                    # result['terrain_slope'] = terrain_slope
                    # result['terrain_flg']  = terrain_flg
                    # result['photon_rate_te'] = photon_rate_te

                    print("out ->", ofilecsv)
                    result.to_csv(ofilecsv, index=None)
            except:
                continue

    def read_multi_h5(self, fpath):
        dfs = []
        # for root_dir, sub_dir, files in os.walk(r'' + fpath):
        for root_dir, sub_dir, files in os.walk(fpath):
            for file in files:
                if file.endswith("h5"):
                    file_name = os.path.join(root_dir, file)
                    DownloadICESAT.read_h5(file_name, None)

    def combine_csv(self, fpath, csv_name):
        os.chdir(fpath)
        extension = "csv"
        all_filenames = [i for i in glob.glob(f"*.{extension}")]
        # combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

        # export to csv
        combined_csv.to_csv(csv_name + ".csv", index=False, encoding="utf-8-sig")
        print("combine csv complete!")
        print(f"{fpath}/{csv_name}.csv")

    def download_icesat_data(self, fpath, csv_name):
        self.region.product_summary_info()
        self.region.avail_granules()
        print("\n--------------\n")
        user_input = input("download bbox? (y/n) ")
        if user_input == "y":
            self.earthdata_download(fpath)
            self.read_multi_h5(fpath)
            self.combine_csv(fpath, csv_name)
        else:
            sys.exit()


def calc_area(boundbox):
    min_lon, min_lat, max_lon, max_lat = boundbox
    width = geodesic((min_lat, min_lon), (min_lat, max_lon)).kilometers
    height = geodesic((min_lat, min_lon), (max_lat, min_lon)).kilometers
    area = width * height
    return area


def _test():
    # t = DownloadICESAT()
    # # bounding = [173.719385, -39.675209, 174.661462, -38.967673]  # area = 6349.11 km2
    # # zlimburg = [173.9661, -39.3599, 174.4213, -39.0109]
    zu_limburg = [5.58311, 50.74222, 6.16390, 51.08249]
    gr_canyon = [-112.21907, 35.97509, -111.76801, 36.31694]

    # # bb = calc_area(zlimburg)

    # t.download("zlimburg")
    # t.input_bbox(zu_limburg)
    # t.earthdata_download()

    fpath = "/Users/leokan/Documents/TUDelft/icesatt/raw_data/nz2"
    nz = DownloadICESAT(nzbbox)
    # nz.download_icesat_data(fpath, 'nz_taranaki_combined')
    # print(f'>>> DONE!')

    # user_input = input('download bbox? (y/n)')
    # nz.download_icesat_data(fpath=fpath, csv_name='tmp_combined')
    # nz.downloadd_data(fpath)
    nz.read_multi_h5("/Users/leokan/Documents/TUDelft/ICEpyx/raw_data/nz2")
    nz.combine_csv(
        fpath="/Users/leokan/Documents/TUDelft/ICEpyx/raw_data/nz2",
        csv_name="nz2_combined_cop",
    )


if __name__ == "__main__":
    _test()

    # nzbbox = [173.9661, -39.3599, 174.4213, -39.0109]
    # # nzbbox2 = [-40, 173, -38, 176]
    # fpath = "/Users/leokan/Documents/TUDelft/icesatt/raw_data/nz2"
    # nz = DownloadICESAT(nzbbox)
    # # nz.download_icesat_data(fpath, 'nz_taranaki_combined')
    # # print(f'>>> DONE!')

    # # user_input = input('download bbox? (y/n)')
    # # nz.download_icesat_data(fpath=fpath, csv_name='tmp_combined')
    # # nz.downloadd_data(fpath)
    # nz.read_multi_h5("/Users/leokan/Documents/TUDelft/ICEpyx/raw_data/nz2")
    # nz.combine_csv(
    #     fpath="/Users/leokan/Documents/TUDelft/ICEpyx/raw_data/nz2",
    #     csv_name="nz2_combined_cop",
    # )
