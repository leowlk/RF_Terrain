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

class DownloadICESAT():
    def __init__(self, bbox, short_name="ATL08", date_range=["2015-01-01", "2023-07-01"]) -> None:
        self.minx, self.miny, self.maxx, self.maxy = bbox
        self.bbox = bbox
        self.short_name = short_name
        self.date_range = date_range
        
    def download_data_to_file(self, fpath, *args, **kwargs):
        region = ipx.Query(self.short_name, self.bbox, self.date_range)
        earthdata_uid = "lkan03"
        email = "leo.kan01@gmail.com"
        region.earthdata_login(earthdata_uid, email)
        # all_info = region.product_all_info
        print("---\nAvailable granules to download:")
        print(region.avail_granules())
        user_input = input("---\nOrder and download graduals? (y/n) ")
        if user_input.lower() == "y":
            region.order_granules()
            region.download_granules(fpath)
        else:
            sys.exit()
            
            
def read_h5(fname, bbox=None):
    """_summary_

    Args:
        fname (_type_): _description_
        bbox (_type_, optional): _description_. Defaults to None.
    """
    
    # Each beam is a group
    group = ["/gt1l", "/gt1r", "/gt2l", "/gt2r", "/gt3l", "/gt3r"]

    # Loop trough each beams
    for k, g in enumerate(group):
        try:
            # 1) Read in data for a single beam #
            # Load variables into memory (more can be added!)
            with h5py.File(fname, "r") as fi:
                lat = fi[g + "/land_segments/latitude"][:]
                lon = fi[g + "/land_segments/longitude"][:]
                h_te_interp = fi[g + "/land_segments/terrain/h_te_interp"][:]
                h_te_best_fit = fi[g + "/land_segments/terrain/h_te_best_fit"][:]
                h_uncertainty = fi[g + "/land_segments/terrain/h_te_uncertainty"][:]

            # 2) Filter data according region and quality #
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

                print("out ->", ofilecsv)
                result.to_csv(ofilecsv, index=None)
        except:
            continue

def multi_read_H5(filepath):
    dfs = []
    # for root_dir, sub_dir, files in os.walk(r'' + fpath):
    for root_dir, sub_dir, files in os.walk(filepath):
        for file in files:
            if file.endswith("h5"):
                file_name = os.path.join(root_dir, file)
                read_h5(file_name, None)
                
def combine_csv(fpath, csv_name="combined.csv"):
    os.chdir(fpath)
    extension = "csv"
    all_filenames = [i for i in glob.glob(f"*.{extension}")]
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

    # export to csv
    combined_csv.to_csv(csv_name + ".csv", index=False, encoding="utf-8-sig")
    print("combine csv complete!")


def download_data(bbox, filepath):
    i = DownloadICESAT(bbox)
    i.download_data_to_file(filepath)
    
def main():
    tasmania = [147.0003094193868,-42.91559892536544,147.39115062433459,-42.595618448793466]
 
    download_data(tasmania, "ICESAT/tasmania")
    multi_read_H5("ICESAT/tasmania")
    combine_csv("ICESAT/tasmania")

if __name__ == "__main__":
    main()



