import interpolation
import pandas as pd

def main():
    data = pd.read_csv("tasmania_test/samples.xyz", sep=' ')
    data = data.to_numpy()

    res = 100
    outname = "tasmania_test/Tasmania_test_aidw.tif"
    interpolation.aidw_interp(data, res, outname, n_neighbours=15)
    
    outname = "tasmania_test/Tasmania_test_idw.tif"
    interpolation.idw_interp(data, res, outname, power=2, n_neighbours=15)

if __name__ == "__main__":
    main()