# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HMC tools - Calibration Make Forcing

__date__ = '20220822'
__version__ = '1.0.0'
__author__ = 'Andrea Libertino (andrea.libertino@cimafoundation.org')
__library__ = 'HMC_calibration_tool'

General command line:
python3 HMC_calibration -settings_file "FILE.json"
20220822 (0.0.1) -->    Beta release single domain
"""
# -------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rioxr
import os, shutil, gzip
import datetime as dt
from netCDF4 import Dataset
import logging, json
from argparse import ArgumentParser
from datetime import date
import time
from copy import deepcopy
import sys

# -------------------------------------------------------------------------------------
# Algorithm information

alg_name = 'HMC tools - Merge forcings'
alg_version = '1.0.0'
alg_release = '2023-07-18'
# Algorithm parameter(s)
time_format = '%Y%m%d%H%M'


# -------------------------------------------------------------------------------------
# Script main

def main():
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, domain = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    if domain is None:
        domain = data_settings["algorithm"]["general"]["domain_name"]

    # Set algorithm logging
    os.makedirs(data_settings["algorithm"]["path"]["log"], exist_ok=True)
    set_logging(logger_file=os.path.join(data_settings["algorithm"]["path"]["log"], domain + "_merge_forcings.log"))

    # Set timing
    date_start = dt.datetime.strptime(data_settings["algorithm"]["time"]["date_start"], "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(data_settings["algorithm"]["time"]["date_end"], "%Y-%m-%d %H:%M")
    forcing_period = pd.date_range(start=date_start, end=date_end, freq=data_settings["algorithm"]["time"]["frequency"])

    # Set output path
    dir_out_generic = data_settings["algorithm"]["path"]["output"]
    ancillary_out = data_settings["algorithm"]["path"]["ancillary"].format(domain=domain)
    os.makedirs(ancillary_out, exist_ok=True)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Initialize
    logging.info(' ============================================================================ ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME : ' + date.today().strftime("%d-%B-%Y %H:%m"))
    logging.info(' ==> START ... ')
    logging.info(' ==> ALGORITHM SETTINGS <== ')
    logging.info(' --> Domain: ' + domain)

    logging.info(' ')

    start_time = time.time()

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Initialize static grid
    logging.info(" ---> Prepare static grid")
    dem_in = os.path.join(data_settings["data"]["input"]["data_static"]["folder_name"],
                          data_settings["data"]["input"]["data_static"]["dem"]).format(domain=domain)
    dem = xr.open_rasterio(dem_in)
    dem_grid = os.path.join(ancillary_out, domain + "_grid.nc")
    os.system("gdal_translate -of netcdf " + dem_in + " " + dem_grid)
    dem_value = np.squeeze(dem.values)
    coords = {}
    Lon = np.sort(dem.x.values)
    if not all(Lon == dem.x.values):
        dem_value = np.fliplr(dem_value)
    coords["lon"] = Lon
    Lat = np.sort(dem.y.values)
    if not all(Lat == dem.y.values):
        dem_value = np.flipud(dem_value)
    coords["lat"] = Lat
    [lon2d, lat2d] = np.meshgrid(Lon, Lat)
    logging.info(" ---> Prepare static grid... DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Initialise vars
    hmc_vars = data_settings["data"]["input"]["data_dynamic"]["hmc_vars"]
    all_vars = [i for i in hmc_vars.keys()]

    valid_vars = {}
    for var in all_vars:
        valid_vars[var] = None

    # Loop trough dates
    logging.info(" ---> Looping trought time steps")
    for time_now in forcing_period:
        logging.info(' ----> Time step: ' + time_now.strftime("%Y-%m-%d %H:%M"))

        template_filled = fill_template(data_settings["algorithm"], time_now)
        template_filled["domain"] = domain
        dir_out_now = dir_out_generic.format(**template_filled)

        check_exist = os.path.isfile(
            os.path.join(dir_out_now, "hmc.forcing-grid.{datetime_file_out}00.nc.gz").format(**template_filled))
        if check_exist is True and data_settings["general"]["skip_existing"]:
            logging.warning("WARNING! Forcing already exist...SKIP")
            continue

        maps = {}

        maps["lon"] = lon2d
        maps["lat"] = lat2d
        maps["dem"] = dem_value

        files_to_open = np.unique([hmc_vars[i]["file_path"] for i in hmc_vars.keys()])

        for file in files_to_open:
            vars_in = [i for i in hmc_vars.keys() if hmc_vars[i]["file_path"] == file]
            file_filled = file.format(**template_filled)

            if os.path.isfile(file_filled):
                logging.info(" --> Opening file " + file_filled)
                if hmc_vars[vars_in[0]]["zipped"]:
                    gunzip(file_filled, file_filled.replace(".gz", ""))

                if hmc_vars[vars_in[0]]["format"] == "netcdf":
                    ds = xr.open_dataset(file_filled.replace(".gz", ""))
                    dims_in = hmc_vars[vars_in[0]]["lat_lon"]
                elif hmc_vars[vars_in[0]]["format"] == "tif":
                    grid = xr.open_rasterio(file_filled.replace(".gz", ""))
                    dims_in = ["y", "x"]

                for var in vars_in:
                    logging.info(" --> Compute variable " + var)
                    if hmc_vars[vars_in[0]]["format"] == "netcdf":
                        if hmc_vars[var]["var_name"] not in ds.keys():
                            maps[var] = assign_last_valid(valid_vars, var, maps)
                            if maps[var] is None:
                                logging.error(" ERROR! Var " + hmc_vars[var][
                                    "var_name"] + "not found in " + file_filled + " and no previous value available!")
                                raise FileNotFoundError
                            else:
                                logging.error(" WARINING! Var " + hmc_vars[var][
                                    "var_name"] + "not found in " + file_filled + " and no previous value available!")
                                continue
                        else:
                            logging.info(" --> Extracting variable " + hmc_vars[var]["var_name"])
                            grid = ds[hmc_vars[var]["var_name"]]
                    if hmc_vars[vars_in[0]]["regrid"]:
                        logging.info(" --> Regrid var " + var)
                        grid = regrid_map(grid, dims_in, Lat, Lon)
                    maps[var] = np.nan_to_num(grid.values, nan=-9999)
                    valid_vars[var] = deepcopy(np.nan_to_num(grid.values, nan=-9999))
            else:
                for var in vars_in:
                    maps[var] = assign_last_valid(valid_vars, var, maps)
                    if maps[var] is None:
                        logging.error(" ERROR! File " + file_filled + "not found and no previous value available!")
                        raise FileNotFoundError
                    else:
                        logging.error(" WARINING! File " + file_filled + "not found, use last data available!")
            if hmc_vars[vars_in[0]]["zipped"]:
                try:
                    os.remove(file_filled.replace(".gz", ""))
                except FileNotFoundError:
                    pass

        try:
            flag_lai = data_settings["algorithm"]["general"]["include_LAI"]
        except:
            flag_lai = True

        create_forcing(dir_out_now, time_now, coords, maps, flag_lai, domain)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    time_elapsed = round(time.time() - start_time, 1)

    logging.info(' ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME ELAPSED: ' + str(time_elapsed) + ' seconds')
    logging.info(' ==> ... END')
    logging.info(' ==> Bye, Bye')
    logging.info(' ============================================================================ ')
    # -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def create_forcing(dir_out_now, time_now, coords, maps, flag_lai, domain_name):
    os.makedirs(dir_out_now, exist_ok=True)

    outMap = Dataset(os.path.join(dir_out_now, "hmc.forcing-grid." + time_now.strftime("%Y%m%d%H") + "00.nc"), "w",
                     format="NETCDF4")

    # Crea dimensioni
    lon = outMap.createDimension("lon", maps["lat"].shape[1])
    lat = outMap.createDimension("lat", maps["lat"].shape[0])
    time = outMap.createDimension("time", 1)

    # Crea variabili
    crs = outMap.createVariable("crs", "i", ("time",), chunksizes=[1])
    time = outMap.createVariable("time", "d", ("time",), chunksizes=[1])
    lat = outMap.createVariable("lat", "f", ("lat",))
    lon = outMap.createVariable("lon", "f", ("lon",))
    Longitude = outMap.createVariable("Longitude", "d", ("lat", "lon",),
                                      chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    Latitude = outMap.createVariable("Latitude", "d", ("lat", "lon",),
                                     chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    Terrain = outMap.createVariable("Terrain", "f", ("lat", "lon",),
                                    chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    AirTemperature = outMap.createVariable("AirTemperature", "f", ("lat", "lon",),
                                           chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    Rain = outMap.createVariable("Rain", "f", ("lat", "lon",), chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    IncRadiation = outMap.createVariable("IncRadiation", "f", ("lat", "lon",),
                                         chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    Wind = outMap.createVariable("Wind", "f", ("lat", "lon",), chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    RelHumidity = outMap.createVariable("RelHumidity", "f", ("lat", "lon",),
                                        chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])

    if flag_lai is True:
        if maps["LAI"] is not None:
            LAI = outMap.createVariable("LAI", "f", ("lat", "lon",),
                                        chunksizes=[maps["lat"].shape[0], maps["lat"].shape[1]])
    else:
        maps["LAI"] = None

    # Attibuti globali
    outMap.filename = "hmc.forcing-grid." + time_now.strftime("%Y%m%d%H") + "00.nc"
    outMap.filedate = time_now.strftime("%Y-%m-%d %H:%M:%S")
    outMap.domainname = domain_name
    outMap.timestep = int(3600)
    outMap.timenow = time_now.strftime("%Y-%m-%d %H:%M:%S")
    outMap.ncols = int(maps["lat"].shape[1])
    outMap.nrows = int(maps["lat"].shape[0])
    outMap.cellsize = coords["lon"][2] - coords["lon"][1]
    outMap.xllcorner = np.amin(coords["lon"]) - ((coords["lon"][2] - coords["lon"][1]) / 2)
    outMap.yllcorner = np.amin(coords["lat"]) - ((coords["lat"][2] - coords["lat"][1]) / 2)
    outMap.nodata_value = -9999
    outMap.comment = "Author(s): Andrea Libertino"
    outMap.project = "Africa multi-rain: " + domain_name
    outMap.references = "http:cf-pcmdi.llnl.gov/; http:Fcf-pcmdi.llnl.gov/documents/cf-standard-names/ecmwf-grib-mapping"
    outMap.website = "http://www.cimafoundation.org"
    outMap.institution = "CIMA Research Foundation - www.cimafoundation.org"
    outMap.algorithm = "HMC"
    outMap.title = "MeteoForcing HMC3"
    outMap.conventions = "CF-1.6"
    outMap.source = "ERA5 - Copernicus LAI"
    outMap.email = "andrea.libertino@cimafoundation.org"
    outMap.history = ""
    outMap.timeworldref = "TimeType:gmt;TimeSave:0;TimeLoad:1;"
    outMap.fileconfigdynamic = ""
    outMap.timeupd = ""

    # attributi crs
    crs.bounding_box = []
    crs.inverse_flattening = 298.2572
    crs.longitude_of_prime_meridian = 0
    crs.grid_mapping_name = "latitude_longitude"
    crs.semi_major_axis = 6378137

    # attributi time
    time[:] = 0
    time.calendar = 'gregorian'
    time.units = 'hours since ' + time_now.strftime("%Y-%m-%d %H:%M:%S")
    time.time_date = ''
    time.time_start = ''
    time.time_end = ''
    time.axis = 'T'

    # attributi lon
    lon[:] = coords["lon"]
    lon.grid_mapping = ''
    lon.coordinates = ''
    lon.cell_method = ''
    lon.pressure_level = ''
    lon.long_name = 'longitude'
    lon.standard_name = 'longitude';
    lon.units = 'degrees_east';
    lon.axis = 'X'
    lon.scale_factor = 1

    # attributi lat
    lat[:] = coords["lat"]
    lat.grid_mapping = ''
    lat.coordinates = ''
    lat.cell_method = ''
    lat.pressure_level = ''
    lat.long_name = 'latitude'
    lat.standard_name = 'latitude';
    lat.units = 'degrees_north';
    lat.axis = 'Y'
    lat.scale_factor = 1

    # attributi longitude
    Longitude[:] = maps["lon"]
    Longitude.grid_mapping = ''
    Longitude.coordinates = ''
    Longitude.cell_method = ''
    Longitude.pressure_level = ''
    Longitude.long_name = 'longitude coordinate'
    Longitude.standard_name = 'longitude_grid';
    Longitude.units = 'degrees_east';
    Longitude.scale_factor = 1

    # attributi latitude
    Latitude[:] = maps["lat"]
    Latitude.grid_mapping = ''
    Latitude.coordinates = ''
    Latitude.cell_method = ''
    Latitude.pressure_level = ''
    Latitude.long_name = 'latitude coordinate'
    Latitude.standard_name = 'latitude_grid';
    Latitude.units = 'degrees_north';
    Latitude.scale_factor = 1

    # attributi terrain
    Terrain[:] = maps["dem"]
    Terrain.grid_mapping = ''
    Terrain.coordinates = ''
    Terrain.cell_method = ''
    Terrain.pressure_level = ''
    Terrain.long_name = 'Terrain'
    Terrain.standard_name = 'Terrain';
    Terrain.units = 'm asl';
    Terrain.scale_factor = 1

    # attributi Rain
    Rain[:] = maps[
        "Rain"]  # np.nan_to_num(np.squeeze(precip.reindex({"lon": Lon, "lat": Lat}, method='nearest')).values, nan=-9999)
    Rain.grid_mapping = 'crs'
    Rain.coordinates = 'latitude longitude'
    Rain.cell_method = ''
    Rain.pressure_level = ''
    Rain.long_name = 'Rain'
    Rain.standard_name = 'Rain'
    Rain.units = 'mm'
    Rain.scale_factor = 1

    # attributi AirTemperature
    AirTemperature[:] = maps["AirTemperature"]
    AirTemperature.grid_mapping = 'crs'
    AirTemperature.coordinates = 'latitude longitude'
    AirTemperature.cell_method = ''
    AirTemperature.pressure_level = ''
    AirTemperature.long_name = 'AirTemperature'
    AirTemperature.standard_name = 'AirTemperature'
    AirTemperature.units = 'degree_C'
    AirTemperature.scale_factor = 1

    # attributi IncRadiation
    IncRadiation[:] = maps["IncRadiation"]
    IncRadiation.grid_mapping = 'crs'
    IncRadiation.coordinates = 'latitude longitude'
    IncRadiation.cell_method = ''
    IncRadiation.pressure_level = ''
    IncRadiation.long_name = 'IncRadiation'
    IncRadiation.standard_name = 'IncRadiation'
    IncRadiation.units = 'W/m^2'
    IncRadiation.scale_factor = 1

    # attributi Wind
    Wind[:] = maps["Wind"]
    Wind.grid_mapping = 'crs'
    Wind.coordinates = 'latitude longitude'
    Wind.cell_method = ''
    Wind.pressure_level = ''
    Wind.long_name = 'Wind'
    Wind.standard_name = 'Wind'
    Wind.units = 'm/s'
    Wind.scale_factor = 1

    # attributi RelHumidity
    RelHumidity[:] = maps["RelHumidity"]
    RelHumidity.grid_mapping = 'crs'
    RelHumidity.coordinates = 'latitude longitude'
    RelHumidity.cell_method = ''
    RelHumidity.pressure_level = ''
    RelHumidity.long_name = 'RelHumidity'
    RelHumidity.standard_name = 'RelHumidity'
    RelHumidity.units = '%'
    RelHumidity.scale_factor = 1

    if maps["LAI"] is not None:
        # attributi LAI
        LAI[:] = maps["LAI"]
        LAI.grid_mapping = 'crs'
        LAI.coordinates = 'latitude longitude'
        LAI.cell_method = ''
        LAI.pressure_level = ''
        LAI.long_name = 'LAI'
        LAI.standard_name = 'LAI'
        LAI.units = '-'
        LAI.scale_factor = 1

    outMap.close()
    os.system("gzip -f " + os.path.join(dir_out_now, "hmc.forcing-grid." + time_now.strftime("%Y%m%d%H") + "00.nc"))


# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Function for fill a dictionary of templates
def fill_template(downloader_settings, time_now):
    empty_template = downloader_settings["templates"]
    template_filled = {}
    for key in empty_template.keys():
        template_filled[key] = time_now.strftime(empty_template[key])
    return template_filled


# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-domain', action="store", dest="domain")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.domain:
        domain = parser_values.domain
    else:
        domain = None

    return alg_settings, domain


# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def regrid_map(grid, coord_names, lat_out, lon_out):
    if len(grid[coord_names[0]].values.shape) == 2:
        logging.info(" --> Variable coordinated are 2d matrix, reorganizing for reindexing")
        ds_temp = xr.DataArray(grid.values.squeeze(), dims=["y", "x"],
                               coords={"y": np.unique(grid[coord_names[0]].values),
                                       "x": np.unique(grid[coord_names[1]].values)})
    else:
        logging.info(" --> Variable coordinated are vectors")
        ds_temp = grid.rename({coord_names[0]: "y", coord_names[1]: "x"})
    return ds_temp.reindex({"x": lon_out, "y": lat_out}, method="nearest")


# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to read file json
def read_file_json(file_name):
    env_ws = {}
    for env_item, env_value in os.environ.items():
        env_ws[env_item] = env_value

    with open(file_name, "r") as file_handle:
        json_block = []
        for file_row in file_handle:

            for env_key, env_value in env_ws.items():
                env_tag = '$' + env_key
                if env_tag in file_row:
                    env_value = env_value.strip("'\\'")
                    file_row = file_row.replace(env_tag, env_value)
                    file_row = file_row.replace('//', '/')

            # Add the line to our JSON block
            json_block.append(file_row)

            # Check whether we closed our JSON block
            if file_row.startswith('}'):
                # Do something with the JSON dictionary
                json_dict = json.loads(''.join(json_block))
                # Start a new block
                json_block = []

    return json_dict


# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.INFO)

    # Open logging basic configuration
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.INFO)
    logger_handle_2.setLevel(logging.INFO)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)

    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)


# -------------------------------------------------------------------------------------
def gunzip(source_filepath, dest_filepath, skip_existing=True, block_size=65536):
    if os.path.isfile(dest_filepath) and skip_existing:
        logging.info(" --> Unzipepd file already exist, skip!")
    else:
        with gzip.open(source_filepath, 'rb') as s_file, open(dest_filepath, 'wb') as d_file:
            shutil.copyfileobj(s_file, d_file, block_size)


def assign_last_valid(valid_vars, var, maps):
    if valid_vars[var] is None:
        return None
    else:
        maps[var] = deepcopy(valid_vars[var])
        return maps[var]


# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------