"""
HMC analysis tools - Extract total basins rainfall
__date__ = '20210318'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'HMC analysis tools'
General command line:
### python hmc_tool_extract_rainfall_basin.py -settings_file setting.json -time "YYYY-MM-DD HH:MM"
Version(s):
20210318 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
from pysheds.grid import Grid
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import rasterio as rio
import json
import time
from argparse import ArgumentParser
import logging
from datetime import datetime, timedelta


def main():

    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'HMC analysis tools - Extract total basins rainfall '
    alg_version = '1.0.0'
    alg_release = '2021-03-18'
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, alg_time = get_args()

    dateRun = datetime.strptime(alg_time, "%Y-%m-%d %H:%M")

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(logger_file=os.path.join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info(' ============================================================================ ')
    logging.info(' ==> START ... ')
    logging.info(' ')

    # Time algorithm information
    start_time = time.time()
    # -------------------------------------------------------------------------------------
    logging.info(' --> Set algorithm time...')

    timeStart = dateRun - timedelta(hours=data_settings['data']['dynamic']['time']['time_observed_period_h']-1)
    timeEnd = dateRun + timedelta(hours=data_settings['data']['dynamic']['time']['time_forecast_period_h'])

    time_range=pd.date_range(timeStart, timeEnd, freq=data_settings['data']['dynamic']['time']['time_frequency'])

    logging.info(' --> Import static data...')
    dirmap_HMC = [8, 9, 6, 3, 2, 1, 4, 7]

    logging.info(' ---> Hydraulic pointers...')
    grid = Grid.from_ascii(data_settings["data"]["static"]["pointers"], data_name='dir')
    grid_spec = xr.open_rasterio(data_settings["data"]["static"]["pointers"])
    logging.info(' ---> Areacell...')
    areacell = rio.open(data_settings["data"]["static"]["areacell"]).read(1)
    logging.info(' ---> Sections...')
    tabular = pd.read_csv(data_settings["data"]["static"]["section_file"], sep="\s+", header=None)
    rHMC, cHMC, basin_name, section_name = tabular.values[:,0], tabular.values[:,1], tabular.values[:,2], tabular.values[:,3]
    logging.info(' --> Import static data...DONE')

    section_name = [i.replace("-","") for i in section_name]

    spatial_daily = {}
    for var in data_settings["data"]["dynamic"]["gridded"]["nc_var"].keys():
        spatial_daily[var] = pd.DataFrame(index=time_range, columns=section_name)

    os.makedirs(data_settings['data']['outcome']['folder'], exist_ok=True)

    logging.info(' --> Delineate basins masks...')
    # mask definition
    for ix, iy, basin, name in zip(cHMC, rHMC, basin_name, section_name):
        logging.info(' ---> section: ' + name)
        grid.catchment(data=grid.dir, x=ix-1, y=iy-1, dirmap=dirmap_HMC, out_name= 'basin_' + name,
                       recursionlimit=15000, nodata_out=0, ytype='index')

        if data_settings["algorithm"]["flags"]["plot_basin"] is True:
            logging.info(' --> Plot basin...')
            catch = grid.view('basin_' + name, nodata=np.nan)
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_alpha(0)

            boundaries = ([0] + sorted(list(dirmap_HMC)))
            plt.grid('on', zorder=0)
            im = ax.imshow(catch, extent=grid.extent, zorder=1, cmap='viridis')
            plt.colorbar(im, ax=ax, boundaries=boundaries, values=sorted(dirmap_HMC), label='Flow Direction')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(basin + '_' + name)

            if not os.path.isdir(os.path.join(data_settings['data']['outcome']['folder'],'figure')):
                os.makedirs(os.path.join(data_settings['data']['outcome']['folder'],'figure'), exist_ok=True)
            plt.savefig(os.path.join(data_settings['data']['outcome']['folder'], 'figure','basin_' + name + '.png'), bbox_inches='tight')
            plt.close()
    logging.info(' --> Delineate basins masks...DONE')

    logging.info(' --> Compute spatial features...')
    template = data_settings["algorithm"]["template"]
    for timeNow in time_range:
        logging.info(' ---> Time : ' + timeNow.strftime('%Y-%m-%d %H:%M'))
        template_now = {}
        for key in template.keys():
            try:
                template_now[key] = timeNow.strftime(template[key])
            except:
                template_now[key] = template[key]

        file_template = os.path.join(data_settings["data"]["dynamic"]["gridded"]["folder"], data_settings["data"]["dynamic"]["gridded"]["filename"])
        file_time_now = file_template.format(**template_now)

        logging.info(' ---> Import gridded rainfall...')
        if data_settings["algorithm"]["flags"]["zipped_outcome"] is True:
            try:
                os.system('yes y | gunzip ' + file_time_now + '.gz')
            except:
                pass

        for var in data_settings["data"]["dynamic"]["gridded"]["nc_var"].keys():
            if var == "ET":
                var_name = "ETcum"
            else:
                var_name = var

            map = xr.open_dataset(file_time_now).squeeze().reindex({data_settings["data"]["dynamic"]["gridded"]["nc_lon"]:grid_spec.x.values, data_settings["data"]["dynamic"]["gridded"]["nc_lat"]:grid_spec.y.values}, method='nearest')[var_name].values

            logging.info(' ---> Compute over basin mask...')
            for name in section_name:
                mask = eval('np.where(grid.basin_' + name  + '>0, 1, np.nan)')
                if data_settings["data"]["dynamic"]["gridded"]["nc_var"][var] == "sum":
                    spatial_daily[var].loc[timeNow][name] = np.nansum(map*mask).astype('float32')
                elif data_settings["data"]["dynamic"]["gridded"]["nc_var"][var] == "average":
                    spatial_daily[var].loc[timeNow][name] = np.nanmean(map*mask).astype('float32')
                else:
                    logging.error(" ERROR! Only sum or average are allowed on the variables!")
                    raise NotImplementedError

    logging.info(' --> Compute spatial features...DONE')

    logging.info(' --> Save output...')
    os.makedirs(os.path.join(data_settings['data']['outcome']['folder'], 'tabs'), exist_ok=True)
    for var in data_settings["data"]["dynamic"]["gridded"]["nc_var"]:
        spatial_daily[var].to_csv(os.path.join(data_settings['data']['outcome']['folder'], 'tabs',["data"]["dynamic"]["gridded"]["nc_var"][var] + '_' + var + '.txt'))


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
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time', action="store", dest="alg_time")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time:
        alg_time = parser_values.alg_time
    else:
        alg_time = None

    return alg_settings, alg_time
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

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------