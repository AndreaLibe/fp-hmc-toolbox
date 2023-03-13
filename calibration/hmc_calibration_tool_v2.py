#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HMC tools - Calibration Multi Domain

__date__ = '20220302'
__version__ = '1.1.1'
__author__ = 'Andrea Libertino (andrea.libertino@cimafoundation.org')
             'Lorenzo Campo (lorenzo.campo@cimafoundation.org')
             'Lorenzo Alfieri (lorenzo.alfieri@cimafoundation.org')
__library__ = 'HMC_calibration_tool'

General command line:
python3 HMC_calibration -settings_file "FILE.json"
20201130 (0.0.1) -->    Beta release single domain
20210226 (1.0.0) -->    Separate multi-domain branch with discharge-only support
20220302 (1.1.0) -->    Changed calibration approach
20220513 (1.1.1) -->    Bug fixes
"""
# -------------------------------------------------------------------------------------
# Complete library
import numpy as np
import os, pickle, math, time
import rasterio as rio
import pandas as pd
from pyDOE import lhs
from cdo import *
import logging
from datetime import date
from argparse import ArgumentParser
import json
import shutil
import warnings
from osgeo import gdal
import datetime as dt
import hydrostats as hs
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information

alg_name = 'HMC tools - Calibration'
alg_version = '1.1.1'
alg_release = '2022-05-13'
# Algorithm parameter(s)
time_format = '%Y%m%d%H%M'

# -------------------------------------------------------------------------------------
# Script main

def main():
    start_time = time.time()
    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    # Import common settings
    domain = data_settings['algorithm']['general']['domain_name']

    # Fill and import paths and make useful dirs
    path_settings = data_settings["algorithm"]["path"]

    # Set calibration settings
    iMin = data_settings['algorithm']['general']['start_with_iteration_number']
    iMax = iMin + data_settings['algorithm']['general']['max_number_of_iterations'] - 1
    nExplor = data_settings['algorithm']['general']['number_of_points_first_iteration']

    calibrated_params = [var for var in data_settings['calibration']['parameters'].keys() if data_settings['calibration']['parameters'][var]["calibrate"]]

    # Import time settings
    run_hydro_start = dt.datetime.strptime(data_settings["algorithm"]["time"]["run_hydro_start"], "%Y-%m-%d %H:%M")
    run_hydro_end = dt.datetime.strptime(data_settings["algorithm"]["time"]["run_hydro_end"], "%Y-%m-%d %H:%M")
    calib_hydro_start = dt.datetime.strptime(data_settings["algorithm"]["time"]["calib_hydro_start"], "%Y-%m-%d %H:%M")

    # Calibration approaches available
    implemented_approaches = {"rescale": rescale_map, "mask": rescale_mask, "uniform": rescale_value}
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Create folders, set logging and initialize algorithm
    # Create folders
    os.makedirs(path_settings["work_path"], exist_ok=True)
    os.makedirs(path_settings["out_path"], exist_ok=True)
    os.makedirs(path_settings['log_path'], exist_ok=True)

    # Set algorithm logging
    set_logging(logger_file = os.path.join(path_settings['log_path'], domain + "_calibration.log"))

    # Initialize
    logging.info(' ============================================================================ ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME : ' + date.today().strftime("%d-%B-%Y %H:%m"))
    logging.info(' ==> START ... ')
    logging.info(' ==> ALGORITHM SETTINGS <== ')
    logging.info(' --> Domain: ' + domain)

    logging.info(' ')

    param_limits = pd.DataFrame(index=['min', 'max', 'sigma', 'best'], columns=calibrated_params)
    section_data = None
    custom_date_parser = lambda x: dt.datetime.strptime(str(x), data_settings["data"]["hydro"]["date_fmt"])

    for par in calibrated_params:
        for lim in ['min', 'max']:
            param_limits.loc[lim][par] = data_settings['calibration']['parameters'][par][lim]
        param_limits.loc['sigma'][par] = np.nan
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Prepare input land data

    logging.info(' --> Import domain land data')

    logging.info(' ---> Load dem...')
    maps_in = {}
    dem = rio.open(os.path.join(data_settings["calibration"]["input_gridded_data_folder"], domain + ".dem.txt"))
    header = dem.profile
    header["driver"] = 'GTiff'

    maps_in['mask'] = dem.read_masks(1) / 255
    maps_in['mask'][maps_in['mask'] == 0] = np.nan
    maps_in['DEM'] = np.where(maps_in['mask'] == 1, dem.read(1), np.nan)

    logging.info(' ---> Load base maps for calibration...')
    available_approaches = set([data_settings['calibration']['parameters'][var]["approach"] for var in calibrated_params])

    calibrated_params_approach = {}

    for approach in available_approaches:
        if approach not in implemented_approaches.keys():
            logging.warning(" ---> WARNING! " + approach + " is not available!")
        calibrated_params_approach[approach] = [var for var in calibrated_params if data_settings['calibration']['parameters'][var]["approach"] == approach]
        if not approach == "uniform":
            for par in calibrated_params_approach[approach]:
                maps_in[par] = np.where(maps_in['mask'] ==1, rio.open(os.path.join(data_settings["calibration"]["input_base_maps"], domain + "." + par + ".txt")).read(1), np.nan)
        logging.info(' ----> Param(s) calibrated with approach ' + approach + ': ' + ' ,'.join(calibrated_params_approach[approach]))

    # Sigma has a different meaning accordign to the approaches: for the rescale it is the amplitude of the arctan rescale interval
    # For the others it is the range of the parameter space
    for par in calibrated_params:
        if par in calibrated_params_approach["rescale"]:
            param_limits.loc['sigma'][par] = 0.5
        else:
            param_limits.loc['sigma'][par] = param_limits.loc['max'][par] - param_limits.loc['min'][par]

    logging.info(' ---> Preparing domain land data.. OK!')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    logging.info(' ---> Read section data...')
    calibration_period = pd.date_range(calib_hydro_start, run_hydro_end,
                                       freq=data_settings["data"]["hydro"]["calib_hydro_resolution"])
    section_data = {}
    section_file = os.path.join(data_settings["calibration"]["input_point_data_folder"], domain + ".info_section.txt")
    area_file = os.path.join(data_settings["calibration"]["input_gridded_data_folder"], domain + ".area.txt")

    sections = pd.read_csv(section_file, sep="\s", header=None, names=["row_HMC", "col_HMC", "basin", "name"],
                           usecols=[0, 1, 2, 3])
    area = rio.open(area_file).read(1)
    logging.info('---> Search observed series')

    for section, basin in zip(sections["name"], sections["basin"]):
        logging.info('---> Section: ' + section)
        sections.loc[sections["name"] == section, "area_ncell"] = area[
            sections.loc[sections["name"] == section, "row_HMC"] - 1, sections.loc[
                sections["name"] == section, "col_HMC"] - 1]
        file_name_sec = os.path.join(data_settings["data"]["hydro"]["folder"],
                                     data_settings["data"]["hydro"]["filename"]).format(domain=domain,
                                                                                        section_name=section,
                                                                                        section_basin=basin)
        if os.path.isfile(file_name_sec):
            section_data[section] = pd.read_csv(file_name_sec,
                                                sep=data_settings["data"]["hydro"]["sep"],
                                                usecols=[data_settings["data"]["hydro"]["date_col"],
                                                         data_settings["data"]["hydro"]["value_col"]],
                                                names=["date", "value"],
                                                index_col=["date"],
                                                parse_dates=True,
                                                date_parser=custom_date_parser,
                                                na_values=data_settings["data"]["hydro"]["null_values"]
                                                )[calib_hydro_start:run_hydro_end].reindex(calibration_period,
                                                                                           method="nearest",
                                                                                           tolerance="1" +
                                                                                                     data_settings[
                                                                                                         "data"][
                                                                                                         "hydro"][
                                                                                                         "calib_hydro_resolution"])

            logging.info('---> ' + str(len(sections)) + ' sections found in the section file')
            logging.info('---> Section: ' + section + "... IMPORTED!")
        else:
            logging.warning('---> WARNING! Section: ' + section + "... NOT FOUND!")
    logging.info(' ---> Read section data...DONE!')
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Start the multi-iterative calibration procedure
    converges = False
    best_score_iter = {}
    for iIter in np.arange(iMin, iMax + 1):

        maps_iter = {}

        if iIter > 1:
            logging.info(' ---> Loading results of iteration ' + str(iIter - 1).zfill(2))
            with open(os.path.join(data_settings["algorithm"]["path"]["out_path"],
                                   'ITER' + str(iIter-1).zfill(2) + '_results.pickle'), "rb") as handle:
                previous_iter = pickle.load(handle)
            logging.info(' ---> Loading results of iteration ' + str(iIter - 1).zfill(2) + "...DONE")

            best_score_iter = previous_iter["best_score_iter"]

            if data_settings["algorithm"]["general"]["error_metrics"]["best_value"] == "max":
                idx_best = np.nanargmax(previous_iter["scores_iter"]["tot"].values) + 1
                best_score_iter[iIter - 1] = np.nanmax(previous_iter["scores_iter"]["tot"].values)
            elif data_settings["algorithm"]["general"]["error_metrics"]["best_value"] == "min":
                idx_best = np.nanargmin(previous_iter["scores_iter"]["tot"].values) + 1
                best_score_iter[iIter - 1] = np.nanmin(previous_iter["scores_iter"]["tot"].values)
            else:
                logging.error(" ---> ERROR! Choose if maximise (max) or minimise (min) the error_metrics")
                raise NotImplementedError(data_settings["algorithm"]["general"]["error_metrics"]["best_value"] + " is not a valid choice for error_metrics")

            logging.info(" ---> Best combination for iteration " + str(iIter - 1).zfill(2) + " is combination: " + str(idx_best).zfill(3))

            if iIter > 2:
                improvement = np.abs((best_score_iter[iIter - 1] - best_score_iter[iIter - 2])/best_score_iter[iIter - 1])
                if improvement < data_settings["algorithm"]["general"]["percentage_min_improvement_quit_optimization"]/100:
                    logging.info(" --> Optimization system converges, difference with previous iter < than " + str(data_settings["algorithm"]["general"]["percentage_min_improvement_quit_optimization"]) + "%")
                    maps_out = previous_iter["maps_iter"][idx_best]
                    converges = True
                    break
                else:
                    logging.info(" --> Improvement compared to previous iteration: " + str(improvement))

            logging.info(" --> Update parameters limits for a new iteration...")

            nExplor = int(nExplor * data_settings["algorithm"]["general"]["percentage_samples_successive_iterations"]/100)
            param_limits = previous_iter["param_limits"]
            param_bests = previous_iter["param"].loc[idx_best]

            for par in calibrated_params:
                maps_in[par] = previous_iter["maps_iter"][idx_best][par]
                param_limits.loc['sigma'][par] = param_limits.loc['sigma'][par]*(data_settings["algorithm"]["general"]["percentage_param_range_reduction"]/100)
                if not par in calibrated_params_approach["rescale"]:
                    param_limits[par]['min'] = np.max((param_limits[par]['min'], param_bests[par] - param_limits.loc['sigma'][par]/2))
                    param_limits[par]['max'] = np.min((param_limits[par]['max'], param_bests[par] + param_limits.loc['sigma'][par]/2))

        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Identify the random parameter seeds for the iteration
        logging.info(' --> Initialize iteration ITER' + str(iIter).zfill(2))
        seedIter = pd.DataFrame(np.array(lhs(len(calibrated_params), nExplor)), index=np.arange(1, nExplor + 1), columns=calibrated_params)

        param = pd.DataFrame(index=np.arange(1, nExplor + 1), columns=calibrated_params)
        param = param.fillna(0)

        for par in calibrated_params:
            if par in calibrated_params_approach["rescale"]:
                # With the rescale method maps are rescaled with arctan rescaling limited to a sigma-wide neighbourhood
                # The neighbourhood is asymmetrical keeping into account the ditance of the average from the limits
                diff_inf = np.abs(np.nanmean(maps_in[par]) - param_limits[par]['min'])
                diff_sup = np.abs(np.nanmean(maps_in[par]) - param_limits[par]['max'])
                min_scale = - (param_limits.loc['sigma'][par]*diff_inf/(diff_sup+diff_inf))
                max_scale = (param_limits.loc['sigma'][par]*diff_sup/(diff_sup+diff_inf))
            else:
                min_scale = param_limits[par]['min']
                max_scale = param_limits[par]['max']
            param[par] = ((seedIter[par] - np.min(seedIter[par])) / (np.max(seedIter[par]) - np.min(seedIter[par]))) * (
                                   max_scale - min_scale) + min_scale

        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Setup the explorative runs
        logging.info(' ---> Setup explorative runs...')
        translate_options = gdal.TranslateOptions(format="AAIGrid", outputType=gdal.GDT_Float32, noData=-9999, creationOptions=['FORCE_CELLSIZE=YES'])
        with open(data_settings["data"]["hmc"]["model_settings"]) as f:
            config_hmc_in = f.read()

        for iExplor in np.arange(1, nExplor + 1):
            logging.info(' --->  ITER' + str(iIter).zfill(2) + '-' + str(iExplor).zfill(3))
            iterPath = os.path.join(path_settings["work_path"], "simulations", 'ITER' + str(iIter).zfill(2) + '-' + str(iExplor).zfill(3))
            os.makedirs(iterPath, exist_ok=True)

            # Generation of iteration static maps
            logging.info(' ---> Generate exploration static maps...')

            logging.info(' ----> Copy static maps...')
            iter_gridded_path = os.path.join(iterPath,'gridded',)
            os.makedirs(iter_gridded_path, exist_ok=True)
            copy_all_files(data_settings["calibration"]["input_gridded_data_folder"], iter_gridded_path)

            logging.info(' ---> Generate exploration static maps...')

            logging.info(' ----> Copy all static maps...')
            iter_gridded_path = os.path.join(iterPath, 'gridded', )
            os.makedirs(iter_gridded_path, exist_ok=True)
            copy_all_files(data_settings["calibration"]["input_gridded_data_folder"], iter_gridded_path)

            logging.info(' ----> Generate parameters maps...')
            maps_out = {}
            for par in calibrated_params:
                maps_out[par] = implemented_approaches[data_settings['calibration']['parameters'][par]["approach"]](par, param[par][iExplor], data_settings['calibration']["parameters"][par], maps_in)
                with rio.open(os.path.join(iter_gridded_path,'temp.tif'), 'w', **header) as dst:
                    dst.write(np.nan_to_num(maps_out[par].astype(rio.float32), nan=-9999), 1)
                gdal.Translate(os.path.join(iter_gridded_path,domain + '.' + par + '.txt'), os.path.join(iter_gridded_path,'temp.tif'), options=translate_options)
                os.remove(os.path.join(iter_gridded_path,'temp.tif'))
            logging.info(' ---> Generate exploration static maps...DONE')

            logging.info(' ----> Copy point data...')
            iter_point_path = os.path.join(iterPath, 'point', )
            os.makedirs(iter_point_path, exist_ok=True)
            copy_all_files(data_settings["calibration"]["input_point_data_folder"], iter_point_path)
            logging.info(' ----> Copy point data...DONE')

            logging.info(' ----> Copy and setup model executable...')
            iter_exe_path = os.path.join(iterPath, 'exe', )
            iter_out_path = os.path.join(iterPath, 'outcome', )
            os.makedirs(iter_exe_path, exist_ok=True)

            shutil.copy(data_settings["data"]["hmc"]["model_exe"], os.path.join(iter_exe_path, "HMC3_calib.x"))
            config_hmc_out = config_hmc_in.format(domain='"' + domain + '"',
                                                  sim_length=str(int((run_hydro_end - run_hydro_start).total_seconds() / 3600)),
                                                  run_hydro_start = run_hydro_start.strftime("%Y%m%d%H%M"),
                                                  path_gridded = iter_gridded_path,
                                                  path_point = iter_point_path,
                                                  path_output = iter_out_path)

            with open(os.path.join(iter_exe_path, domain + ".info.txt"), "w") as f:
                f.write(config_hmc_out)
            make_launcher(iter_exe_path, domain, data_settings["data"]["hmc"]["system_env_libraries"])
            logging.info(' ----> Copy and setup model executable...DONE')

            maps_iter[iExplor] = maps_out

        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Launch the explorative runs
        bashCommand = "for iIt in $(seq -f \"%03g\" 1 " + str(nExplor) + "); do cd " \
                      + os.path.join(path_settings["work_path"], "simulations", 'ITER' + str(iIter).zfill(2)) \
                      + "-$iIt/exe/; chmod +x launcher.sh; ./launcher.sh & done\n wait"
        process = subprocess.run(bashCommand, shell=True, executable="/bin/bash")

        logging.info(' --> Simulation runs... OK!')

        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Read observed and modelled series
        logging.info(' --> Read model output...')

        # Import model output data
        logging.info(' ---> Read model output data...')
        hmc_results = {}
        for iExplor in np.arange(1, nExplor + 1):
            logging.info(' --->  Read results of ITER' + str(iIter).zfill(2) + '-' + str(iExplor).zfill(3))
            iterPath = os.path.join(path_settings["work_path"], "simulations",
                                    'ITER' + str(iIter).zfill(2) + '-' + str(iExplor).zfill(3))

            iter_settings_file = os.path.join(iterPath,'exe',domain + ".info.txt")
            with open(iter_settings_file, 'r') as input:
                for line in input:
                    if 'sPathData_Output_TimeSeries' in line:
                        iter_out_path = line.split("=")[1].replace("\n","").replace('"','').replace("'","")
                        break
            try:
                hmc_results[iExplor] = read_discharge_hmc(output_path=iter_out_path, col_names=sections["name"].values,
                                                      start_time=calib_hydro_start).reindex(calibration_period, method="nearest", tolerance="1" + data_settings["data"]["hydro"]["calib_hydro_resolution"])
            except FileNotFoundError:
                logging.error(" ---> WARNING! HMC output time-series file not found at path " + iter_out_path)
                hmc_results[iExplor] = None
                # raise FileNotFoundError
        logging.info(' ---> Read model output data...DONE')

        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Calculate scores
        logging.info (" --> Calculate iteration scores..")
        scores = pd.DataFrame(index=np.arange(1, nExplor + 1), columns=section_data.keys())
        scores_iter = pd.DataFrame(index=np.arange(1, nExplor + 1), columns=["tot"])
        eval_score = getattr(hs, data_settings["algorithm"]["general"]["error_metrics"]["function"])
        for iExplor in np.arange(1, nExplor + 1):
            logging.info(" ---> Parameters set " + str(iExplor) + "...")
            if hmc_results[iExplor] is None:
                for section in section_data.keys():
                    scores.loc[iExplor, section] = np.nan
                scores_iter.loc[iExplor, "tot"] = np.nan
            else:
                for section in section_data.keys():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scores.loc[iExplor,section] = eval_score(hmc_results[iExplor][section].values, section_data[section].values.squeeze())
                # Calculate overall scores for the iteration weighting the section score with its upstream contributing area
                if data_settings["algorithm"]["general"]["error_metrics"]["minimum_ins_inf"]:
                    scores_tmp = scores.loc[iExplor]
                    scores_tmp[scores_tmp<-1] = -1
                    scores.loc[iExplor] = scores_tmp
                    data_settings["algorithm"]["general"]["error_metrics"]["shift_for_positive"] = 1
                scores_iter.loc[iExplor,"tot"]= (np.nansum(np.log(sections.set_index("name")["area_ncell"]) * \
                           (data_settings["algorithm"]["general"]["error_metrics"]["shift_for_positive"] + scores.loc[iExplor]))) / \
                            np.nansum(np.log(sections.set_index("name")["area_ncell"]))
        logging.info(" --> Calculate iteration scores..DONE")

        # -------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Save iter outputs
        logging.info(" ---> Save outputs...")
        scores.to_csv(os.path.join(data_settings["algorithm"]["path"]["out_path"], 'ITER' + str(iIter).zfill(2) + '_sections_scores.csv'))
        with open(os.path.join(data_settings["algorithm"]["path"]["out_path"], 'ITER' + str(iIter).zfill(2) + '_results.pickle'),"wb") as handle:
            pickle.dump({"param_limits": param_limits, "param": param, "scores_iter":scores_iter, "maps_iter" : maps_iter, "best_score_iter": best_score_iter}, handle)
        logging.info(" ---> Save outputs...DONE")

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Max number of iterations reached without converging
    if iIter == iMax and converges is False:
        logging.warning(' ---> Max number of iterations has been reached without converging!')

        logging.info(' ---> Loading results of iteration ' + str(iIter).zfill(2))
        with open(os.path.join(data_settings["algorithm"]["path"]["out_path"],
                           'ITER' + str(iIter).zfill(2) + '_results.pickle'), "rb") as handle:
            current_iter = pickle.load(handle)
        logging.info(' ---> Loading results of iteration ' + str(iIter).zfill(2) + "...DONE")

        best_score_iter = current_iter["best_score_iter"]

        if data_settings["algorithm"]["general"]["error_metrics"]["best_value"] == "max":
            idx_best = np.nanargmax(current_iter["scores_iter"]["tot"].values) + 1
            best_score_iter[iIter] = np.nanmax(current_iter["scores_iter"]["tot"].values)
        elif data_settings["algorithm"]["general"]["error_metrics"]["best_value"] == "min":
            idx_best = np.nanargmin(current_iter["scores_iter"]["tot"].values) + 1
            best_score_iter[iIter] = np.nanmin(current_iter["scores_iter"]["tot"].values)
        else:
            logging.error(" ---> ERROR! Choose if maximise (max) or minimise (min) the error_metrics")
            raise NotImplementedError(data_settings["algorithm"]["general"]["error_metrics"][
                                      "best_value"] + " is not a valid choice for error_metrics")


        logging.info(" ---> Best combination for iteration " + str(iIter).zfill(2) + " is combination: " + str(idx_best).zfill(3))
        improvement = np.abs((best_score_iter[iIter] - best_score_iter[iIter - 1]) / best_score_iter[iIter])
        logging.info(" --> Improvement compared to previous iteration: " + str(improvement))
        maps_out = current_iter["maps_iter"][idx_best]


    # -------------------------------------------------------------------------------------
    logging.info(" --> Write resulting best maps...")
    os.makedirs(os.path.join(data_settings["algorithm"]["path"]["out_path"], "gridded",""), exist_ok=True)
    for par in calibrated_params:
        with rio.open(os.path.join(data_settings["algorithm"]["path"]["out_path"], "gridded", 'temp.tif'), 'w', **header) as dst:
            dst.write(np.nan_to_num(maps_out[par].astype(rio.float32), nan=-9999), 1)
        gdal.Translate(os.path.join(data_settings["algorithm"]["path"]["out_path"], "gridded", domain + '.' + par + '.txt'),
                       os.path.join(data_settings["algorithm"]["path"]["out_path"], "gridded", 'temp.tif'), options=translate_options)
        os.remove(os.path.join(data_settings["algorithm"]["path"]["out_path"], "gridded", 'temp.tif'))
    logging.info(" --> Write resulting best maps...DONE")

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
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    return alg_settings

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

# -------------------------------------------------------------------------------------
# Method to fill path names
def fillScriptSettings(data_settings, domain):
    path_settings = {}

    for k, d in data_settings["algorithm"]['path'].items():
        for k1, strValue in d.items():
            if isinstance(strValue, str):
                if '{' in strValue:
                    strValue = strValue.replace('{domain}', domain)
            path_settings[k1] = strValue

    return path_settings

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Makes a Latin Hyper Cube sample returns a matrix X of size n by p of a LHS of n values on each of p variables
# for each column of X, the n values are randomly distributed with one from each interval #(0,1/n), (1/n,2/n), ..., (1-1/n,1)
# and they are randomly permuted
def lhssample(n, p):
    x = np.random.uniform(size=[n, p])
    for i in range(0, p):
        x[:, i] = (np.argsort(x[:, i]) + 0.5) / n
    return x

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def rescale_map(map_name, par, par_settings, maps_in):
    map_max = par_settings["max"] * maps_in["mask"]
    map_min = par_settings["min"] * maps_in["mask"]

    scalaATan = (1 - np.double((2 - (1 - np.sign(par))) > 0)) * (maps_in[map_name] - map_min) + np.double((2 - (1 - np.sign(par))) > 0) * (map_max - maps_in[map_name])
    map = maps_in[map_name] + (scalaATan / (math.pi / 2)) * np.arctan(2 * (map_max - map_min) * ((math.pi / 2) / scalaATan) * par)

    if "lakes_mask" in par_settings.keys():
        logging.info(" ---> Lakes in " + map_name + " map are masked!")
        lakes_mask = rio.open(par_settings["lakes_mask"]).read(1)
        map = np.where(lakes_mask==1, maps_in[map_name], map)

    return map
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def rescale_mask(map_name, par, par_settings, maps_in):
    map = maps_in[map_name] * par
    map[map<0] = -9999
    return map
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def rescale_value(map_name, par, par_settings, maps_in):
    map = maps_in['mask'] * par
    return map
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def copy_all_files(source_folder, destination_folder):
    for file_name in os.listdir(source_folder):
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(destination_folder, file_name)
        if os.path.isfile(source):
            shutil.copy(source, destination)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method create HMC launcher
def make_launcher(iter_exe_path, domain_name, env_path):
    with open(os.path.join(iter_exe_path,"launcher.sh"), "w") as launcher:
        launcher.write("#!/bin/bash\n")
        launcher.write("source " + env_path + "\n")
        launcher.write("cd " + iter_exe_path+ "\n")
        launcher.write("chmod 777 HMC3_calib.x\n")
        launcher.write("ulimit -s unlimited\n")
        launcher.write("./HMC3_calib.x " + domain_name + ".info.txt")

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def read_discharge_hmc(output_path='', col_names=None, output_name="hmc.hydrograph.txt", format='txt', start_time=None, end_time=None):
    if format=='txt':
        custom_date_parser = lambda x: dt.datetime.strptime(x, "%Y%m%d%H%M")
        if col_names is None:
            print(' ---> ERROR! Columns names parameter not provided!')
            raise IOError("Section list should be provided as col_names parameter!")
        hmc_discharge_df = pd.read_csv(os.path.join(output_path,output_name), header=None, delimiter=r"\s+", parse_dates=[0], index_col=[0], date_parser=custom_date_parser)
        if len(col_names)==len(hmc_discharge_df.columns):
            hmc_discharge_df.columns=col_names
        else:
            print(' ---> ERROR! Number of hmc output columns is not consistent with the number of stations!')
            raise IOError("Verify your section file, your run setup or provide a personal column setup!")
        if start_time is None:
            start_time = min(hmc_discharge_df.index)
        if end_time is None:
            end_time = max(hmc_discharge_df.index)

    return hmc_discharge_df[start_time:end_time]
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to calculate hydro costs
def costiIdro(matSim, matObs, stations):
    J = np.empty((1, len(matSim.columns.values) + 1))
    ind = np.empty((1, len(matSim.columns.values) + 1))
    ii = 0

    for staz in matSim.columns.values:
        xSim = matSim[staz]
        xObs = matObs[staz]
        KGE = 1 - np.sqrt(np.power((np.corrcoef(xSim, xObs)[0, 1] - 1), 2) + np.power(
            (((np.std(xSim) / np.mean(xSim)) / (np.std(xObs) / np.mean(xObs))) - 1), 2) + np.power(
            ((np.mean(xSim) / np.mean(xObs)) - 1), 2))
        J[(0, ii)] = (2 / np.pi) * np.arctan(1 - KGE)
        ind[(0, ii)] = np.log(stations.area[staz])
        ii = ii + 1

    J[(0, -1)] = np.nansum(J[0, 0:-1] * ind[0, 0:-1]) / np.nansum(ind[0, 0:-1] * (J[0, 0:-1] / J[0, 0:-1]))
    return J

# ----------------------------------------------------------------------------
# Call script from external library

if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------