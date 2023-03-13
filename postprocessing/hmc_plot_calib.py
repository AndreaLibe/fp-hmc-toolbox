# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HMC tools - Calibration Plot best

__date__ = '20220822'
__version__ = '1.0.0'
__author__ = 'Andrea Libertino (andrea.libertino@cimafoundation.org')
__library__ = 'HMC_calibration_tool'

General command line:
python3 HMC_calibration -settings_file "FILE.json"
20220822 (0.0.1) -->    Beta release single domain
"""
# -------------------------------------------------------------------------------------
import pandas as pd
import datetime as dt
import os
from fpLibs.hmc.lib_io_generic import read_discharge_hmc
import matplotlib.pyplot as plt
import logging, json, time
from argparse import ArgumentParser
import hydrostats as hs
import hydrostats.metrics as hm
import plotly.tools as tls
import plotly.io as pio
import numpy as np
from matplotlib.colors import hsv_to_rgb
from cycler import cycler

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Script main

def main():
    start_time = time.time()
    cm = 1 / 2.54
    colors = [hsv_to_rgb([(i * 0.318033988749895) % 1.0, 1, 1]) for i in range(100)]
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)
    domain = data_settings["algorithm"]["general"]["domain_name"]

    # Set algorithm logging
    os.makedirs(data_settings["algorithm"]["path"]["log"], exist_ok=True)
    set_logging(logger_file=os.path.join(data_settings["algorithm"]["path"]["log"], domain + "_results_analysis.log"))

    # Set timing
    calib_hydro_start = dt.datetime.strptime(data_settings["algorithm"]["time"]["calib_hydro_start"], "%Y-%m-%d %H:%M")
    calib_hydro_end = dt.datetime.strptime(data_settings["algorithm"]["time"]["calib_hydro_end"], "%Y-%m-%d %H:%M")
    calibration_period = pd.date_range(start=calib_hydro_start, end=calib_hydro_end, freq=data_settings["algorithm"]["time"]["frequency"])

    custom_date_parser = lambda x: dt.datetime.strptime(x, data_settings["data"]["station"]["date_fmt"])

    # Set output path
    output_path = data_settings["algorithm"]["path"]["output"].format(domain=domain)
    os.makedirs(output_path, exist_ok=True)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    logging.info(" --> Read hmc static files")
    logging.info(" ---> Section file")
    section_data = {}
    section_file = data_settings["data"]["hmc"]["section_file"].replace("{domain}", domain)
    if data_settings["algorithm"]["general"]["mode"] == "validation":
        sections = pd.read_csv(section_file, sep="\s", header=None, names=["row_HMC", "col_HMC", "basin", "name", "file_name"], usecols=[0, 1, 2, 3, 4])
    elif data_settings["algorithm"]["general"]["mode"] == "calibration":
        sections = pd.read_csv(section_file, sep="\s", header=None, names=["row_HMC", "col_HMC", "basin", "name"], usecols=[0, 1, 2, 3])
    else:
        raise NotImplementedError
    logging.info(" --> Read hmc static files... DONE!")
    logging.info(" --> " + str(len(sections)) + " sections found in the file!")
    logging.info(" --> Read hmc output file")

    hmc_results_dict = {}
    for run_name in data_settings["data"]["hmc"]["model_run"].keys():
        logging.info(" ---> Read output for " + run_name + " model run")
        hmc_out_path = data_settings["data"]["hmc"]["model_run"][run_name]["output_folder"].replace("{domain}", domain)
        hmc_results_dict[run_name] = read_discharge_hmc(output_path=hmc_out_path, output_name= data_settings["data"]["hmc"]["model_run"][run_name]["output_filename"], col_names=sections["name"].values,
                                 start_time=calib_hydro_start, end_time=calib_hydro_end).reindex(calibration_period,
                                                                       method="nearest",
                                                                       tolerance="1" + data_settings["data"]["station"]["calib_hydro_resolution"])
    logging.info(" --> Read hmc output file... DONE!")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    logging.info(" --> Read section data")
    missing = []
    dic_name = {}
    for section, basin in zip(sections["name"], sections["basin"]):
        print('---> Section: ' + section)
        if data_settings["algorithm"]["general"]["mode"] == "validation":
            section_name = sections.loc[sections["name"]==section, "file_name"].values[0]
        elif data_settings["algorithm"]["general"]["mode"] == "calibration":
            section_name = section
        file_name_sec = os.path.join(data_settings["data"]["station"]["folder"], data_settings["data"]["station"]["filename"]).format(domain=domain, section_name=section_name, section_basin=basin)
        if os.path.isfile(file_name_sec):
            section_data[section] = pd.read_csv(file_name_sec,
                                                sep=data_settings["data"]["station"]["sep"],
                                                usecols=[data_settings["data"]["station"]["date_col"], data_settings["data"]["station"]["value_col"]],
                                                names=["date", "value"],
                                                index_col=["date"],
                                                parse_dates=True,
                                                header=0,
                                                date_parser=custom_date_parser,
                                                na_values=data_settings["data"]["station"]["null_values"]
                                                )
            section_data[section] = section_data[section][np.max((calib_hydro_start,min(section_data[section].index))):np.min((calib_hydro_end,max(section_data[section].index)))]

            if len(section_data[section]) == 0:
                print('---> Section: ' + section + "... SKIPPED! No data in selected time slice!")
                continue
            else:
                pass
            dic_name[section] = section_name
            print('---> Section: ' + section + "... IMPORTED!")
        else:
            missing = missing + [section]
            print('---> WARNING! Section: ' + section + "... NOT FOUND!")
            if data_settings["algorithm"]["general"]["plot_ungauged"]:
                section_data[section] = pd.DataFrame(index=hmc_results_dict[run_name].index, columns=["value"])

    section_data_avail = section_data.keys() - missing

    resume_table = {}
    for run_name in data_settings["data"]["hmc"]["model_run"].keys():
        resume_table[run_name] = pd.DataFrame(index=section_data_avail, columns=["ADHI_ID","nRMSE", "MAPE", "R2", "CORR", "NS", "r_KGE","a_KGE","b_KGE","KGE"])

    for section in section_data.keys():
        fig_object = plt.figure(figsize=(18 * cm, 8 * cm))
        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        #hmc_results[section].values[hmc_results[section].values>2000] = np.nan

        mod = {}

        for run_name in data_settings["data"]["hmc"]["model_run"].keys():
            mod[run_name], = plt.plot(hmc_results_dict[run_name][section].index, hmc_results_dict[run_name][section].values, '-')

            if section not in missing:
                try:
                    resume_table[run_name].loc[section,["ADHI_ID","nRMSE","MAPE","R2","CORR","NS","r_KGE","a_KGE","b_KGE","KGE"]] = [dic_name[section],
                                                                            hm.nrmse_mean(hmc_results_dict[run_name][section].values.astype('float32'), section_data[section].values.squeeze()),
                                                                            hm.mape(hmc_results_dict[run_name][section].values.astype('float32'), section_data[section].values.squeeze()),
                                                                            hm.r_squared(hmc_results_dict[run_name][section].values.astype('float32'), section_data[section].values.squeeze()),
                                                                            hm.pearson_r(hmc_results_dict[run_name][section].values.astype('float32'), section_data[section].values.squeeze()),
                                                                            hm.nse(hmc_results_dict[run_name][section].values.astype('float32'), section_data[section].values.squeeze())] + [i for i in hm.kge_2012(hmc_results_dict[run_name][section].values.astype('float32'), section_data[section].values.squeeze(), return_all=True)]
                except:
                    resume_table[run_name].loc[section] = -9999

        obs, = plt.plot(section_data[section].index, pd.to_numeric(section_data[section].values.squeeze(), errors='coerce'), '.', markersize=2)
        plt.title(section)
        hmc_names = [i for i in data_settings["data"]["hmc"]["model_run"].keys()]
        plt.legend([obs] + [mod[i] for i in hmc_names],["obs"] + hmc_names)
        plt.savefig(os.path.join(output_path, section + ".png"))
        plotly_fig = tls.mpl_to_plotly(fig_object)
        pio.write_html(plotly_fig, os.path.join(output_path, section + '.html'))

    for run_name in data_settings["data"]["hmc"]["model_run"].keys():
        resume_table[run_name].to_csv(os.path.join(output_path, run_name + "_scores.csv"))


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
if __name__ == "__main__":
    main()
# -----------