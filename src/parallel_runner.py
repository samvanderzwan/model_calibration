import shutil

import yaml
from src import meas_data
from src import wanda_wrapper
from src import epanet_wrapper
from src import GA
from src import PSO
import pandas as pd
import os
import queue
import time
import multiprocessing as mp
import pywanda
import numpy as np
import plotly.graph_objects as go
import math
import csv


class run_GA:
    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        self.model_bin = cfg['General']['bin-directory']
        self.base_model = cfg['model']['model-file']
        self.number_of_processes = cfg['General']['number_of_processes']
        self.pop_size = cfg['model']['population_size']
        self.results_folder = cfg['General']['results_folder']
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

        # load parameters
        self.parameters = []
        if self.model_bin:
            self.load_parameters_wanda(cfg)
        else:
            self.load_parameters_basic(cfg)

        # load measurement data
        data = pd.read_excel(cfg['measurements']['measurement-data'])
        time_axis = data['Time'].tolist()
        self.meas_data_list = []
        for para, prop in zip(cfg['measurements']['locations'], cfg['measurements']['quantity']):
            if para in data:
                time_series = data[para].to_list()
            else:
                raise RuntimeError(para + ' is not in measurement data')
            self.meas_data_list.append(meas_data.MeasData(para, prop, time_axis, time_series))

        # create  for each individual in the population
        self.cases = []
        case_directory, case_name = os.path.split(self.base_model)
        for i in range(0, self.pop_size):
            file_dir = os.path.join(case_directory, str(i))
            if os.path.isdir(file_dir):
                shutil.rmtree(file_dir)  # delete directory if it already exists
            os.mkdir(file_dir)
            dst = os.path.join(file_dir, case_name)
            shutil.copyfile(self.base_model, dst)
            # thise needs to be changed if it is an epanet model
            if self.model_bin:
                self.cases.append(
                    wanda_wrapper.WandaWrapperGA(dst, self.model_bin, self.meas_data_list, self.parameters))
            else:
                self.cases.append(
                    epanet_wrapper.EpanetWrapperGA(dst, self.model_bin, self.meas_data_list, self.parameters))
        self.result = []

    def load_parameters_wanda(self, cfg):
        base_model = pywanda.WandaModel(self.base_model, self.model_bin)
        for para, prop, minimum, maximum in zip(cfg['parameter']['components'], cfg['parameter']['property'],
                                                cfg['parameter']['minimum'], cfg['parameter']['maximum']):
            comps = base_model.get_components_with_keyword(para)
            if comps:
                self.parameters.append(meas_data.Parameter(para, prop, minimum, maximum, iskeyword=True))
                continue
            if base_model.component_exists(para):
                if base_model.get_component(para).contains_property(prop):
                    self.parameters.append(meas_data.Parameter(para, prop, minimum, maximum))
                else:
                    raise Exception(para + " does not have property " + prop)
            else:
                raise Exception(para + " does not exist in Wanda model")
        base_model.close()

    def load_parameters_basic(self, cfg):
        for para, prop, minimum, maximum in zip(cfg['parameter']['components'], cfg['parameter']['property'],
                                                cfg['parameter']['minimum'], cfg['parameter']['maximum']):
            self.parameters.append(meas_data.Parameter(para, prop, minimum, maximum))

    def calc_fitness(self, values):
        # recopying the case file to ensure it is correct.
        case_directory, case_name = os.path.split(self.base_model)
        for i in range(0, self.pop_size):
            file_dir = os.path.join(case_directory, str(i))
            dst = os.path.join(file_dir, case_name)
            shutil.copyfile(self.base_model, dst)
        # calculates in parallel the fitness
        tasks_to_accomplish = mp.Queue()
        tasks_that_are_done = mp.Queue()
        for case, value in zip(self.cases, values):
            case.give_values(value)
            tasks_to_accomplish.put(case.calc_fitness)
        # creating processes
        processes = []
        for w in range(self.number_of_processes + 1):
            p = mp.Process(target=do_work, args=(tasks_to_accomplish, tasks_that_are_done))
            processes.append(p)
            p.start()
        # completing process
        print("waiting for processes to finish")
        for p in processes:
            p.join()
        # print the output
        result = {}

        while not tasks_that_are_done.empty():
            res = tasks_that_are_done.get()
            result.update(res)
        # sorting the result
        sorted_result = []
        for value in values:
            sorted_result.append(result[str(value)])
        return sorted_result

    def run_GA(self, criteria):
        maximum = [x.maximum for x in self.parameters]
        minimum = [x.minimum for x in self.parameters]
        gen_algo = GA.GA(self.calc_fitness, self.pop_size, minimum, maximum)
        # gen_algo = PSO.PSO(self.calc_fitness, self.pop_size, minimum, maximum)
        while (gen_algo.generation < 100) & (gen_algo.best_objective < criteria):
            gen_algo.create_new_generation()
            print(gen_algo.times_not_changed)
            # gen_algo.plot_current_position()
            if gen_algo.times_not_changed > 5:
                break
        gen_algo.plot_current_position(self.results_folder)
        self.result = 10 ** gen_algo.best_individual
        self.cases[0].plot_result(10 ** gen_algo.best_individual, res_folder=self.results_folder)

    def post_processing(self):
        # recopying the case file to ensure it is correct.
        case_directory, case_name = os.path.split(self.base_model)
        for i in range(0, self.pop_size):
            file_dir = os.path.join(case_directory, str(i))
            dst = os.path.join(file_dir, case_name)
            shutil.copyfile(self.base_model, dst)
        # calc residual and plot it as cum distribution compare to normal distribution with same mu and sigma
        residuals = []
        self.cases[0].give_values(self.result)
        residuals.append(self.cases[0].calc_residual())
        change = 0.9
        for i in range(1, len(self.parameters) + 1):
            values = np.log10(self.result[:])
            values[i - 1] = change * values[i - 1]
            self.cases[i].give_values(10 ** values)
            residuals.append(self.cases[i].calc_residual())
        self.plot_residual_normal(residuals[0])
        # Calc jacobian
        jacobian = []
        for i in range(len(self.parameters)):
            step = change * np.log10(self.result[i]) - np.log10(self.result[i])
            jacobian.append([(x - y) / step for x, y in zip(residuals[i + 1], residuals[0])])
        jac_np = np.array(jacobian)
        hes = np.inner(jac_np, jac_np)
        [u, s, v] = np.linalg.svd(jac_np)
        error = 10 ** (1.0 / s)
        minval = [x * y ** -2 for x, y in zip(self.result, error)]
        maxval = [x * y ** 2 for x, y in zip(self.result, error)]
        print(minval)
        print(maxval)
        # save to file

        with open(self.results_folder + 'result.csv', mode='w') as result_file:
            csv_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(np.log10(self.result[:]))
            csv_writer.writerow(minval)
            csv_writer.writerow(maxval)

        # Hes = jac * Jac transposed

        # SVD of hesian to determine uncertintiy in result

    def plot_residual_normal(self, residual):
        mean = np.mean(residual)
        stdev = np.std(residual)
        xvals = []
        yvals = []
        for x in np.arange(mean - 4 * stdev, mean + 4 * stdev, stdev / 100):
            xvals.append(x)
            yvals.append(math.exp(-math.pow(x / stdev, 2) / (2.0 * stdev * math.sqrt(2.0 * math.pi))))
        hist, bin_edges = np.histogram(residual, bins=50)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xvals, y=np.cumsum([yval / sum(yvals) for yval in yvals])))
        fig.add_trace(go.Scatter(x=bin_edges[0:-2], y=np.cumsum([yval / sum(hist) for yval in hist])))
        fig.show()
        fig.write_image(self.results_folder + 'residual.png')


def do_work(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.                '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            print("process started" + str(task))
            result = task()
            tasks_that_are_done.put(result)
            time.sleep(0.5)
    return
