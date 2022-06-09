import pywanda
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelWrapperGA:
    def __init__(self, model_file, model_bin, meas_data_list, para_list):
        self.model_file = model_file
        self.bin = model_bin
        self.sensors_data = []
        self.meas_data_list = meas_data_list
        self.para_list = para_list
        self.list_of_values = []
        self.simulated_series = []
        self.time_axis = []
        self.RMSE_value = None

    def give_values(self, values):
        self.list_of_values = values

    def plot_result(self, values, res_folder=''):
        self.give_values(values)
        self.calc_fitness()
        fig = make_subplots(rows=len(self.meas_data_list), cols=1)
        i = 1
        for meas_series, sim_series in zip(self.meas_data_list, self.simulated_series):
            fig.add_trace(go.Scatter(x=self.time_axis, y=meas_series.values, name=meas_series.location + " Meas"),
                          row=i, col=1)
            fig.add_trace(go.Scatter(x=self.time_axis, y=sim_series, name=meas_series.location + " Sim"), row=i, col=1)

            i += 1
        fig['layout']['xaxis']['title'] = 'Time (s)'
        fig['layout']['xaxis2']['title'] = 'Time (s)'
        fig['layout']['yaxis']['title'] = 'Head (m)'
        fig['layout']['yaxis2']['title'] = 'Head (m)'
        fig.show()
        fig.write_image(res_folder + 'time_series.png')

    def calc_residual(self):
        self.calc_fitness()
        residual = []
        for meas_series, sim_series in zip(self.meas_data_list, self.simulated_series):
            for (sim_val, meas_val) in zip(sim_series, meas_series.values):
                residual.append((meas_val - sim_val) / (max(meas_series.values)))
        return residual

    def calc_fitness(self):
        raise NotImplemented


class WandaWrapperGA(ModelWrapperGA):
    def set_values(self, model, values):
        # sets the given sets of values in the model
        for (parameter, value) in zip(self.para_list, values):
            comps = []
            if parameter.iskeyword:
                comps = model.get_components_with_keyword(parameter.location)
            else:
                comps.append(model.get_component(parameter.location))
            for comp in comps:
                if comp.contains_property(parameter.property):
                    prop = comp.get_property(parameter.property)
                    prop.set_scalar(value / prop.get_unit_factor())
        model.save_model_input()

    def RMSE(self, model, factor=2):
        # Calculates the RMS of the given series with the measurement data, which is the fitness
        RMSE_total = 0.0
        self.time_axis = model.get_time_steps()
        self.simulated_series = []
        # is this working correctly and using the correct series f
        for meas_series in self.meas_data_list:
            # todo change that you also can get components
            if model.component_exists(meas_series.location):
                prop = model.get_component(meas_series.location).get_property(meas_series.property)
            else:
                prop = model.get_node(meas_series.location).get_property(meas_series.property)
            #self.simulated_series.append([x * prop.get_unit_factor() for x in prop.get_series()])
            # unti factor seems not te be workign therefore disabled for now.
            self.simulated_series.append([x for x in prop.get_series()])
            RMSE = 0.0
            for (sim_val, meas_val) in zip(self.simulated_series[-1], meas_series.values):
                RMSE += pow(sim_val - meas_val, factor)
            RMSE_total += math.pow(RMSE / len(meas_series.values) / max(meas_series.values), 1.0 / factor)
        return 1.0 / (RMSE_total / len(self.meas_data_list))

    def calc_fitness(self):
        model = pywanda.WandaModel(self.model_file, self.bin)
        self.set_values(model, self.list_of_values)
        model.run_steady()
        model.run_unsteady()
        self.RMSE_value = self.RMSE(model)
        model.close()
        return {str(self.list_of_values): self.RMSE_value}


