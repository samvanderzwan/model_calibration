#Class for epanet files
import epynet
from src import wanda_wrapper
import math


class EpanetWrapperGA(wanda_wrapper.ModelWrapperGA):
    def __init__(self, model_file, model_bin, meas_data_list, para_list):
        super().__init__(model_file, model_bin, meas_data_list, para_list)

    def set_values(self, model, values):
        # sets the given sets of values in the model
        for pipe in model.pipes:
            if pipe.roughness >= 1.0:
                diameter = pipe.diameter
                new_diameter = values[0] * diameter
                pipe.diameter = new_diameter

    def RMSE(self, model, factor=2):
        # Calculates the RMS of the given series with the measurement data, which is the fitness
        RMSE_total = 0.0
        self.simulated_series = []
        for meas_series in self.meas_data_list:
            if meas_series.property == 'Head':
                self.simulated_series.append(model.nodes[meas_series.location].head)
            elif meas_series.property == 'Pressure':
                self.simulated_series.append(model.nodes[meas_series.location].pressure)
            self.time_axis = self.simulated_series[0].axes[0]
            RMSE = 0.0
            for (sim_val, meas_val) in zip(self.simulated_series[-1], meas_series.values):
                RMSE += pow(sim_val - meas_val, factor)
            RMSE_total += math.pow(RMSE / len(meas_series.values) / max(meas_series.values), 1.0 / factor)
        return 1.0 / (RMSE_total / len(self.meas_data_list))

    def calc_fitness(self):
        model = epynet.Network(self.model_file)
        for meas_point in self.meas_data_list:
            model.add_node_of_interest(meas_point.location)
        self.set_values(model, self.list_of_values)
        model.save_inputfile(self.model_file)
        model.run()
        self.RMSE_value = self.RMSE(model)
        return {str(self.list_of_values): self.RMSE_value}
