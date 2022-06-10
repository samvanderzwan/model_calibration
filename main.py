from src import parallel_runner


def main(config_file):
    ga_object = parallel_runner.run_GA(config_file)
    result = ga_object.run_GA(500.0)
    ga_object.post_processing()


if __name__ == '__main__':
    config_files = [
        r'.\config.yml',
        #  r'd:\projecten\calibration\config_taw_5_meas.yml',
        #  r'd:\projecten\calibration\config_taw_10_meas.yml',
        #  r'd:\projecten\calibration\config_taw_10_meas_start.yml',
        #  r'd:\projecten\calibration\config_taw_2_meas.yml',
        #  r'd:\projecten\calibration\config_taw_20_meas.yml'
    ]
    for config_file in config_files:
        print('running now for ' + config_file)
        main(config_file)
