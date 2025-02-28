import os
from glob import glob

import pandas as pd
import pyspi

import evaluateSPI as evaspi


def evaluate_spi(result_path: str, spi_result_path: str = None, target_folder: str = "results"):

    if spi_result_path is not None:
        results = pd.read_parquet(spi_result_path)
        evaspi.print_results(results)
        return results

    # get the dataset from the result folder and make sure there is only one
    print(os.path.join(result_path, '*.parquet'))
    dataset_path = glob(os.path.join(result_path, '*.parquet'))
    print(dataset_path)
    assert len(dataset_path) == 1, f'There is more than one dataset in the result path: {dataset_path}.'
    dataset_path = dataset_path.pop()
    dataset = pd.read_parquet(dataset_path)
    print(f'We have {dataset.shape[1]} signals with {dataset.shape[0]} samples per signal.')

    # start the PySPI package once (so all JVM and octave are active)
    with evaspi.HiddenPrints():
        calc = pyspi.calculator.Calculator(subset='fabfour')

    # load the results
    evaspi.find_and_load_results(result_path, dataset, False, True, reorder_name='similarity')
    return None

if __name__ == '__main__':
    __root_path = os.path.join("measurements", "all_spis")
    paths = ['spi_keti', 'spi_plant1', 'spi_plant2', 'spi_plant3', 'spi_plant4', 'spi_rotary', 'spi_soda']
    __path_gen = (os.path.join(__root_path, __p) for __p in paths)
    for __path in __path_gen:
        evaluate_spi(__path)