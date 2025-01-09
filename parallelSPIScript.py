import argparse
import os.path

from pyspi.calculator import Calculator
import time
import dill
import pandas as pd


def main(parquet_path: str, save_path: str, subset: str):

    # get the dataset
    if os.path.isfile(parquet_path):
        dataset = pd.read_parquet(parquet_path)
    else:
        raise ValueError(f'[{parquet_path}] is not a valid file.')

    # get rid of rooms that have constant signals
    std = dataset.std()
    rooms = {key.split('_')[0] for key in std[std <= 1e-10].index.tolist()}
    print(f'Delete {len(rooms)}/{len(set(col.split("_")[0] for col in dataset.columns))} rooms due to zero std.')
    dataset = dataset.loc[:, [column for column in dataset.columns if column.split('_')[0] not in rooms]]
    print(f'We have {dataset.shape[1]} signals after deletion.')

    # set up the calculation of dependency measures
    tt = time.perf_counter()
    calc = Calculator(dataset=dataset.to_numpy().T, configfile=subset)

    # get the results
    calc.compute()
    print('Computation took', time.perf_counter()-tt, 'Seconds')

    # drop the dataset so the saved results get smaller
    calc._dataset = None

    # save the calculator object as a .pkl
    with open(os.path.join(save_path, 'saved_calculator.pkl'), 'wb') as f:
        dill.dump(calc, f)
    return calc


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-pp', '--parquet_path', type=str, default=r"./")
    _parser.add_argument('-sp', '--save_path', type=str, default='soda')
    _parser.add_argument('-s', '--subset', type=str, default='fast')
    _args = _parser.parse_args()
    main(_args.parquet_path, _args.save_path, _args.subset)
