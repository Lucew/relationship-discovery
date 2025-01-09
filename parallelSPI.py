# we use this nice article by
# https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
# WARNING: This only works on LINUX platforms.

import os
import subprocess
import multiprocessing as mp
import loadBuildingData as ldb
import yaml
from yaml.representer import Representer
from yaml.dumper import Dumper
import argparse
import time
import functools
from tqdm import tqdm
import psutil


def run_calculator(input_tuple: tuple[str, str], parquet_path: str, timeout_s: int):

    # unpack the tuple we received as input
    save_path, subset = input_tuple

    # check whether we are on a posix system
    if os.name != 'posix':
        raise EnvironmentError('This program can only be run on linux machines!')

    # create the command we want to run
    cmd = ['python', 'parallelSPIScript.py',
           '--parquet_path', parquet_path, '--save_path', save_path, '--subset', subset]

    # create the file we want to write the output in
    filet = open(os.path.join(save_path, 'output_file'), 'w')

    # start running the process (multiprocessing can't time out, therefore going this way)
    proc = subprocess.Popen(cmd, start_new_session=True, stderr=filet, stdout=filet)

    # put a timeout on the process
    try:
        proc.wait(timeout=timeout_s)
    # check whether we reached a timeout and kill the process with all its children
    except subprocess.TimeoutExpired:
        print(f'\n\n\nTimeout for {cmd} ({timeout_s}s) expired', file=filet)
        print('Terminating the whole process group...', file=filet)

        # https://gist.github.com/jizhilong/6687481?permalink_comment_id=3057122#gistcomment-3057122
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()
        proc.kill()

    finally:
        filet.close()


def iterate_config(config_path: str):
    # load the config we want to use for the pyspi run
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # go through the config and write single configs out of there
    for spi_type, spi_names in config.items():
        for spi_name, spi_configs in spi_names.items():

            # create the dict we want to keep
            keep_dict = {key: val for key, val in spi_configs.items() if key != 'config'}

            # check whether we have configs then iterate over those
            spi_configs = spi_configs.get('configs', None)
            if spi_configs is None:
                keep_dict['configs'] = None
                yield {spi_type: {spi_name: keep_dict}}
            else:
                for config in spi_configs:
                    # this needs to be a list otherwise it will fail!
                    keep_dict['configs'] = [config]
                    yield {spi_type: {spi_name: keep_dict}}


class BlankNone(Representer):
    """
    Print None as blank into yaml when used as context manager
    https://stackoverflow.com/a/67524482
    """
    def represent_none(self, *_):
        return self.represent_scalar(u'tag:yaml.org,2002:null', u'')

    def __enter__(self):
        self.prior = Dumper.yaml_representers[type(None)]
        yaml.add_representer(type(None), self.represent_none)

    def __exit__(self, exc_type, exc_val, exc_tb):
        Dumper.yaml_representers[type(None)] = self.prior


def seconds2str(n: int):

    # get the days first
    days, n = divmod(n, 24 * 3600)

    # then compute the hours
    hours, n = divmod(n, 3600)

    # get the minutes and seconds
    minutes, seconds = divmod(n, 60)

    return f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"


def main(path: str, dataset_name: str, sampling_rate: str, timeout_s: int, workers: int = None,
         deselect_rooms: bool = True):

    # get the dataset
    dataset_name = dataset_name.lower()
    if dataset_name == 'keti':
        dataset, _, _ = ldb.load_keti(os.path.join(path, 'KETI'), sample_rate=sampling_rate)
    elif dataset_name == 'soda':
        dataset, _, _ = ldb.load_soda(os.path.join(path, 'Soda'), sample_rate=sampling_rate, sensor_count=2)
    else:
        raise ValueError(f'Did not recognize the specified dataset: [{dataset_name}].')
    print(f'For data set {dataset_name}, we have {dataset.shape[1]} signals with {dataset.shape[0]} samples/signal.')

    # get rid of rooms that have constant signals
    std = dataset.std()
    if deselect_rooms:
        rooms = {key.split('_')[0] for key in std[std <= 1e-10].index.tolist()}
        print(f'Delete {len(rooms)}/{len(set(col.split("_")[0] for col in dataset.columns))} rooms due to zero std.')
        dataset = dataset.loc[:, [column for column in dataset.columns if column.split('_')[0] not in rooms]]
    else:
        columns = std[std >= 1e-10].index
        print(f'Delete {len(dataset.columns)-len(columns)}/{len(dataset.columns)} signals due to zero std.')
        dataset = dataset[columns.tolist()]
    print(f'We have {dataset.shape[1]} signals after deletion.')

    # make a folder for the current run
    curr_path = f'spi_{int(time.time())}'
    os.mkdir(curr_path)

    # save the dataset into the current working directory
    parquet_path = os.path.join(curr_path, f'{dataset_name}.parquet')
    dataset.to_parquet(parquet_path)

    # flatten the config into a list of things to do and create the corresponding folders
    config_paths = []
    for idx, config in enumerate(iterate_config('config.yaml')):

        # create the new folder
        result_path = os.path.join(curr_path, str(idx))
        os.mkdir(result_path)

        # save the new config
        config_path = os.path.join(result_path, 'config.yaml')
        with BlankNone(), open(config_path, 'w') as filet:
            yaml.dump(config, filet, default_flow_style=False)
        config_paths.append((result_path, config_path))

    # set the number of workers
    if workers is None:
        workers = mp.cpu_count()//2
    if workers > mp.cpu_count():
        raise ValueError(f'We can not use more workers than cores. You defined [{workers}], we have {mp.cpu_count()}.')

    # set the time for the dataset if not specified, here we allow 250ms per comparison
    if timeout_s is None:
        timeout_s = int(dataset.shape[1]*dataset.shape[1]*0.250)

    # save the configuration in there
    with open(os.path.join(curr_path, 'config.txt'), 'w') as filet:
        filet.write(f'path: {path}\n'
                    f'dataset_name: {dataset_name}\n'
                    f'sampling_rate: {sampling_rate}\n'
                    f'timeout_s: {timeout_s}\n'
                    f'workers: {workers}\n'
                    f'deselect rooms: {deselect_rooms}')

    # calculate the approximate duration of the computation with timeouts for perfect parallelization
    estimated_duration = seconds2str(int(timeout_s * len(config_paths) / workers))
    print(f'With perfect parallelization the estimated duration for is: {estimated_duration}')

    # now make the multiprocessing over all the metrics
    spi_computing = functools.partial(run_calculator, parquet_path=parquet_path, timeout_s=timeout_s)
    with mp.Pool(workers) as pool:

        # capture the keyboard interrupt to kill all child processes
        try:
            # do this to have a progress bar
            result_iterator = tqdm(pool.imap_unordered(spi_computing, config_paths),
                                   desc=f'Computing SPIs ({seconds2str(timeout_s)})', total=len(config_paths))
            for _ in result_iterator:
                pass
        except KeyboardInterrupt as er:

            # get the currently running process and kill all its children
            proc = psutil.Process(os.getpid())
            for child in proc.children(recursive=True):
                child.kill()

            # reraise the keyboard interrupt
            raise er


def check_bool(inputed: str):
    if inputed.lower() not in ['true', 'false']:
        raise ValueError(f'We only accept true/false you specified: [{inputed}].')
    return inputed.lower() == 'true'


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-p', '--path', type=str, default=r"C:\Users\Lucas\Data")
    _parser.add_argument('-d', '--dataset', type=str, default='keti')
    _parser.add_argument('-s', '--sampling_rate', type=str, default='1min')
    _parser.add_argument('-t', '--timeout_s', type=int, default=None)
    _parser.add_argument('-w', '--workers', type=int, default=None)
    _parser.add_argument('-dr', '--delete_rooms', type=check_bool, default=True)
    _args = _parser.parse_args()
    main(_args.path, _args.dataset, _args.sampling_rate, _args.timeout_s, _args.workers, _args.delete_rooms)
