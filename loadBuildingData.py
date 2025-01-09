import collections

import pandas as pd
import multiprocessing as mp
from glob import glob
from tqdm import tqdm
import os
import functools
import numpy as np


def aggregate_ts(df: pd.DataFrame, start_date: np.datetime64, end_date: np.datetime64, sample_rate: str = "1min",
                 method: str = "linear") -> pd.DataFrame:

    # check whether we have a named index, otherwise name it
    if df.index.name is None:
        index_name = "datetime"
        df.index.name = index_name
    else:
        index_name = df.index.name

    # get all the signals we have at the same time stamps
    signals = df.columns

    # create the new sampling frequency time stamps
    sampling_frequency_index = pd.date_range(
        start=start_date,
        end=end_date,
        freq=sample_rate,
        name=index_name,
        inclusive='both'
    )

    # insert the time stamps into the other dataframe and fill with NaN
    df = df.combine_first(pd.DataFrame(index=sampling_frequency_index)).sort_index()

    # fill the now missing time stamps using the chosen method of interpolation (this should also make sure that
    # there always is a value per time stamp)
    if method == "ffill":
        df = df.ffill()
    elif method == "linear":
        df = df.interpolate("time")
    else:
        raise ValueError(f"Method {method} not available for aggregation.")

    # compute the delta time stamps (can never be zero as the time is an index and therefore unique)
    df["dt"] = (pd.to_numeric(df.index.to_series()).diff()).shift(-1)

    # go over all the signals and make the aggregation
    for signal in signals:
        if method == "ffill":
            df[signal] = df[signal] * df["dt"]  # area of rectangle
        elif method == "linear":
            df[signal] = df["dt"] * (df[signal] + df[signal].shift(-1)) / 2  # area of trapezoid
        else:
            raise ValueError(f"Method {method} not available for aggregation.")

    # resample the signal to only keep the values that we need and sum up the areas
    df = df.resample(sample_rate, label="left").sum()

    # go over the signals and divide the area by the summed time so we have a good
    for signal in signals:
        df[signal] = df[signal] / df["dt"]

    # get rid of the delta time column
    df = df.drop(["dt"], axis=1)

    # get rid of the signal before and after
    df = df.loc[start_date: end_date, :]

    # find the indices of all NaN values and check they are only at the beginning and at the end so we do not end
    # up with non-regular time series if we drop NaN values
    nandx = np.where(df.isna())[0]
    assert len(nandx) < 3 or np.all(np.diff(nandx) < 2), f"There are some values missing in between: {nandx[1:]}."

    # get rid of the nan values
    df = df.ffill()
    df = df.bfill()
    return df


def aggregate_room(input_tuple: (str, dict[str: pd.DataFrame]),
                   start_date: np.datetime64, end_date: np.datetime64,
                   sample_rate: str) -> (str, dict[str: pd.DataFrame]):

    # unpack the input
    room_name, room_data = input_tuple

    # go through all signals in the room
    for room, signal in room_data.items():
        room_data[room] = aggregate_ts(signal, start_date, end_date, sample_rate)
    return room_name, room_data


def load_data_from_folder(path: str) -> dict[str: pd.DataFrame]:

    # find all the csv files in the folder
    files = glob(os.path.join(path, '*.csv'))

    # load them into a dictionary
    files = {os.path.split(os.path.splitext(file)[0])[-1]: pd.read_csv(file, index_col=0, header=None, names=["value"])
             for file in files}

    # convert the index into time stamps
    start_timestamp = 0
    end_timestamp = 0
    for file in files.values():
        start_timestamp = max(start_timestamp, file.index.min())
        end_timestamp = max(end_timestamp, file.index.max())
        file.index = pd.to_datetime(file.index, unit='s')
        file.index.name = 'datetime'

    # sanity check the folders
    assert len(files) == 5, f'Folder {path} has {len(files)}.'
    return path, files, start_timestamp, end_timestamp


def load_keti(path: str, sample_rate: str):

    # find all the folders in the directory
    folders = glob(os.path.join(path, f'*{os.path.sep}'))

    # make the wrapper for the tqdm progress bar
    wrapper = functools.partial(tqdm, desc=f'Loading Data', total=len(folders))

    # go through all the folders and load the files include a loading bar
    # TODO: Some timestamps start at the 25.08.2013 and some already end at 26.08.2013!
    start_timestamp = 0
    end_timestamp = 0
    data = dict()
    with mp.Pool(mp.cpu_count()//2) as pool:
        for path, files, start_date, end_date in wrapper(pool.imap_unordered(load_data_from_folder, folders)):
            data[os.path.split(os.path.split(path)[0])[-1]] = files
            start_timestamp = max(start_timestamp, start_date)
            end_timestamp = max(end_timestamp, end_date)

    # convert the maximum timestamps
    start_timestamp = pd.to_datetime(start_timestamp, unit='s').ceil(sample_rate)
    end_timestamp = pd.to_datetime(end_timestamp, unit='s').floor(sample_rate)

    # make the wrapper for the tqdm progress bar
    wrapper = functools.partial(tqdm, desc=f'Resampling Data', total=len(data))

    # make a function we can work with for resampling
    resampler = functools.partial(aggregate_room, start_date=start_timestamp, end_date=end_timestamp,
                                  sample_rate=sample_rate)
    with mp.Pool(mp.cpu_count()//2) as pool:
        data = {room: room_data for room, room_data in wrapper(pool.imap_unordered(resampler, data.items()))}

    # check that all sensors have the same length
    length = set(df.shape[0] for room in tqdm(data.values(), desc='Check length') for df in room.values())
    start = set((df.index.min(), df.index.max()) for room in tqdm(data.values(), desc='Check index')
                for df in room.values())
    assert len(length) == 1, 'Length is different for some values.'
    assert len(start) == 1, 'Index is different for some values.'
    length = length.pop()
    start = start.pop()

    # combine into one dataframe
    reformated = []
    for room, room_data in tqdm(data.items(), desc='Rename and Reformat'):
        for sensor, df in room_data.items():
            df = df.rename(columns={'value': f'{room}_{sensor}'})
            reformated.append(df)

    # put into one dataframe
    df = pd.concat(reformated, axis=1)

    # check whether there are any NaN
    assert not df.isna().any(axis=1).any(), 'There are NaN values.'

    # make information print
    print(f'Loaded and resampled all with sampling rate {sample_rate} signals from {start} with length {length}.')
    return df, length, start


def read_soda_ground_truth(path: str):
    """
    modified from
    https://github.com/MingzheWu418/Joint-Training/blob/79f112114d182738444ddebbf23c4a14250d0eb4/colocation/Data.py#L55

    :param path: the path where the ground truth is located
    :return: the sensor with corresponding room information
    """

    # get the lines of the ground truth file
    with open(path, 'r') as filet:
        file_lines = filet.readlines()

    # go through the lines and make the sensor associations
    sensors_information = dict()
    for sensor, information in zip(file_lines[0::2], file_lines[1::2]):

        # get the sensor name
        sensor = sensor.strip()

        # parse the sensor information
        information = information.strip().split(',')

        # check whether we find the room id in the information of the lines
        if len(information) > 4 and information[4].startswith('room-id'):

            # split the original room information
            name = information[3].split(":")
            identifier = information[4].split(":")

            # make some assertions that need to be true to create a fitting sensor name
            assert len(name) == 3 and name[2] == 'c', \
                f'For sensor {sensor} we have an unexpected room name: {information[3]}.'
            assert len(identifier) == 3 and identifier[2] == 'v', \
                f'For sensor {sensor} we have an unexpected room identifier: {information[4]}.'

            # create a unique room name from the information
            name = f'{name[1].replace("_", "-")}|{identifier[1].replace("_", "-")}'
            sensors_information[sensor] = name
    return sensors_information


def load_soda(path: str, sample_rate: str, sensor_count: int = 3):

    # read in the ground truth information
    ground_truth = read_soda_ground_truth(os.path.join(path, "SODA-GROUND-TRUTH"))

    # set not allowed sensors (that have std of zero in almost all cases)
    not_allowed = ["TMR", "AGN", "ARS", "ASO", "_S"]

    # go through all csv files
    room_sensors = collections.defaultdict(dict)
    for path in tqdm(glob(os.path.join(path, '*.csv')), desc='Load SODA'):

        # check whether the file is not from excluded sensors
        file_name = os.path.split(os.path.splitext(path)[0])[-1]
        if any(file_name.endswith(na) for na in not_allowed):
            continue

        # check whether we have room information for the sensor
        if file_name not in ground_truth:
            continue

        # replace the underscores with minus
        sensor_type = file_name.split('_')[-1]

        # read the data
        data = pd.read_csv(path, index_col=0, header=None, names=["value"])
        data.index = pd.to_datetime(data.index, unit='s')
        data.index.name = 'datetime'

        # get the room id
        room_id = ground_truth[file_name]

        # append the sensor to the dict
        assert sensor_type not in room_sensors[room_id], 'Something is off. We have multiple similar sensors per room.'
        room_sensors[room_id][sensor_type] = data

    # only keep rooms with corresponding sensor number
    room_sensors = {room: information for room, information in room_sensors.items()
                    if len(information) >= sensor_count}

    # check if all the rooms have the same sensors
    sensors = collections.Counter(tuple(sorted(sensors.keys())) for sensors in room_sensors.values())

    # convert the index into time stamps
    start_timestamp = pd.to_datetime(0, unit='s')
    end_timestamp = pd.to_datetime(0, unit='s')
    for sensor_dict in room_sensors.values():
        for df in sensor_dict.values():
            start_timestamp = max(start_timestamp, df.index.min())
            end_timestamp = max(end_timestamp, df.index.max())
    # convert the maximum timestamps
    start_timestamp = pd.to_datetime(start_timestamp, unit='s').ceil(sample_rate)
    end_timestamp = pd.to_datetime(end_timestamp, unit='s').floor(sample_rate)

    # make the wrapper for the tqdm progress bar
    wrapper = functools.partial(tqdm, desc=f'Resampling Data', total=len(room_sensors))

    # make a function we can work with for resampling
    resampler = functools.partial(aggregate_room, start_date=start_timestamp, end_date=end_timestamp,
                                  sample_rate=sample_rate)
    with mp.Pool(mp.cpu_count() // 2) as pool:
        data = {room: room_data for room, room_data in wrapper(pool.imap_unordered(resampler, room_sensors.items()))}

    # check that all sensors have the same length
    length = set(df.shape[0] for room in tqdm(data.values(), desc='Check length') for df in room.values())
    start = set((df.index.min(), df.index.max()) for room in tqdm(data.values(), desc='Check index')
                for df in room.values())
    assert len(length) == 1, 'Length is different for some values.'
    assert len(start) == 1, 'Index is different for some values.'
    length = length.pop()
    start = start.pop()

    # combine into one dataframe
    reformated = []
    for room, room_data in tqdm(data.items(), desc='Rename and Reformat'):
        for sensor, df in room_data.items():
            df = df.rename(columns={'value': f'{room}_{sensor}'})
            reformated.append(df)

    # put into one dataframe
    df = pd.concat(reformated, axis=1)

    # check whether there are any NaN
    assert not df.isna().any(axis=1).any(), 'There are NaN values.'

    # make information print
    print(f'Loaded and resampled all with sampling rate {sample_rate} signals from {start} with length {length}.')
    return df, length, start


if __name__ == '__main__':
    load_soda(r"C:\Users\Lucas\Data\Soda", '1min')
    load_keti(r"C:\Users\Lucas\Data\KETI", '1min')
