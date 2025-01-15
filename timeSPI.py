import functools
import pandas as pd
from glob import glob
import os
import rankSPI as rspi


def load_results(path: str):

    # get all the results into memory
    results = {os.path.split(cp)[-1].split('_')[-1].split('.')[0]: (pd.read_parquet(cp))
               for cp in glob(os.path.join(path, 'results', f'timings_spi*.parquet'))}

    return results


def make_timings(path: str = './', use_fused: bool = False):

    # load the files
    results = load_results(path)

    # load the spis that finished the computations
    _, _, overall_rank, _ = rspi.make_create_ranks(metric_subset="all", dataset_subset="all", use_fused=False)

    # go through the results and compute the means as well as a set of metrics that finished for all
    spis = list(overall_rank.index)
    timings = dict()
    for name, data in results.items():
        # get the average ranks over the different metrics for each dataset and do the rename
        timings[name] = data.loc[0, spis]

    # now create the set of metrics that is the same for all of them
    timings = pd.DataFrame.from_dict(timings, orient='columns')

    return timings


if __name__ == '__main__':
    make_timings(use_fused=False)