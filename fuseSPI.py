import numpy as np
import evaluateSPI as evspi
import snf
import pyspi.calculator
from glob import glob
import os
import pandas as pd
import collections
import sklearn
import multiprocessing as mp


def fuse(affinity_list: list[np.ndarray], method: str = 'snf'):

    # now fuse the similarities
    if method == 'snf':
        for ele in affinity_list:
            ele[np.diag_indices_from(ele)] = np.nanmean(ele, axis=0)
        affinity_list = z_score_affinity_list(affinity_list)
        fused_similarities = snf.snf(*affinity_list)
    elif method == 'add':
        affinity_list = z_score_affinity_list(affinity_list)
        fused_similarities = sum(affinity_list)
    elif method == 'pca':

        # replace the middle elements
        for ele in affinity_list:
            ele[np.diag_indices_from(ele)] = np.nanmean(ele, axis=0)

        # flatten and concatenate
        shaped = affinity_list[0].shape
        overall_array = np.concatenate(list(ele.reshape(-1, 1) for ele in affinity_list), axis=1)

        # make the z-score normalization on the complete columns
        overall_mean = overall_array.mean(axis=0)
        overall_std = overall_array.std(axis=0)
        overall_std[overall_std == 0] = 1
        overall_array = (overall_array-overall_mean)/overall_std

        # compute the correlation matrix
        pca_transf = sklearn.decomposition.PCA(n_components=1)
        overall_array = pca_transf.fit_transform(overall_array)
        overall_array = overall_array.reshape(shaped)
        overall_array[np.diag_indices_from(overall_array)] = np.nan
        fused_similarities = overall_array
    elif method == 'median':
        affinity_list = z_score_affinity_list(affinity_list)
        overall_array = np.stack(affinity_list, axis=2)
        fused_similarities = np.mean(overall_array, axis=2)
    else:
        raise ValueError(f'Method {method} not defined.')
    return fused_similarities


def z_score_affinity_list(affinity_list: list[np.ndarray]):
    for idx, ele in enumerate(affinity_list):
        mean = np.nanmean(ele, axis=0)
        std = np.nanstd(ele, axis=0)
        std[std == 0] = 1
        affinity_list[idx] = (ele - mean) / std
    return affinity_list


def fuse_multiple(fuse_methods: list[str], similarity_collection: list[np.ndarray], result_df: pd.DataFrame,
                  subset: str):

    measures = []
    for method in fuse_methods:

        # now get the fusion
        fused_similarities = fuse(similarity_collection, method)

        # get a name of and existing spi, so we can copy the format
        sup_name = set(result_df.columns.get_level_values(0)).pop()
        new_name = f'fused_{subset}_{method}'

        # set the fusion into the results
        copied = result_df[[sup_name]].copy()
        copied = copied.rename(columns={sup_name: new_name})
        copied.iloc[:, :] = fused_similarities
        result_df = pd.concat([result_df, copied], axis=1)
        measures.append(new_name)
    return result_df, measures


def get_measure_subset(subset_name: str):
    # decide for the feature set
    if subset_name == 'performance':
        measures_selection = ['ce_gaussian', 'pec', 'prec-sq_ShrunkCovariance', 'gc_gaussian_k-1_kt-1_l-1_lt-1',
                              'si_gaussian_k-1', 'tlmi_gaussian']
    elif subset_name == 'speed':
        measures_selection = ['prec-sq_ShrunkCovariance', 'cov-sq_ShrunkCovariance', 'pec']
    else:
        raise NotImplementedError
    return measures_selection


def main(result_path: str):

    # get the dataset from the result folder and make sure there is only one
    dataset_path = glob(os.path.join(result_path, '*.parquet'))
    assert len(dataset_path) == 1, f'There is more than one dataset in the result path: {dataset_path}.'
    dataset_path = dataset_path.pop()
    dataset = pd.read_parquet(dataset_path)
    print(f'We have {dataset.shape[1]} signals with {dataset.shape[0]} samples per signal.')

    # start the PySPI package once (so all JVM and octave are active)
    with evspi.HiddenPrints():
        calc = pyspi.calculator.Calculator(subset='fast')

    # load the results
    result_df, measures, timing_dict, defined, terminated, undefined = evspi.find_and_load_results(result_path, dataset)

    # find the rooms
    rooms = {col.split('_')[0] for col in dataset.columns}

    # find sensors that have no neighbors by counting
    cn = collections.Counter(col.split('_', 1)[0] for col in dataset.columns)
    cn = {name for name, amount in cn.items() if amount <= 1}
    dropped_sensors = [ele for ele in dataset.columns if any(ele.startswith(name) for name in cn)]

    # drop the sensors that have no neighbors
    print(f'Dropping {len(dropped_sensors)} sensors as they have no partner.')
    result_df = result_df.drop(index=dropped_sensors)
    result_df = result_df.drop(columns=dropped_sensors, level=1)

    # create dataframe to save results
    results = pd.DataFrame(index=measures,
                           columns=["gross accuracy", "Mean Reciprocal Rank", "pairwise auroc", "Adjusted Rand Index",
                                    "Normalized Mutual Information", "Adjusted Mutual Information", "Homogeneity",
                                    "Completeness", "V-Measure", 'Triplet Accuracy', "Mean Average Precision",
                                    "Normalized Discount Cumulative Gain"],
                           dtype=float)

    # now get the things we would like to select
    for measure_subset in ['speed', 'performance']:
        similarity_collection = [result_df[selector].copy().to_numpy() for selector in get_measure_subset(measure_subset)]
        # similarity_collection = [result_df[selector].copy().to_numpy() for selector in measures]

        # make the fusion using multiple measures
        result_df, new_fused = fuse_multiple(['add', 'pca', 'median', 'snf'], similarity_collection,
                                             result_df, measure_subset)
        measures.extend(new_fused)

    # compute the results
    evspi.compute_triplet_accuracy(result_df, measures, results)
    evspi.compute_gross_accuracy(result_df, measures, results)
    evspi.compute_reciprocal_rank(result_df, measures, results)
    evspi.compute_pairwise_auroc(result_df, measures, results)
    evspi.compute_clustering(result_df, measures, results, len(rooms))
    evspi.compute_map(result_df, measures, results)
    evspi.compute_normalized_discounted_gain(result_df, measures, results)

    # save the results for quick loading
    results.to_parquet(os.path.join("results" ,f'result_fused_{os.path.split(result_path)[-1]}.parquet'))
    return results


if __name__ == '__main__':
    __root_path = r"measurements\all_spis"
    paths = ['spi_keti', 'spi_plant1', 'spi_plant2', 'spi_plant3', 'spi_plant4', 'spi_rotary', 'spi_soda']
    __path_gen = (os.path.join(__root_path, __p) for __p in paths)
    with mp.Pool(6) as pool:
        for _ in pool.imap_unordered(main, __path_gen):
            pass
