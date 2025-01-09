import collections
import os
import sys
from glob import glob
import typing
import multiprocessing as mp

from sklearn.metrics import roc_auc_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

import pyspi
from pyspi.calculator import Calculator

import dill
from tqdm import tqdm
import numpy as np
import pandas as pd


# https://stackoverflow.com/a/45669280
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def parse_spi_information(path: str = 'distance_or_similarity.txt'):
    spi_information = dict()
    categorized_spis = collections.defaultdict(list)
    with open(path, 'r') as filet:
        prev_line = ''
        curr_category = ''
        for line in filet.readlines():

            # check whether we changed category
            if line.startswith('#') and (prev_line == '\n' or prev_line == ''):
                curr_category = line[1:].strip()
            prev_line = line

            if line.startswith('#') or not line or line == '\n':
                continue
            if ' # ' in line:
                line = line.split(' # ', 1)[0]
            name, typed = line.split(': ')
            typed = typed.strip()
            name = name.strip()
            spi_information[name] = typed
            categorized_spis[curr_category].append(name)
            assert typed == 'distance' or typed == 'similarity', f'Something is off {line}.'
    return spi_information, categorized_spis


def find_and_load_results(result_path: str, original_dataset: pd.DataFrame):

    # find all the folders in the directory
    folders = glob(os.path.join(result_path, f'*{os.path.sep}'))

    # load the spi information
    spi_information, _ = parse_spi_information()

    # Go through all the folders and check the output file if the process was terminated
    # or successful. Additionally, load the pkl calculator if the process was not terminated
    terminated = []
    timing_dict = dict()
    undefined = []
    defined = []
    spi_groups = collections.Counter()
    for folder in folders:

        # load the output file
        with open(os.path.join(folder, 'output_file')) as filet:
            output = filet.readlines()

        # try to get the SPI name
        spi_name_estimate = [line for line in output if line.startswith('Processing [')]
        assert len(spi_name_estimate) >= 1, f'There is something off with the output of in: {folder}'
        spi_name_estimate = spi_name_estimate[0].split(']')[0].split(' ')[-1]
        spi_groups[spi_name_estimate.replace('-', '_').split('_', 1)[0]] += 1

        # check the lines of the process was terminated
        if any(line.startswith('Terminating the whole process group...') for line in output):
            terminated.append((spi_name_estimate, folder))
            continue

        # check the output whether it terminated
        timing_lines = [line for line in output if line.startswith("Calculation complete. Time taken: ")]
        assert len(timing_lines) <= 1, f'There is something off with the output of in: {folder}'

        if len(timing_lines) == 0:
            undefined.append((spi_name_estimate, folder))
            continue

        # extract the timing
        timing_lines = timing_lines.pop()
        timing_lines = timing_lines.split(': ')[-1].replace("s", " ")
        timing_lines = float(timing_lines)

        # load the results into memory
        with open(os.path.join(folder, 'saved_calculator.pkl'), 'rb') as f:
            intermediate = dill.load(f)
        defined.append(intermediate.table)

        # find the name of the SPI
        spi_name = set(intermediate.table.columns.get_level_values(0))
        assert len(spi_name) == 1, f'In folder {folder} are multiple SPIs: {spi_name}.'
        spi_name = spi_name.pop()
        assert spi_name == spi_name_estimate, f'There is something off with the output of in: {folder}'
        timing_dict[spi_name] = timing_lines

        # check if the spi is a distance or similarity and invert the values if it is a distance
        spi_group = spi_name.split('_', 1)[0]
        spi_group = spi_group.split('-', 1)[0]
        if spi_information[spi_group] == 'distance':
            mini = defined[-1].min().min()
            maxi = defined[-1].max().max()
            defined[-1] = np.exp(-(defined[-1]-mini)/(maxi-mini))

    # make a debug print
    print(f'From originally {len(folders)} found SPIs. {len(defined)} are defined {len(terminated)} were terminated '
          f'and {len(undefined)} were undefined.')

    # create one big table from all the results
    result_df = pd.concat(defined, axis=1)

    # rename the processes for every spi
    result_df.index = original_dataset.columns
    first_table = list(result_df.columns.get_level_values(0))[0]
    result_df.rename(columns={level_two: col for (level_two, col) in zip(result_df[first_table].columns, original_dataset.columns)}, inplace=True)

    # get all the measures we want and that have only main diagonal NaN
    measures = set(result_df.columns.get_level_values(0))
    original_amount = len(measures)
    measures = [table for table in measures
                if result_df[table].shape[1] == result_df[table].isna().to_numpy().sum()
                and not np.isinf(result_df[table].to_numpy()).any()]
    result_df = result_df[measures]
    print(f'We retained {len(measures)}/{original_amount} similarity measures (others were NaN or Inf).')
    return result_df, measures, timing_dict, defined, terminated, undefined


def metric_progress_wrapper(func_name: str, iterable: typing.Iterable, total: int = None) -> tqdm:
    description_text = f'{func_name} - Going through measures'
    if total is not None:
        return tqdm(iterable, desc=description_text, total=total)
    else:
        return tqdm(iterable, desc=description_text)


def compute_gross_accuracy(spi_df: pd.DataFrame, measures: list[str], results: pd.DataFrame):

    # produce the results
    for table in metric_progress_wrapper('Gross Accuracy', measures):
        gross_accuracy = 0

        # drop from the index, so we can't use it to compare
        currtab = spi_df[table].copy()
        results.loc[table, "gross accuracy"] = 0

        # check whether we have too many nan values
        if currtab.isna().to_numpy().sum() > currtab.shape[1]:
            print(table, 'has NaN.')
            continue

        # go through the sensor and check the k closest
        for element in currtab.columns:

            # drop the own element
            series = currtab[element].drop(index=element)

            # go to the corresponding column and check the closest elements
            closest = series.nlargest(1)

            # check majority class and also the closest element
            rooms = dict()
            for index, content in closest.items():
                room = index.split('_')[0]
                if room not in rooms:
                    rooms[room] = [1, content]
                else:
                    rooms[room][0] += 1
                    rooms[room][1] = max(content, rooms[room][1])

            # get the maximum element
            max_ele = max(rooms.items(), key=lambda x: x[1])[0]
            gross_accuracy += max_ele == element.split('_')[0]
        results.loc[table, "gross accuracy"] = gross_accuracy / currtab.shape[1]


def compute_reciprocal_rank(spi_df: pd.DataFrame, measures: list[str], results: pd.DataFrame):

    # produce the results
    for table in metric_progress_wrapper('Reciprocal Rank', measures):

        # get a copy of the results
        currtab = spi_df[table].copy()

        # go through all columns and check the rank of other sensors from the same room
        reciprocal_ranks = 0
        for col in currtab.columns:
            # Extract the room number and type from the original string
            room, type_to_exclude = col.split('_', 1)

            # get the median rank of the sensors
            max_rank = currtab[col].rank(ascending=False).filter(regex=rf'^{room}_(?!{type_to_exclude}$)').min()
            reciprocal_ranks += 1/max_rank
        results.loc[table, "Mean Reciprocal Rank"] = reciprocal_ranks / len(currtab.columns)


def compute_pairwise_auroc(spi_df: pd.DataFrame, measures: list[str], results: pd.DataFrame):

    # now we can also use auroc to for pairwise interactions
    for table in metric_progress_wrapper('Pairwise Auroc', measures):

        # get a copy of the results
        currtab = spi_df[table].copy()

        # make a copy to create the groundtruth
        gt = currtab.copy()
        gt.loc[:, :] = 0

        # make a numpy array that shows the rooms
        first_letters = [ele[0] for ele in gt.index.str.split('_')]

        # go through the columns of the ground truth and place the ground truth
        for col in gt.columns:
            room = col.split('_')[0]
            gt.loc[[ele == room for ele in first_letters], col] = 1

        # Create a mask for the main diagonal
        diagonal_mask = np.eye(currtab.shape[0], dtype=bool)

        # Invert the mask to get the non-diagonal elements
        non_diagonal_mask = ~diagonal_mask

        # compute the roc_auc
        roc_auc = roc_auc_score(gt.values[non_diagonal_mask], currtab.values[non_diagonal_mask])

        # put in the auc score
        results.loc[table, "pairwise auroc"] = roc_auc


def triplet_binary_search(positive_similarities: np.ndarray, negative_similarities: np.ndarray):

    # sort both arrays so we can binary search in them
    negative_similarities.sort(axis=0)
    positive_similarities.sort(axis=0)

    # we assume that positive examples are fewer, therefore, we iterate over the positive
    # and search in negative with binary search (our correct triplets are how many negatives are
    # smaller, which is exactly the index result of binary search)
    correct = np.sum(np.searchsorted(negative_similarities, positive_similarities))
    return correct, (positive_similarities.shape[0]*negative_similarities.shape[0])


def triplet_pandas_cross(positive_similarities: pd.DataFrame, negative_similarities: pd.DataFrame):

    # make the cross-product of negative and positive samples
    cross = positive_similarities.merge(negative_similarities, how='cross').to_numpy()

    # count how many positive similarities are higher
    count = np.sum(cross[:, 0] > cross[:, 1])
    return count, cross.shape[0]


def compute_triplet_accuracy(spi_df: pd.DataFrame, measures: list[str], results: pd.DataFrame):

    # This is the fastest (and somewhat memory efficient) triplet accuracy implementation I could come up with.
    # Remember: Never use loops in Python, especially when nested three times.
    #
    # WARNING: This counts triplets at least twice, but it is the same implementation as in other papers,
    # so we keep it like this to be comparable
    for table in metric_progress_wrapper('Triplet Accuracy', measures):
        # get a copy of the results
        currtab = spi_df[table].copy()

        # go through all the different triplets and make accuracy
        sensors = list(currtab.columns)

        # get the rooms from the sensors
        rooms = [ele.split('_', 1)[0] for ele in sensors]

        # for every room get the positive and negative examples
        positive_idces = dict()
        for room in set(rooms):
            positive_idces[room] = currtab.index.str.startswith(room)

        # Make all the triplets and whether they are successful
        summed = 0
        triplets = 0
        for adx, anchor in enumerate(sensors):

            # find the room of the anchor sensor
            anchor_system = rooms[adx]

            # get the positive samples values with the same room
            positive = currtab.loc[positive_idces[anchor_system], [anchor]]
            positive = positive.drop(anchor, axis=0, inplace=False)

            # get all the negative samples
            negative = currtab.loc[~positive_idces[anchor_system], [anchor]]

            # check how many positive samples are more similar than negative ones
            # correct1, count1 = triplet_pandas_cross(positive, negative)
            correct, count = triplet_binary_search(positive.to_numpy()[:, 0], negative.to_numpy()[:, 0])
            # assert correct1 == correct
            # assert count1 == count
            summed += correct
            triplets += count

        # compute the triplet accuracy
        results.loc[table, 'Triplet Accuracy'] = summed / triplets


def compute_clustering(spi_df: pd.DataFrame, measures: list[str], results: pd.DataFrame, num_clusters: int):

    # now we can also use auroc to for pairwise interactions
    for table in metric_progress_wrapper('Clustering', measures):
        # get a copy of the results
        currtab = spi_df[table].copy()
        currtab.loc[:, :] -= currtab.min().min()

        # Compute the symmetric DataFrame by averaging with its transpose
        currtab = (currtab + currtab.T) / 2

        # fill the main diagonal of the dataframe
        np.fill_diagonal(currtab.values, 1)

        # make some clustering
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
        predicted_labels = clustering.fit_predict(currtab)

        # evaluate the clustering
        gt = [col.split('_')[0] for col in currtab.columns]

        ari_score = adjusted_rand_score(gt, predicted_labels)
        results.loc[table, "Adjusted Rand Index"] = ari_score

        nmi_score = normalized_mutual_info_score(gt, predicted_labels)
        results.loc[table, "Normalized Mutual Information"] = nmi_score

        ami_score = adjusted_mutual_info_score(gt, predicted_labels)
        results.loc[table, "Adjusted Mutual Information"] = ami_score

        homogeneity = homogeneity_score(gt, predicted_labels)
        results.loc[table, "Homogeneity"] = homogeneity

        completeness = completeness_score(gt, predicted_labels)
        results.loc[table, "Completeness"] = completeness

        v_measure = v_measure_score(gt, predicted_labels)
        results.loc[table, "V-Measure"] = v_measure


def compute_map(spi_df: pd.DataFrame, measures: list[str], results: pd.DataFrame):
    # Precision@n: https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_484
    # AP: https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_482
    # MAP: https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_492

    # produce the results
    for table in metric_progress_wrapper('Mean Average Precision', measures):

        # drop from the index, so we can't use it to compare
        currtab = spi_df[table].copy()
        ranked_currtab = currtab.rank(method='first', ascending=False)

        # go through all the columns
        average_precision_sum = 0
        for col in ranked_currtab.columns:

            # find the room that belongs to the current column
            system = col.split('_', 1)[0]

            # locate the correct positions
            locator = [ele for ele in currtab.index if ele.startswith(system) and ele != col]

            # compute the average precision by sorting the ranks of the interesting elements
            # and then divide the number of retrieved elements by these ranks, build the sum and divide
            # by the number of interesting elements as defined in the formula of average precision
            sorted_ranks = np.sort((ranked_currtab.loc[locator, col].to_numpy()))
            average_precision_sum += np.sum(np.array(range(1, len(locator)+1))/sorted_ranks)/len(locator)

        results.loc[table, "Mean Average Precision"] = average_precision_sum/len(currtab.columns)


def compute_normalized_discounted_gain(spi_df: pd.DataFrame, measures: list[str], results: pd.DataFrame):

    # produce the results
    for table in metric_progress_wrapper('Normalized Discount Cumulative Gain', measures):

        # drop from the index, so we can't use it to compare
        currtab = spi_df[table].copy()
        ranked_currtab = currtab.rank(method='average', ascending=False)

        # make a copy to create the ground truth
        gt = ranked_currtab.copy()
        gt.loc[:, :] = 0

        # make a numpy array that shows the rooms
        first_letters = [ele[0] for ele in gt.index.str.split('_')]

        # go through the columns of the ground truth and place the ground truth
        for col in gt.columns:
            room = col.split('_')[0]
            gt.loc[[ele == room for ele in first_letters], col] = 1

        # get rid of the rooms themselves (as those are not part of the correct ground truth)
        diagonal_mask = np.eye(currtab.shape[0], dtype=bool)
        gt[diagonal_mask] = 0

        # compute the logarithm of all the ranks and weight with relevance
        ranked_currtab[diagonal_mask] = 1  # make sure there are no NaN in the array
        ranked_currtab.loc[:, :] += 1
        ranked_currtab = np.log2(ranked_currtab.to_numpy())

        # compute the dcg per sensor
        dcg = np.sum((1/ranked_currtab)*gt.to_numpy(), axis=0)

        # compute the idcg per sensor
        per_sensor_correct = np.sum(gt.to_numpy(), axis=0).astype('int')
        idcg = np.array([np.sum(1/np.log2(np.array(range(2, amount+2)))) for amount in per_sensor_correct])

        # fill in the result
        results.loc[table, "Normalized Discount Cumulative Gain"] = np.mean(dcg/idcg)


def print_results(results: pd.DataFrame):

    # print the results
    for col in results.columns:
        print(col)
        print(results[col].nlargest(5))
        print(results[col].nsmallest(5))
        print('\n\n')


def evaluate_spi(result_path: str, spi_result_path: str = None, target_folder: str = "results"):

    if spi_result_path is not None:
        results = pd.read_parquet(spi_result_path)
        print_results(results)
        return results

    # get the dataset from the result folder and make sure there is only one
    dataset_path = glob(os.path.join(result_path, '*.parquet'))
    assert len(dataset_path) == 1, f'There is more than one dataset in the result path: {dataset_path}.'
    dataset_path = dataset_path.pop()
    dataset = pd.read_parquet(dataset_path)
    print(f'We have {dataset.shape[1]} signals with {dataset.shape[0]} samples per signal.')

    # find the rooms
    rooms = {col.split('_')[0] for col in dataset.columns}

    # find sensors that have no neighbors by counting
    cn = collections.Counter(col.split('_', 1)[0] for col in dataset.columns)
    cn = {name for name, amount in cn.items() if amount <= 1}
    dropped_sensors = [ele for ele in dataset.columns if any(ele.startswith(name) for name in cn)]

    # start the PySPI package once (so all JVM and octave are active)
    with HiddenPrints():
        calc = pyspi.calculator.Calculator(subset='fast')

    # load the results
    result_df, measures, timing_dict, defined, terminated, undefined = find_and_load_results(result_path, dataset)
    # print('Specifiers for the terminated SPIs:')
    # print(collections.Counter(ele[0].split('_', 1)[0].split('-', 1)[0] for ele in terminated))
    # print('Specifiers for the undefined SPIs:')
    # print(collections.Counter(ele[0].split('_', 1)[0].split('-', 1)[0] for ele in undefined))

    # save the timing as a dataframe
    timings = pd.DataFrame.from_dict({key: [val] for key, val in timing_dict.items()})
    timings.to_parquet(os.path.join(target_folder, f'timings_{os.path.split(result_path)[-1]}.parquet'))

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

    # compute the results
    compute_triplet_accuracy(result_df, measures, results)
    compute_gross_accuracy(result_df, measures, results)
    compute_reciprocal_rank(result_df, measures, results)
    compute_pairwise_auroc(result_df, measures, results)
    compute_clustering(result_df, measures, results, len(rooms))
    compute_map(result_df, measures, results)
    compute_normalized_discounted_gain(result_df, measures, results)

    # save the results for quick loading
    # print_results(results)
    results.to_parquet(os.path.join(target_folder, f'result_{os.path.split(result_path)[-1]}.parquet'))
    print('---------------------------------------------------------------------------------------')
    return results


if __name__ == '__main__':
    __root_path = r"measurements\all_spis"
    paths = ['spi_keti', 'spi_plant1', 'spi_plant2', 'spi_plant3', 'spi_plant4', 'spi_rotary', 'spi_soda']
    __path_gen = (os.path.join(__root_path, __p) for __p in paths)
    with mp.Pool(6) as pool:
        for _ in pool.imap_unordered(evaluate_spi, __path_gen):
            pass

