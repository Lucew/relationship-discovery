import os.path
import functools

import numpy as np
import rankSPI as rspi
import pandas as pd


def insert_histograms_in_tikz(histogram_strings: list[str]):
    tikz_sourrounding = r"""
\begin{tikzpicture}[xscale=0.5, yscale=0.1]
%CONTENT%
\end{tikzpicture}
"""
    # replace the content with the histograms
    tikz_code = tikz_sourrounding.replace("%CONTENT%", "\n\n".join(histogram_strings))
    return tikz_code

def array2str(array: np.ndarray):
    return ", ".join(str(round(ele, 3)) for ele in array)

def create_tikz_src(counts: np.ndarray, bins: np.ndarray, min_val: int, max_val: int, rx: int, cx: int):
    # compute the start and end index of the histograms
    start = cx*2

    # define the default layout for the tikz histogram
    tikz_default = r"""

    % Define bin borders and counts for histogram %HIST_NUMBER%
    \def\binborders{{%BIN_ARRAY%}} % Bin borders
    \def\counts{{%COUNT_ARRAY%}}        % Counts

    % Draw histogram bars with black borders for histogram %HIST_NUMBER%
    \foreach \i in {1, ..., %VALUE_COUNT%} {
        \pgfmathsetmacro{\xstart}{\binborders[\i - 1]}
        \pgfmathsetmacro{\xend}{\binborders[\i]}
        \pgfmathsetmacro{\height}{\counts[\i - 1]}

        % Draw a rectangle with a black border for the current bin
        \fill[blue!70] (\xstart, %HISTOGRAM_ROW%) rectangle (\xend, \height);
        \draw[black] (\xstart, %HISTOGRAM_ROW%) rectangle (\xend, \height); % Add black border
    }

    % Add labels for the first and last bin borders for histogram %HIST_NUMBER%
    \node[below] at (%HISTOGRAM_START%, %HISTOGRAM_ROW%) {%MIN_VALUE%};
    \node[below] at (%BIN_COUNT%, %HISTOGRAM_ROW%) {%MAX_VALUE%};

    """

    # create the bin borders
    bins = np.linspace(0, num=bins.shape[0], stop=1)

    # replace the placeholders by corresponding values
    tikz_src = tikz_default.replace("%BIN_ARRAY%", array2str(bins+start))
    tikz_src = tikz_src.replace("%HISTOGRAM_START%", str(bins[0]+start))
    tikz_src = tikz_src.replace("%HISTOGRAM_ROW%", str(rx*4))
    tikz_src = tikz_src.replace("%COUNT_ARRAY%", array2str(counts))
    tikz_src = tikz_src.replace("%VALUE_COUNT%", str(counts.shape[0]))
    tikz_src = tikz_src.replace("%BIN_COUNT%", str(bins[-1]+start))
    tikz_src = tikz_src.replace("%MIN_VALUE%", str(min_val))
    tikz_src = tikz_src.replace("%MAX_VALUE%", str(max_val))
    tikz_src = tikz_src.replace("%HIST_NUMBER%", str(f"{rx}.{cx}"))
    return tikz_src


def main_old():
    # load the data into memory
    datasets = rspi.load_results(os.path.join("..", "results"))

    # go through all datasets and find the spis that completed for all datasets
    spis = list(functools.reduce(lambda x, y: x & y, (set(data.index) for data in datasets.values())))

    # define the metrics we want to look at (defined in the paper)
    metrics = ["Mean Reciprocal Rank", "Mean Average Precision", "Normalized Discount Cumulative Gain",
               "Triplet Accuracy", "Adjusted Rand Index", "Adjusted Mutual Information", "V-Measure"]

    # get rid of unwanted spis
    datasets = {key: df.loc[spis, metrics] for key, df in datasets.items()}

    # define minimum and maximum value for every measure
    min_max_vals = {"Mean Reciprocal Rank": [0, 1], "Mean Average Precision": [0, 1],
                    "Normalized Discount Cumulative Gain": [0, 1], "Triplet Accuracy": [0, 1],
                    "Adjusted Rand Index": [-0.5, 1], "Adjusted Mutual Information": [-1, 1],
                    "V-Measure": [0, 1]}
    assert set(metrics) == set(min_max_vals.keys()), 'You forgot to specify min max properties of some metrics.'

    # go through the datasets and compute the histograms
    tikz_src = []
    for dx, (dataset, data_df) in enumerate(datasets.items()):

        # in theory adjuste mutual information is only upper bounded
        # so check that no value is smaller than minimum

        # go through all columns and create a histogram
        for cdx, col in enumerate(data_df.columns):

            # check minimum and maximum
            assert data_df[col].min() >= min_max_vals[col][0], f'For {dataset} a {col} value is smaller than minimum value.'
            assert data_df[col].max() <= min_max_vals[col][1], f'For {dataset} a {col} value is larger than maximum value.'

            # get description
            data_description = data_df[col].describe()

            # make a histogram of the values using numpy
            hist = np.histogram(data_df[col], bins=np.linspace(*min_max_vals[col], 11))

            # create the tikz source
            tikz_src.append(create_tikz_src(*hist, *min_max_vals[col], rx=dx, cx=cdx))

    with open('absolute_results.tex', 'w') as f:
        f.write(insert_histograms_in_tikz(tikz_src))


def create_table(dataset_names: list[str]):

    # define the default table header
    table_header = r"""\begin{table}[bp]
    \caption{Absolute metric values for all datasets. Higher is better for all metrics. The statistics are calculated over all relationship measures per evaluation metric and dataset. Using $\wedge$: Minimum ,$\varnothing$: Mean, $\vee$: Maximum as symbols.}
	\label{tab:metrics}
	\centering
	{\fontsize{6.5}{7.8}\selectfont
	\begin{tabular}{cc%POSITION_PLACEHOLDER%}
		\toprule
		\textbf{\footnotesize Metric} & & %DATASET_NAMES% \\ \toprule
		%CONTENT%
	\end{tabular}
	}
\end{table}%"""

    # create the string for the columns
    header_string = ' & '.join(f"\\textbf{{\\footnotesize {ds.capitalize()}}}" for ds in dataset_names)
    table = table_header.replace("%DATASET_NAMES%", header_string)
    table = table.replace("%POSITION_PLACEHOLDER%", "c"*len(dataset_names))
    return table


def fill_table(datasets: dict[str: pd.DataFrame]):

    # create the table from using the datasets
    dataset_list = list(datasets.keys())
    dataset_list = dataset_list[1:5] + dataset_list[5:] + dataset_list[0:1]
    table_str = create_table(dataset_list)

    # get all the metrix
    metrics = list(next(iter(datasets.values())).describe().columns)

    # make a dict to save the data for each dataset
    dataset_description = {metric: {'min': [], 'max': [], 'mean': []} for metric in metrics}

    # go through the datasets
    for dataset_name in dataset_list:

        # get the corresponding dataframe
        data_df = datasets[dataset_name]

        # get the description of all columns
        described = data_df.describe()

        # go through the metrics and append the min, mean, and max value
        for metric in metrics:
            dataset_description[metric]['min'].append(described.loc['min', metric])
            dataset_description[metric]['max'].append(described.loc['max', metric])
            dataset_description[metric]['mean'].append(described.loc['mean', metric])

    # a string that represents a table row (so a metrix
    table_row = r"""
\multirow{3}{*}{\footnotesize %METRIC_NAME%} & $\wedge$  & %MINIMUM% \\ \cline{2-%DATASET_NUMBER%} 
        & $\varnothing$ & %MEAN% \\ \cline{2-%DATASET_NUMBER%}
        & $\vee$  & %MAXIMUM% \\ %RULE%
"""
    table_row = table_row.replace("%DATASET_NUMBER%", str(2+len(dataset_list)))

    # short descriptions
    short_names = {"Mean Reciprocal Rank": "MRR", "Mean Average Precision": "MAP",
                   "Normalized Discount Cumulative Gain": "NDCG", "Triplet Accuracy": "TA",
                   "Adjusted Rand Index": "ARI", "Adjusted Mutual Information": "AMI", "V-Measure": "VM"}

    # go through the metrics and create the rows
    table_rows = []
    for metric in metrics:

        # place the values into the table rows
        curr_row = table_row.replace("%METRIC_NAME%", short_names[metric])
        curr_row = curr_row.replace('%MINIMUM%', ' & '.join(list_to_str(dataset_description[metric]['min'])))
        curr_row = curr_row.replace('%MEAN%', ' & '.join(list_to_str(dataset_description[metric]['mean'])))
        curr_row = curr_row.replace('%MAXIMUM%', ' & '.join(list_to_str(dataset_description[metric]['max'])))

        # place the end rule
        curr_row = curr_row.replace("%RULE%", r"\specialrule{0.8pt}{0.5pt}{0.5pt}" if metric != metrics[-1] else r"\bottomrule[1.1pt]")

        # append to the list of rows
        table_rows.append(curr_row)

    # place the rows into the table and write to file
    table_str= table_str.replace('%CONTENT%', "\n".join(table_rows))
    return table_str

def list_to_str(numbers: list[float]):
    return [f"{ele:0.2f}" for ele in numbers]


def main():
    # load the data into memory
    datasets = rspi.load_results(os.path.join("..", "results"))

    # go through all datasets and find the spis that completed for all datasets
    spis = list(functools.reduce(lambda x, y: x & y, (set(data.index) for data in datasets.values())))

    # define the metrics we want to look at (defined in the paper)
    metrics = ["Mean Reciprocal Rank", "Mean Average Precision", "Normalized Discount Cumulative Gain",
               "Triplet Accuracy", "Adjusted Rand Index", "Adjusted Mutual Information", "V-Measure"]

    # get rid of unwanted spis
    datasets = {key: df.loc[spis, metrics] for key, df in datasets.items()}
    table_str = fill_table(datasets)

    with open('absolute_results.tex', 'w') as f:
        f.write(table_str)



if __name__ == '__main__':
    main()