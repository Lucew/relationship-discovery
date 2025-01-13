![Concept](./plots/Concept.svg)
# Overview
This repository contains the code to reproduce the experiments from the paper
"Relationship Discovery for Heterogeneous Time Series Integration: A Comparative Analysis for Industrial and 
Building Data" published at the [BTW2025](https://btw2025.gi.de/).

Not all used datasets are publicly available, but the building datasets can be accessed:
- [KETI](https://www.kaggle.com/datasets/ranakrc/smart-building-system) or [here](https://github.com/MingzheWu418/Joint-Training)
- [SODA](https://github.com/MingzheWu418/Joint-Training/tree/main/colocation/rawdata/metadata/Soda)

Computing the relationship measures can be reproduced by installing python and the necessary dependencies
listed in the `requirements.txt` and then execute `parallelSPI.py` with the corresponding
parameters in a shell.

# Reproducibility

Computing over 200 relationship measures for every of the seven datasets is the main computational bottleneck. Even when
distributing the computations on a capable compute server, the results take days to complete.

Therefore, we provide the [precalculated similarity/distance matrices](./measurements) within this repository. Computing the relationship
measures uses [available](https://github.com/DynamicsAndNeuralSystems/pyspi) and
[published](https://arxiv.org/abs/2201.11941) Code, therefore, this step is not part of the reproducibility guide. For
the publicly available datasets, one can run `parallelSPI.py` as described above.

In order to reproduce the results of our paper starting from the precomputed similarity/distance matrices take the 
following steps:
1. Clone this repository.
2. Install the necessary requirements using the `requirements.txt`
3. Run `python evaluateSPI.py` and `python fuseSPI.py` to compute the evaluation metrics (paper: section 3.3) and produce the ranking of the relationship measures (paper: section 3.4)
4. Using the jupyther notebook [plots_results_paper.ipynb](./plots_results_paper.ipynb) you can then reproduce all plots of the paper
5. Run `python absoluteResultsTable.py` to create the .tex file for table 6 in the paper

We also provide a [dockerfile](./Dockerfile) an accompanying [docker compose file](./compose.yaml). You can start the
container by installing [docker](https://docs.docker.com/get-started/get-docker/) and then navigating to
[http://127.0.0.1:8081/notebooks/plots_results_paper.ipynb](http://127.0.0.1:8081/notebooks/plots_results_paper.ipynb)
after the build and start up is finished.

This repository also already contains the intermediate results from step 2. Therefore, you could start at step three right away after cloning the repository.

# Contact
Find our contact information [here](https://www.cs6.tf.fau.eu/person/lucas-weber/).