# Overview
This repository contains the code to reproduce the experiments from the paper
"Uncovering Pairwise Relations in Heterogeneous Time Series Collections: A Comparative Analysis" currently
under review for the BTW2025.

Not all used datasets are publicly available, but the building datasets can be accessed:
- [KETI](https://www.kaggle.com/datasets/ranakrc/smart-building-system) or [here](https://github.com/MingzheWu418/Joint-Training)
- [SODA](https://github.com/MingzheWu418/Joint-Training/tree/main/colocation/rawdata/metadata/Soda)

Computing the relationship measures can be reproduced by installing python and the necessary dependencies
listed in the `requirements.txt` and then execute `parallelSPI.py` with the corresponding
parameters in a shell.

Already computed relationship measures may be made available upon request for a selection of datasets
and measures.

To reproduce the evaluation, comparison and plots of the papers, the results
directory contains the evaluation results for all relationship measures over all datasets.
The directory visualization contains the corresponding code files to create the plots. Run `rankSPI.py`
to recreate the plots from the evaluation metrics.