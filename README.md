# Welcome

This is a research project I conducted with one of my professors, Dr. Alexander Franks, during Summer 2024. Our work was supported by the CCS Summer Undergraduate Research Fellowship.

Our goal was to develop new metrics to evaluate the skill of basketball players. We approached this through a causal inference framework. 
This project was intended to become a package, and much of the code is designed to support this, though we have no intention of creating a package until
we achieve satisfactory performance.

Much of the code, as well as comments outlining the research, is contained in a Jupyter notebook file. It can be found here: [code/analyze/propensity_scores.ipynb](code/analyze/propensity_scores.ipynb)

# Directory

## code

The majority of code for this project.

### analyze

- `propensity_scores.ipynb`: While originally intended to be a notebook which only produced propensity score models, it evolved to contain the code for most of the project. This notebook contains our metrics as well as its validation schemes.
- `fit_rapm_model.ipynb`: A notebook containing a baseline RAPM model.
- `sub_cache.json`: A cache, for use in propensity_scores.

### preprocess

- `ryurko_model.qmd`: A Quarto file written by Ron Yurko explaining his R script.
- `ryurko_prepare_data.R`: A script written by Ron Yurko converting raw NBA data into stints.

## data

The data of this project. Primarily contains 2022–2023 and 2023–2024 seasonal data, including NBA data and box score statistics.

## design_matrices

Design matrices for the RAPM model.

## misc

Personal notes.

## old

Old code I didn't want to delete.

## results

Figures I downloaded. This folder does not contain all figures and was more for personal use.
