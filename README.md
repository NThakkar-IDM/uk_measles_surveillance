# uk_measles_surveillance
Estimating dynamic measles reporting rates in the pre-vaccine-era UK

This is a repository of Python 3.8 code associated with the preprint *A modeling approach for estimating dynamic measles case detection rates*, 2022. In the paper, I fit a measles transmission model to the 60 city, United Kingdom data set, and I use that model to better understand measles case detection over time.

The main three scripts are:
1. `FullDataVis.py`, which makes the paper's first figure.
2. `SurvivalAnalysis.py`, which makes the paper's second figure and a serialized pandas dataframe used as input into the full model.
3. `TransmissionModel.py`, which fits the full model and makes the paper's remaining figures.

The Python environment is managed through [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and described in `environment.yml`.
