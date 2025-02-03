# analyzing-pandemic-response

CIC COVID Student Working Group

## Status

* Version 1: Worked on Winter 2023 to Spring 2024. See issue for existing status.

## Datasets
* OxCGRT Policy [[Source](https://github.com/OxCGRT/covid-policy-dataset)]
* Census bureau (deomographic characteristics) [[Source](https://data.census.gov/all?g=010XX00US&y=2020)]
* Google Mobility [[Source](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh/about_data)]
* Hospital Capacity [[Source](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh/about_data)]
* Epidemiological [[Source](https://github.com/lisphilar/covid19-sir)]

## Exploratory Data Analysis 

* Basic EDA (confirm assumptions, missing data, distributions, etc.)
* Correlation analysis, particularly of policies 
* Clustering of states by policy, demographics (how approaches differ)
* Time series analysis (differencing, trend, seasonality, autocorrelations)

## Modeling 

* Case, death rates 
* R0 reproductive rate (SIR model)
* Adjust for confounding, add policy vec in ARIMA 

## Analysis

* Simulate and compare outcomes under national v. state policy, how much improvement over baseline national guidelines? 
* Cluster by approach, simulate under respective approaches and compare trajectories 
* Analyze trajectories: estimate R0 reproductive rate w.r.t. \beta/\gamma/\tau values, 
* Sensitivity analysis, compare to other methods if possible

## Sources

1. Hernan, Miguel A., Robins, James M. Causal Inference: What if? Chapman & Hall. 2024: Chapters on g-methods.
2. Hyndman, Rob J., Athanasopoulos, George. Forecasting: Principles and Practice: Chapters on ARIMA models, exponential smoothing. 

