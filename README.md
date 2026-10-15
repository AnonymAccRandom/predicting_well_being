# Predicting Well-Being Across Person-, Situation-, and Country-Level Variables


This repository contains the complete analysis code. 

- Code (this repository, archived release): [CODE DOI: Zenodo, to be added]
- Processed data (N = 10,015) and codebooks: [DATA DOI: to be added]
- Code for extracting the sensing features: [DOI: to be added]
- Result files of the cross-validation (input for all figures and tables): [DOI: to be added]


## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
   - [Cloning the Repository](#cloning-the-repository)
   - [Get the Data](#get-the-data)
   - [Installing Python](#installing-python)
   - [Installing Requirements](#installing-requirements)
   - [Software Versions](#software-versions)
3. [Usage](#usage)
   - [Main Function](#main-function)
   - [Preprocessing Config](#preprocessing-config)
   - [Analysis Config](#analysis-config)
   - [Postprocessing Config](#postprocessing-config)
   - [Recommended Execution Workflow](#recommended-execution-workflow)
   - [Computation of SHAP Interaction Values](#computation-of-shap-interaction-values)
   - [Speed up Computations](#speed-up-computations)
   - [Troubleshooting](#troubleshooting)
4. [Reproducing Results](#reproducing-results)
   - [Reproducing Figures and Tables Without Rerunning the Analysis](#reproducing-figures-and-tables-without-rerunning-the-analysis)
   - [Rerunning the Full Pipeline](#general-description)
5. [Project Structure](#project-structure)
6. [License](#license)

## Introduction

To reproduce the results, please execute the following steps:
- Clone the repository
- Download the data 
- Install Python 
- Install the requirements 
- Run the analyses as described below

---

## Installation

### Cloning the Repository

To begin working with the project, you first need to copy it to your local machine. This process is called "cloning". There are two ways to clone the repository: using the command line or using a graphical user interface (GUI).

#### Using the Command Line
If you are comfortable with the command line, you can use the following commands:

```bash
git clone https://github.com/OleHaetscher/predicting_well_being.git
cd predicting_well_being
```

#### Cloning via GitHub Website

1. **Navigate to the Repository**:
   - Open your web browser and go to the repository's page on GitHub. Use the URL: (https://github.com/OleHaetscher/predicting_well_being/)

2. **Clone the Repository**:
   - Above the file list, click the green button labeled **Code**.
   - To clone the repository using HTTPS, under "Clone with HTTPS", click the clipboard icon to copy the repository URL.
   - If you’re prompted, sign in to your GitHub account.

3. **Download and Extract the Repository**:
   - After copying the URL, you can download the repository as a ZIP file to your computer.
   - Click the **Download ZIP** button from the dropdown menu under the **Code** button.
   - Once the download is complete, extract the ZIP file to your desired location on your computer to start working with the project files.

This method does not require any special software and is perfect for those unfamiliar with command-line tools. You will have a complete copy of the repository files, ready to be used with any code editor of your choice.

### Get the Data

1. **Download the data**:
   - Download the zip file **data** from the data deposit ([DATA DOI: to be added]).  

2. **Unzip the data**:
   - Unzip the **data** file without changing its structure. (e.g., unpacking the folder to the **downloads** directory) 

3. **Paste the data in the repository**:
   - Paste the unzipped **data** folder in the repository. Ensure that the data folder is placed directly within the main repository directory as shown below. 
```plaintext
predicting_well_being/
│
├── configs/
├── data/
├── ...
```

### Installing Python

1. **Visit the Python Downloads Page**:
   - Navigate to the Python downloads page for version 3.11.5 by clicking the following link: [Python 3.11.5 Download](https://www.python.org/downloads/release/python-3115/). This page contains installers for various operating systems including Windows, macOS, and Linux.

2. **Select the Appropriate Installer**:
   - Choose the installer that corresponds to your operating system. If you are using Windows, you might need to decide between the 32-bit and 64-bit versions. Most modern computers will use the 64-bit version.

3. **Run the Installer**:
   - After downloading the installer, run it by double-clicking the file. Ensure that you check the box labeled "Add Python 3.11.5 to PATH" before clicking "Install Now". This option sets up Python in your system's PATH, making it accessible from the command line.

4. **Verify the Installation**:
   - To confirm that Python has been installed correctly, open your command line interface (Terminal for macOS and Linux, Command Prompt for Windows) and type the following command:
     ```bash
     python --version
     ```
   - This command should return "Python 3.11.5". If it does not, you may need to restart your terminal or computer.


### Installing Requirements 
To ensure your setup is correctly configured to run the code, follow these steps to install the necessary dependencies:

1. **Open your terminal**: Before proceeding, make sure you are in the project's root directory.

2. **Check your Python installation**: Ensure that Python 3.11.5 is installed.

3. **Set up a virtual environment (recommended)**: To avoid conflicts with other Python projects, create a virtual environment by running:

    ```bash
    python -m venv venv
    ```

    Activate the virtual environment:
    - On Windows, use:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux, use:
        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages**: Install all dependencies listed in the `requirements.txt` file by running:

    ```bash
    pip install -r requirements.txt
    ```

Following these steps will prepare your environment for running the project without any issues related to dependencies.

### Software Versions

- Python 3.11.5
- All Python packages with their exact versions are pinned in `requirements.txt`. The main ones are
  numpy 2.0.1, pandas 2.2.2, scikit-learn 1.5.1, shap 0.46.0, shapiq 1.0.1, statsmodels 0.14.2,
  matplotlib 3.9.2, and rpy2 3.6.1.
- R 4.4.3 with the packages lavaan 0.6-19 and semTools 0.5-7, used via rpy2 for the
  measurement invariance tests only (`src/postprocessing/InvarianceTester.py`). Importing this module tries to
  install lavaan.mi and semTools.mi from CRAN. The installed package versions were not recorded; 0.6-19 and 0.5-7
  are the versions available on CRAN when the measurement invariance tests were run (July and August 2025).
- The cross-validation was run on a SLURM cluster (`src/ss_cluster1.sh`, `src/ss_cluster2.sh`).

---

## Usage

This repository contains several scripts to perform a sequence of analyses. Main tasks are 
- 'preprocessing'
- 'analysis' 
- 'postprocessing'
- The `main.py` script coordinates all the tasks, while the configurations in the `configs` directory determine which specific analyses are executed with which parameters. Thus, the config files are the only files that need to be changed when using this repository. 

### Main Function

The `main.py` script is the primary entry point to run all analyses in this repository.
Tu run the main function, do the following 
- In Editor (e.g., Pycharm): Set up a Run Configuration with the virtual environment and the main script of this project and run main. 
- In Terminal: run the script from the `src/` folder, with the repository root on the Python path (the configs
  are read relative to `src/`, and the modules are imported as `src.…`):
  ```bash
  cd src
  PYTHONPATH=.. python main.py        # Bash (macOS, Linux)
  ```
  ```powershell
  cd src
  $env:PYTHONPATH=".."; python main.py  # PowerShell (Windows)
  ```


### Preprocessing Config 

The `cfg_preprocessing.yaml` file is the user interface for the preprocessing of the data. It contains information about:
- The datasets (e.g., item names, scale endpoints of variables, mappings between categories).
- All steps needed to preprocess the raw data files into the format required for the machine-learning-based analysis.

Preprocessing steps are logged in the `logs/` folder. The resulting preprocessed data is stored in the `data/preprocessed/` folder under the name `full_data`.

To conduct preprocessing, set the following parameter in the `cfg_preprocessing.yaml` file:
```yaml
execute_preprocessing: true
```

### Analysis Config 

The `cfg_analysis.yaml` file is the user interface for the machine-learning-based analysis in this repository. Results are generated based on the current configuration and stored in the `results/` folder. \
To perform the analysis, set the following parameter in the `cfg_analysis.yaml` file:
```yaml
execute_analysis: true
```
One run of the main.py function corresponds to one analysis (10x10x10 cross-validation) for a specific configuration:
- **Criterion** [`e.g., wb_state`]
- **Samples to include** [`e.g., all`]
- **Feature configuration** [`e.g., pl_srmc`]
- **Prediction model** [`e.g., elasticnet`] 

These defining parameters can be adjusted in lines 7-10 in `cfg_analysis.yaml` to specify the analysis setting. Available settings include:

#### Types of Predictors:
- **Person-Level** [`pl`]
- **Situation-Level (ESM)** [`srmc`]
- **Situation-Level (Sensing)** [`sens`]
- **Country-Level** [`mac`]
- **Combined feature combinations** (e.g., [`pl_srmc`]) are listed under `feature_sample_combinations` in the configuration file.

#### Possible Prediction Models:
- **Elastic Net Regression** [`elasticnet`]
- **Random Forest Regression** [`randomforestregressor`]

#### Samples Inclusion Strategies:
- **All datasets** [`all`]
- **Reduced datasets** [`selected`]
- **Control** [`control`]

#### Possible Criteria:
- **Experienced well-being** [`wb_state`]
- **Experienced positive affect** [`pa_state`]
- **Experienced negative affect** [`na_state`]
- **Remembered well-being** [`wb_trait`]
- **Remembered positive affect** [`pa_trait`]
- **Remembered negative affect** [`na_trait`]

To reproduce the results, please don't change any parameters except parameters in lines 7-10.

### Postprocessing Config

The `cfg_postprocessing.yaml` file is the user interface for postprocessing the machine-learning-based results. This includes:
- **Creating tables and plots**
- **Conducting significance tests**

Results generated during postprocessing are stored in the `results/` folder and its subfolders that can be defined under `general: data_paths` in `cfg_postprocessing.yaml`.

To conduct postprocessing, adjust the `methods_to_apply` section at the top of the `cfg_postprocessing.yaml` file to specify which postprocessing steps to execute.

### Recommended Execution Workflow

1. Preprocess the raw data (or use the preprocessed file "full_data").
2. Run the machine learning analysis (for most analyses, this may be heavily time-consuming on a local computer).
3. Conduct postprocessing analyses as needed to summarize the results. 

### Computation of SHAP Interaction Values 

As we have described in the paper, we did only compute the SHAP interaction values for specific analysis settings. 
With the following description, SHAP interaction values for every analysis setting can be computed. 

Modify the following line in the `cfg_analysis.yaml` to set up the main analysis parameters (under `shap_ia_values`, set this to "true"). This is only implemented for the RFR. 
- `comp_shap_ia_values: true`

Do the computations as described in the next section. If `comp_shap_ia_values: true`, the SHAP interaction values will be computed and stored during the machine-learning-based analysis. This may be very time-consuming. 

### Speed up Computations

To speed up the computations, the machine learning-based analysis can be parallelized (on a local computer). As a default, the analysis is not parallelized.
Therefore, the `parallelize` block in `cfg_analysis.yaml` needs to be adjusted to parallelize the computations. 
I recommend these settings for local parallelization where the specific number depends on the cores available at the local device. 
All parallelized operations specified are sequential (first datasets are imputed, then inner_cv is conducted, etc.).

- `parallelize_shap: true`
- `shap_n_jobs: 5`
- `parallelize_inner_cv: true`
- `inner_cv_n_jobs: 5`
- `parallelize_shap_ia_values: true`
- `shap_ia_values_n_jobs: 5`
- `parallelize_imputation_runs: true`
- `imputation_runs_n_jobs: 5`

### Troubleshooting 

If main.py is not running as expected, consider the following steps 
- Check if you installed a virtual environment. Consider re-installing the virtual environment 
- Check if you installed the `requirements.txt` file 
- Try running the code in an editor, if it does not work from the terminal 
- Check the python version and the python path
- Check if you accidentally changed any settings in the config files. 

---

## Reproducing Results 

### Reproducing Figures and Tables Without Rerunning the Analysis

The figures and tables can be regenerated from the stored result files, without the participant data and
without rerunning the cross-validation.

1. Download the result files ([DOI: to be added]) and place the folder **results** directly within the main
   repository directory (next to `configs/` and `src/`).
2. In `cfg_postprocessing.yaml`, set the postprocessing methods to apply in `methods_to_apply` and run
   `main.py` with `execute_postprocessing: true` (see [Main Function](#main-function) and Step 4 of the
   Walk-Through below). These methods only use the result files:

| Display item | Method in `methods_to_apply` | Output |
|---|---|---|
| Figure 2 | `create_cv_results_plots` | `results/plots/combo/wb_state/` |
| Figure 3, Supplementary Figures S1 to S15 | `create_shap_plots` | `results/plots/<samples>/<crit>/<model>/` |
| Performance tables | `condense_cv_results` | `results/final_results/cv_results_*.xlsx` |
| Tables of significance tests | `conduct_significance_tests` | `results/sig_tests_1312/` |
| Correspondence of SHAP values | `shap_sanity_checks` | `results/shap_summary_tables/` |

   The regression tables (`calculate_exp_lin_models`), the descriptive statistics (`create_descriptives`), and
   the measurement invariance tests (`test_measurement_invariance`) need the preprocessed data (see
   [Get the Data](#get-the-data)).
3. The Source Data file of the article is created with `create_source_data.py`, run like `main.py` (from `src/`:
   `PYTHONPATH=.. python create_source_data.py`). It reads only the
   result files and writes one Excel file per display item, a README that maps each display item to its file,
   and a zipped folder to `results/source_data/`. The SHAP values contain one row per participant and are
   summarized per predictor in the Source Data. The display items, their result files, and the output folder
   are set in the block `create_source_data` of `cfg_postprocessing.yaml`.
4. The Supplementary Data files are created with `create_supplementary_data.py`, run like `main.py` (from
   `src/`: `PYTHONPATH=.. python create_supplementary_data.py <path to the Supplementary Information docx>`).
   They hold the
   Supplementary Tables that extended over more than one page in the revised submission (R2): the overview of
   statistical tests, the codebooks of the sensed and country-level predictors, and the descriptives (read
   from the docx, where these tables exist only), and the measurement invariance tables (built from
   `results/mi_across_datasets/` and `results/mi_across_countries/`). Output: `results/supplementary_data/`.
   The block `si_numbering` in `cfg_postprocessing.yaml` sets which R2 tables became Supplementary Data and the
   prefix of the final numbers; `src/utils/SINumbering.py` derives the final numbers from it, and both scripts
   label their files with them. The block `create_supplementary_data` holds the other settings.

Figures 2 and 3 are rendered at their final print size with the settings in `cfg_postprocessing.yaml`
(`create_cv_results_plots: plot_type: box`, `general: figure_style`). Setting `plot_type: bar` and
`figure_style: apply: false` restores the bar plots of earlier versions.

### General Description 
To reproduce the results, adjust the params in `cfg_analysis.yaml` in lines 7-10 and execute the analyses sequentially. 

#### Supplementary Analyses
The feature combinations in `cfg_analysis.yaml` with the suffix `nnse` represent the analysis excluding the neuroticism facets and 
self-esteem. 

The feature combinations in `cfg_analysis.yaml` with the suffix `npers` represent the analysis excluding all personality traits.

For the supplementary analyses using the alternative imputation strategy, no separate config entry is provided. However, 
setting `add_study_var: true` in the `imputation` section of `cfg_analysis.yaml` will trigger the alternative imputation 
procedure and generate the corresponding results.

### Walk-Through 
We provide a walk-through how to reproduce the ML-based analysis for a specific configuration. 
The **data** folder contains the raw data (data/raw) as well as the preprocessed data (data/preprocessed) for the machine learning-based analysis. 
If one only wishes to run the machine learning-based analysis, one may skip Steps 1. Step 1 will reproduce the data used for the machine learning-based analysis. 

#### Step 1: Run preprocessing
Modify the top lines in the configuration files to execute preprocessing
- `execute_preprocessing: true`
- `execute_analysis: false`
- `execute_postprocessing: false`

Run the main function after setting up the above parameters.

#### Step 2: Run analysis 
Modify the top lines in the configuration files to conduct the analysis. 
- `execute_preprocessing: false`
- `execute_analysis: true`
- `execute_postprocessing: false`

Set the parameters in lines 7-10 in `cfg_analysis.yaml` to specify the analysis setting.
- `prediction_model: elasticnet`
- `crit: wb_state`
- `feature_combination: srmc`
- `samples_to_include: all`

Run the main function after setting up the above parameters (this may take a while)

#### Step 3: Summarize analysis results 

Modify the result_dir in `utils/ClusterSummarizer.py` to the directory where the results are stored.
Run the ClusterSummarizer.py script to summarize the results.
This produces e.g., M and SD of prediction results across outer folds as reported in the paper.

#### Step 4: Run postprocessing
Note: Some steps only make sense if all results are calculated. 

Modify the top lines in the configuration files to conduct postprocessing. 
- `execute_preprocessing: false`
- `execute_analysis: false`
- `execute_postprocessing: true`

One may adjust the specific postprocessing methods to apply under `methods_to_apply` in `cfg_postprocessing.yaml`.

Run the main function once these settings are adjusted.

---

## Project Structure

```plaintext
predicting_well_being/
│
├── configs/
│   ├── cfg_analysis.yaml
│   ├── cfg_postprocessing.yaml
│   ├── cfg_preprocessing.yaml
│   └── name_mapping.yaml
│
├── data/ 
│   ├── preprocessed/  # needs to be created / inserted 
│   └── raw/  # needs to be inserted
│
├── logs/  # created when producing logs 
│
├── results/  # created when storing results
│
├── src/
│   ├── analysis/
│   │   ├── AdaptiveImputerEstimator.py
│   │   ├── BaseMLAnalyzer.py
│   │   ├── CustomIterativeImputer.py
│   │   ├── CustomScaler.py
│   │   ├── ENRAnalyzer.py
│   │   ├── Imputer.py
│   │   ├── NonLinearImputer.py
│   │   ├── PearsonFeaureSelector.py
│   │   ├── RFRAnalyzer.py
│   │   ├── SafeLogisticRegression.py
│   │   ├── ShuffledGroupKFold.py
│   │
│   ├── postprocessing/
│   │   ├── CVResultProcessor.py
│   │   ├── DescriptiveStatistics.py
│   │   ├── InvarianceTester.py
│   │   ├── LinearRegressor.py
│   │   ├── Postprocessor.py
│   │   ├── ResultPlotter.py
│   │   ├── SampleMissingsAnalyzer.py
│   │   ├── ShapProcessor.py
│   │   ├── SignificanceTesting.py
│   │   ├── SuppFileCreator.py
│   │
│   ├── preprocessing/
│   │   ├── BasePreprocessor.py
│   │   ├── CocoesmPreprocessor.py
│   │   ├── CocomsPreprocessor.py
│   │   ├── CocoutPreprocessor.py
│   │   ├── EmotionsPreprocessor.py
│   │   ├── PiaPreprocessor.py
│   │   ├── ZpidPreprocessor.py
│   │
│   ├── utils/
│   │   ├── ClusterSummarizer.py
│   │   ├── ConfigParser.py
│   │   ├── DataLoader.py
│   │   ├── DataSaver.py
│   │   ├── DataSelector.py
│   │   ├── Logger.py
│   │   ├── SanityChecker.py
│   │   ├── SINumbering.py
│   │   ├── SlurmHandler.py
│   │   ├── Timer.py
│   │   ├── utilfuncs.py
│   │
│   ├── create_source_data.py
│   ├── create_supplementary_data.py
│   ├── main.py
│   ├── ss_cluster1.sh
│   ├── ss_cluster2.sh
│
└── .gitignore
└── README.md
└── requirements.txt
```

---

## License

This repository is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). You may share and adapt the code for any purpose, provided you
give appropriate credit by citing the article and this repository.
