# Predicting Well-Being Across Personal, Situational, and Societal Factors 

This repository contains the complete analysis code. 


## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
   - [Cloning the Repository](#cloning-the-repository)
   - [Get the Data](#get-the-data)
   - [Installing Python](#installing-python)
   - [Installing Requirements](#installing-requirements)
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
git clone https://github.com/AnonymAccRandom/predicting_well_being.git
cd predicting_well_being
```

#### Cloning via GitHub Website

1. **Navigate to the Repository**:
   - Open your web browser and go to the repository's page on GitHub. Use the URL: (https://github.com/AnonymAccRandom/predicting_well_being/)

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
   - Navigate to the OSF project that contains the data by clicking on the link provided in the manuscript. Download the zip file **data**.  

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
- In Terminal (e.g., Bash, Powershell): Just run the script using `python src/main.py`


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
- **Personal** [`pl`]
- **Situational ESM** [`srmc`]
- **Situational Sensing** [`sens`]
- **Societal** [`mac`]
- **Combined feature combinations** (e.g., [`pl_srmc`]) are listed in lines 39-57 of the configuration file.

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

Results generated during postprocessing are stored in the `results/` folder and its subfolders that can be defined in lines 80-85 of `cfg_postprocessing.yaml`.

To conduct postprocessing, adjust the `methods_to_apply` section (lines 7-16) in the `cfg_postprocessing.yaml` file to specify which postprocessing steps to execute.

### Recommended Execution Workflow

1. Preprocess the raw data (or use the preprocessed file "full_data").
2. Run the machine learning analysis (for most analyses, this may be heavily time-consuming on a local computer).
3. Conduct postprocessing analyses as needed to summarize the results. 

### Computation of SHAP Interaction Values 

As we have described in the paper, we did only compute the SHAP interaction values for specific analysis settings. 
With the following description, SHAP interaction values for every analysis setting can be computed. 

Modify the following line in the `cfg_analysis.yaml` to set up the main analysis parameters (line 139, set this to "true"). This is only implemented for the RFR. 
- `comp_shap_ia_values: true`

Do the computations as described in the next section. If `comp_shap_ia_values: true`, the SHAP interaction values will be computed and stored during the machine-learning-based analysis. This may be very time-consuming. 

### Speed up Computations

To speed up the computations, the machine learning-based analysis can be parallelized (on a local computer). As a default, the analysis is not parallelized.
Therefore, lines 702-713 in `cfg_analysis.yaml` need to be adjusted to parallelize the computations. 
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

### General Description 
To reproduce the results, adjust the params in `cfg_analysis.yaml` in lines 7-10 and execute the analyses sequentially. 

#### Supplementary Analyses
The feature combinations in `cfg_analysis.yaml` with the suffix `nnse` represent the analysis excluding the neuroticism facets and 
self-esteem. 

The feature combinations in `cfg_analysis.yaml` with the suffix `npers` represent the analysis excluding all personality traits.

For the supplementary analyses using the alternative imputation strategy, no separate config entry is provided. However, 
setting `add_study: true` in the `Imputation` section of `cfg_analysis.yaml will trigger the alternative imputation 
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

One may adjust the specific postprocessing methods to apply in lines 7-16 in `cfg_postprocessing.yaml`.

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
│   │   ├── PearsonFeatureSelector.py
│   │   ├── RFRAnalyzer.py
│   │   ├── SafeLogisticRegression.py
│   │   ├── ShuffledGroupKFold.py
│   │
│   ├── postprocessing/
│   │   ├── CVResultProcessor.py
│   │   ├── DescriptiveStatistics.py
│   │   ├── InvarianceTester.py
│   │   ├── LinearRegressor.py
│   │   ├── PostProcessor.py
│   │   ├── ResultPlotter.py
│   │   ├── SampleMissingsAnalyzer.py
│   │   ├── ShapProcessor.py
│   │   ├── SignificanceTesting.py
│   │   ├── SuppFileCreator.py
│   │
│   ├── preprocessing/
│   │   ├── BasePreprocessor.py
│   │   ├── CocosemPreprocessor.py
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
│   │   ├── SlurmHandler.py
│   │   ├── Timer.py
│   │   ├── utilfuncs.py
│   │
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

CC-By Attribution 4.0 International
