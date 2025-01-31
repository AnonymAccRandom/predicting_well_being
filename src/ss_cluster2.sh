#!/bin/bash

: '
This SLURM script is used to generate multiple jobs at a time for a given analysis setting.

Key parameters:
- BASE_MINUTES is adjusted dynamically based on the analysis type.
- CPUS_PER_TASK is fixed for all analyses.

Changeable variables/settings:
- prediction_model ("elasticnet", "randomforestregressor")
- crit ("state_wb", "state_pa", "state_na", "trait_wb", "trait_na", "trait_pa")
- feature_combination ("pl", "srmc", "sens", "mac", etc.)
- samples_to_include ("all", "selected", "control")
'

# Variables
PREDICTION_MODELS=("elasticnet")
CRITERIA=("wb_state")
FEATURE_COMBINATIONS=("sens_fs")
SAMPLES_TO_INCLUDE=("selected")

# Parameters
COMP_SHAP_IA_VALUES="false"
PARALLELIZE_INNER_CV="true"
PARALLELIZE_SHAP="true"
PARALLELIZE_SHAP_IA_VALUES="true"
PARALLELIZE_IMPUTATION_RUNS="true"

# New parameters
SPLIT_REPS="true"    # If "true", split repetitions into separate jobs
NUM_REPS=10          # 10
SPECIFIC_REP=""      # Set to specific rep number if needed; leave empty otherwise

BASE_MINUTES=300     # Base Time for the job, may be increased depending on analysis parameters
CPUS_PER_TASK=5      # Fixed number of CPUs per analysis
NUM_NODES=1          # If set to 1, no multi-node analysis happens
PARTITION="normal"   # cluster_2: normal

# Memory specification based on COMP_SHAP_IA_VALUES
if [ "$COMP_SHAP_IA_VALUES" == "true" ]; then
  MEMORY_REQUEST="#SBATCH --mem-per-cpu=15G"
else
  MEMORY_REQUEST=""
fi

# Removed account names for blinding
BASE_DIR="/scratch/xxx/coco_wb_ml_code/run_2811"
ENV_PATH="/scratch/xxx/coco_wb_ml_code/mpi_env/bin/activate"
PYTHONPATH_BASE="/scratch/xxx/coco_wb_ml_code"

CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Adjust the time limits based on analysis parameters (e.g., feature_combination, samples_to_include, model)
for crit in "${CRITERIA[@]}"; do
  for prediction_model in "${PREDICTION_MODELS[@]}"; do

    case $prediction_model in
      "randomforestregressor") PRED_MODEL_MULT=2 ;;
      "elasticnet") PRED_MODEL_MULT=1 ;;
    esac

    for feature_combination in "${FEATURE_COMBINATIONS[@]}"; do
      for samples_to_include in "${SAMPLES_TO_INCLUDE[@]}"; do

        case $samples_to_include in
          "all") SAMPLE_MULT=1 ;;
          "selected") SAMPLE_MULT=1 ;;
          "control") SAMPLE_MULT=1 ;;
        esac

        if [[ $feature_combination == *"sens"* ]]; then
          FEATURE_MULT=1
        else
          FEATURE_MULT=1
        fi

        TOTAL_MINUTES=$((BASE_MINUTES * PRED_MODEL_MULT * SAMPLE_MULT * FEATURE_MULT))
        HOURS=$((TOTAL_MINUTES / 60))
        MINUTES=$((TOTAL_MINUTES % 60))
        TIMELIMIT=$(printf "%02d:%02d:00" $HOURS $MINUTES)

        RESULT_DIR="${BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        mkdir -p "$RESULT_DIR"

        LOG_BASE_DIR="../slurm_logs"
        LOG_DIR="${LOG_BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        mkdir -p "$LOG_DIR"

        # Decide on DISTRIBUTED, NTASKS
        if [ "$NUM_NODES" -eq 1 ]; then
          DISTRIBUTED=false
          NTASKS=1
        else
          DISTRIBUTED=true
          NTASKS=1  # Adjust if necessary
        fi

        # Set RUN_COMMAND
        RUN_COMMAND="python main.py"

            if [ "$SPLIT_REPS" == "true" ]; then
              if [ -n "$SPECIFIC_REP" ]; then
                REP_IDS=($SPECIFIC_REP)
                NUM_REPS=1   # Override NUM_REPS when a specific rep is set
              else
                REP_IDS=($(seq 0 $(($NUM_REPS - 1))))
              fi
              for rep_id in "${REP_IDS[@]}"; do

            JOB_NAME="${feature_combination}_${samples_to_include}_${crit}_${prediction_model}_rep${rep_id}"

            JOB_LOG_NAME="job_${CURRENT_TIME}_rep${rep_id}"
            FULL_LOG_PATH_LOG="${LOG_DIR}/${JOB_LOG_NAME}.log"
            FULL_LOG_PATH_ERR="${LOG_DIR}/${JOB_LOG_NAME}.err"

            SLURM_SCRIPT="${LOG_DIR}/slurm_script_${JOB_LOG_NAME}.sh"  # Save SLURM script in the same directory

            cat > $SLURM_SCRIPT << EOF
#!/bin/bash

# SLURM Directives
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIMELIMIT
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=xxx  # removed for blinding
#SBATCH -J $JOB_NAME
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}
$MEMORY_REQUEST

# Load modules
module load lang/Python/3.11.3-GCCcore-12.3.0
module load lib/mpi4py/3.1.4-gompi-2023a  # this actually loads 3.11.5

# Activate your Python environment
unset PYTHONPATH
source $ENV_PATH
export PYTHONPATH=${PYTHONPATH_BASE}:${PYTHONPATH_BASE}/src/:\$PYTHONPATH

# Navigate to script directory
cd ${PYTHONPATH_BASE}/src/

# Debugging statements
echo "Python executable: \$(which python)"
echo "Python version: \$(python --version)"
echo "PYTHONPATH: \$PYTHONPATH"
echo "SPLIT_REPS: $SPLIT_REPS"
echo "rep_id: $rep_id"

# Run the program
$RUN_COMMAND \\
    --prediction_model "$prediction_model" \\
    --crit "$crit" \\
    --feature_combination "$feature_combination" \\
    --samples_to_include "$samples_to_include" \\
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \\
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \\
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \\
    --parallelize_shap "$PARALLELIZE_SHAP" \\
    --parallelize_imputation_runs "$PARALLELIZE_IMPUTATION_RUNS" \\
    --split_reps "$SPLIT_REPS" \\
    --rep "$rep_id" \\
    --output_path "$RESULT_DIR/"
EOF

            # Submit the SLURM job
            sbatch $SLURM_SCRIPT

          done
        else
          # When SPLIT_REPS is not "true", create a single job without repetitions
          JOB_NAME="${feature_combination}_${samples_to_include}_${crit}_${prediction_model}"

          JOB_LOG_NAME="job_${CURRENT_TIME}"
          FULL_LOG_PATH_LOG="${LOG_DIR}/${JOB_LOG_NAME}.log"
          FULL_LOG_PATH_ERR="${LOG_DIR}/${JOB_LOG_NAME}.err"

          SLURM_SCRIPT="${LOG_DIR}/slurm_script_${JOB_LOG_NAME}.sh"

          cat > $SLURM_SCRIPT << EOF
#!/bin/bash

# SLURM Directives
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --partition=normal
#SBATCH --time=$TIMELIMIT
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxx  # removed for blinding
#SBATCH -J $JOB_NAME
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}

# Load modules
module load lang/Python/3.11.3-GCCcore-12.3.0
module load lib/mpi4py/3.1.4-gompi-2023a  # this actually loads 3.11.5

# Activate your Python environment
unset PYTHONPATH
source $ENV_PATH
export PYTHONPATH=${PYTHONPATH_BASE}:${PYTHONPATH_BASE}/src/:\$PYTHONPATH

# Navigate to script directory
cd ${PYTHONPATH_BASE}/src/

# Debugging statements
echo "Python executable: \$(which python)"
echo "Python version: \$(python --version)"
echo "PYTHONPATH: \$PYTHONPATH"
echo "SPLIT_REPS: $SPLIT_REPS"

# Run the program
$RUN_COMMAND \\
    --prediction_model "$prediction_model" \\
    --crit "$crit" \\
    --feature_combination "$feature_combination" \\
        --samples_to_include "$samples_to_include" \\
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \\
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \\
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \\
    --parallelize_shap "$PARALLELIZE_SHAP" \\
    --parallelize_imputation_runs "$PARALLELIZE_IMPUTATION_RUNS" \\
    --split_reps "$SPLIT_REPS" \\
    --output_path "$RESULT_DIR/"
EOF

          # Submit the SLURM job
          sbatch $SLURM_SCRIPT

        fi

      done
    done
  done
done