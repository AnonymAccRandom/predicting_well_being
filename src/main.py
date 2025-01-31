"""
Main analysis script. If specified, this executes
   - Data preprocessing
   - ML-based analysis (integrating slurm job variables)
   - Results postprocessing
Note: In the final version
    - we did not use mpi4py for the ML-based analysis
    - we always split repetitions into separate jobs and aggregated
      the results on the cluster (see ClusterSummarizer.py)
"""
import os

import numpy as np
import pandas as pd

from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.SlurmHandler import SlurmHandler

if __name__ == "__main__":
    # Load configurations and parse SLURM args before importing mpi4py
    data_loader = DataLoader()
    data_saver = DataSaver()
    base_cfg_path = "../configs/"

    cfg_preprocessing = data_loader.read_yaml(
        os.path.join(base_cfg_path, "cfg_preprocessing.yaml")
    )
    cfg_analysis = data_loader.read_yaml(
        os.path.join(base_cfg_path, "cfg_analysis.yaml")
    )
    cfg_postprocessing = data_loader.read_yaml(
        os.path.join(base_cfg_path, "cfg_postprocessing.yaml")
    )
    name_mapping = data_loader.read_yaml(
        os.path.join(base_cfg_path, "name_mapping.yaml")
    )

    slurm_handler = SlurmHandler()
    args = slurm_handler.get_slurm_vars()
    cfg_analysis_updated = slurm_handler.update_cfg_with_slurm_vars(
        cfg_analysis=cfg_analysis, args=args
    )
    use_mpi = cfg_analysis_updated["use_mpi4py"]
    split_reps = cfg_analysis_updated["split_reps"]
    rep = args.rep

    # Only import mpi, if specified in the config
    if use_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    print("Use MPI:", use_mpi)
    print(f"Hello from rank {rank} out of {size}", flush=True)
    print(f"Starting main.py on rank {rank} out of {size}")

    # Rest of your imports
    from src.analysis.ENRAnalyzer import ENRAnalyzer
    from src.analysis.RFRAnalyzer import RFRAnalyzer
    from src.postprocessing.Postprocessor import Postprocessor
    from src.preprocessing.CocoesmPreprocessor import CocoesmPreprocessor
    from src.preprocessing.CocomsPreprocessor import CocomsPreprocessor
    from src.preprocessing.CocoutPreprocessor import CocoutPreprocessor
    from src.preprocessing.EmotionsPreprocessor import EmotionsPreprocessor
    from src.preprocessing.PiaPreprocessor import PiaPreprocessor
    from src.preprocessing.ZpidPreprocessor import ZpidPreprocessor

    # >>>Preprocessing<<< (all ranks execute)
    if cfg_preprocessing["execute_preprocessing"]:
        datasets_included = cfg_preprocessing["general"]["datasets_to_be_included"]
        df_lst = []

        preprocessor_mapping = {
            "cocoesm": CocoesmPreprocessor,
            "cocoms": CocomsPreprocessor,
            "cocout": CocoutPreprocessor,
            "emotions": EmotionsPreprocessor,
            "pia": PiaPreprocessor,
            "zpid": ZpidPreprocessor,
        }

        for dataset_name in datasets_included:
            if dataset_name in preprocessor_mapping:
                preprocessor_class = preprocessor_mapping[dataset_name]
                preprocessor = preprocessor_class(cfg_preprocessing=cfg_preprocessing)
                df = preprocessor.apply_preprocessing_methods()
                df_lst.append(df)
            else:
                raise ValueError(
                    f"Warning: No preprocessor defined for dataset '{dataset_name}'"
                )

        df = pd.concat(df_lst, axis=0, ignore_index=False, sort=False, join="outer")

        if cfg_preprocessing["general"]["store_data"]:
            path_to_full_data = os.path.join(
                cfg_preprocessing["general"]["path_to_preprocessed_data"], "full_data"
            )
            data_saver.save_pickle(df, path_to_full_data)

    else:
        df = None

    # >>>Analysis<<< (all ranks participate)
    if cfg_analysis_updated["execute_analysis"]:
        slurm_handler = SlurmHandler()

        # Load data if required (if mpi4py: all ranks need to load the data)
        if cfg_analysis_updated["load_data"]:
            path_to_full_data = os.path.join(
                cfg_preprocessing["general"]["path_to_preprocessed_data"], "full_data"
            )
            df = data_loader.read_pkl(path_to_full_data)
        else:
            pass

        # Handle SLURM job variables
        if os.getenv("SLURM_JOB_ID"):
            total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            print("total_cores in main:", total_cores)

            cfg_analysis_updated = slurm_handler.allocate_cores(
                cfg_analysis=cfg_analysis_updated, total_cores=total_cores
            )
            cfg_analysis_updated = slurm_handler.sanity_checks_cfg_cluster(
                cfg_analysis=cfg_analysis_updated
            )
            output_dir = args.output_path

            if not use_mpi and split_reps and args.rep is not None:
                rep = args.rep
            else:
                rep = None

        else:
            output_dir = slurm_handler.construct_local_output_path(
                cfg_analysis=cfg_analysis_updated
            )
            rep = None

        np.random.seed(cfg_analysis_updated["random_state"])
        prediction_model = cfg_analysis_updated["params"]["prediction_model"]

        if prediction_model == "elasticnet":
            enr_analyzer = ENRAnalyzer(
                cfg_analysis=cfg_analysis_updated,
                output_dir=output_dir,
                df=df,
                rank=rank,
                rep=rep,
            )
            enr_analyzer.apply_methods(comm=comm)

        elif prediction_model == "randomforestregressor":
            rfr_analyzer = RFRAnalyzer(
                cfg_analysis=cfg_analysis_updated,
                output_dir=output_dir,
                df=df,
                rank=rank,
                rep=rep,
            )
            rfr_analyzer.apply_methods(comm=comm)

        else:
            raise ValueError(f"Model {prediction_model} not implemented")

    # >>>Postprocessing<<<
    if cfg_postprocessing["execute_postprocessing"]:
        if use_mpi and rank == 0:
            postprocessor = Postprocessor(
                cfg_preprocessing=cfg_preprocessing,
                cfg_analysis=cfg_analysis_updated,
                cfg_postprocessing=cfg_postprocessing,
                name_mapping=name_mapping,
            )
            postprocessor.apply_methods()

        elif not use_mpi:
            postprocessor = Postprocessor(
                cfg_preprocessing=cfg_preprocessing,
                cfg_analysis=cfg_analysis_updated,
                cfg_postprocessing=cfg_postprocessing,
                name_mapping=name_mapping,
            )
            postprocessor.apply_methods()
