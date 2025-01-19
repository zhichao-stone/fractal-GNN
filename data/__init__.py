from .loading import LOAD_FUNCTION_MAP, NODE_PRED_DATA, LINK_PRED_DATA, GRAPH_PRED_DATA, RAW_DIR
from .data_augmentation import DataAugmentator



def load_dataset(dataset_name:str, raw_dir:str=RAW_DIR):
    if dataset_name in NODE_PRED_DATA or dataset_name in LINK_PRED_DATA or dataset_name in GRAPH_PRED_DATA:
        data = LOAD_FUNCTION_MAP[dataset_name](raw_dir=raw_dir)
        return data
    else:
        raise Exception(f"Unsurpported Dataset: {dataset_name}")


