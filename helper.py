from typing import List
import prettytable as pt
import os
import pandas as pd
import numpy as np


shorten_col_name = {
    "WER_Seed1": "W1", 
    "WER_Seed2": "W2",
    "WER_Seed3": "W3",
    "WER_Avg": "Wa",
    "CER_Seed1": "C1",
    "CER_Seed2": "C2",
    "CER_Seed3": "C3",
    "CER_Avg": "Ca"
}


def analyze_result(log_path):
    if os.path.exists(log_path): 
        with open(log_path) as f:
            contents = f.readlines()
            WER = contents[-6].strip()
            WER = WER.split(" ")[-1]

            CER = contents[-2].strip()
            CER = CER.split(" ")[-1]
    else:
        print("\033[0;37;41m\t!!Data Missing!!\033[0m")
        WER = -1
        CER = -1
    assert len(WER) < 6
    return WER, CER


def gather_result_RQ1(asr:str, dataset:str, tool:str):
    """gather wer and cer from output log
    
    Args:
        asr: asr name
        dataset: dataset name
        tool: tool name
    """
    
    sizes = [100, 200, 300, 400]

    wers = {}
    cers = {}

    for seed in [1, 2, 3]:
        wers[f"seed_{seed}"] = []
        cers[f"seed_{seed}"] = []
        
        for size in sizes:
            size = str(size)
            
            if tool in ["random", "error_model", "error_model_triphone_rich", "error_model_without_diversity_enhancing", "error_model_pure_diversity", "asrevolve_error_model_real"] :
                data_path = f"data/l2arctic/processed/{dataset}/manifests/train/{asr}/{tool}/"
                path_to_log = os.path.join(data_path, size,f"seed_{seed}", "test_out_ori_log.txt")
            elif tool in ["word_error_predictor_real/word_enhance", "word_error_predictor_real/no_word_enhance"] :
                data_path = f"data/l2arctic/processed/{dataset}/manifests/train/{asr}/word_error_predictor_real/"
                path_to_log = os.path.join(data_path, size, tool.split("/")[-1], f"seed_{seed}", "test_out_ori_log.txt")
            else :
                raise ValueError(f"Undefined tool name: {tool}")

            try:
                WER, CER = analyze_result(path_to_log)
                WER = float(WER)
                CER = float(CER)
            except:
                print(path_to_log)
                WER = -1
                CER = -1

            wers[f"seed_{seed}"].append(WER)
            cers[f"seed_{seed}"].append(CER)
        
    wers_sum = np.mean([[x for x in wers["seed_1"]], [x for x in wers["seed_2"]], [x for x in wers["seed_3"]]],
                        axis=0)
    
    wers_sum = [round(x, 2) for x in wers_sum]
    cers_sum = np.mean([[x for x in cers["seed_1"]], [x for x in cers["seed_2"]], [x for x in cers["seed_3"]]],
                        axis=0)
    cers_sum = [round(x, 2) for x in cers_sum]

    df = pd.DataFrame(data={
                        'Dataset':[dataset]*len(sizes), 
                        'Size': sizes, 
                        'WER_Seed1': wers["seed_1"],
                        'WER_Seed2': wers["seed_2"],
                        'WER_Seed3': wers["seed_3"],
                        'WER_Avg': wers_sum,
                        'CER_Seed1': cers["seed_1"],
                        'CER_Seed2': cers["seed_2"],
                        'CER_Seed3': cers["seed_3"],
                        'CER_Avg': cers_sum
    })
                        
    return df


def gather_result_RQ2(asr:str, dataset:str, tool:str):
    """gather wer and cer from output log
    
    Args:
        asr: asr name
        dataset: dataset name
        tool: tool name
    """
    data_path = f"data/l2arctic/processed/{dataset}/manifests/train/{asr}/{tool}/"

    
    sizes = [100, 200, 300, 400]

    wers = {}
    cers = {}

    
    for seed in [1, 2, 3]:
        wers[f"seed_{seed}"] = []
        cers[f"seed_{seed}"] = []
        
        for size in sizes:
            size = str(size)

            
            
            if tool in ["random", "icassp_real_mix", "icassp_without_diversity_enhancing_real_mix", "pure_diversity", "triphone_rich", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]:
                path_to_log = f"models/pretrained_checkpoints/{asr}/finetuned/{tool}/{dataset}/{size}/seed_{seed}/test_infer_log.txt"
            else :
                raise ValueError("Undefined tool")

            try:
                WER, CER = analyze_result(path_to_log)
                WER = float(WER)
                CER = float(CER)
            except:
                print(path_to_log)
                WER = -1
                CER = -1
            
            wers[f"seed_{seed}"].append(WER)
            cers[f"seed_{seed}"].append(CER)
        # wers[f"seed_{seed}"].append(np.mean(wers[f"seed_{seed}"]))
        # cers[f"seed_{seed}"].append(np.mean(cers[f"seed_{seed}"]))
        
        
    wers_sum = np.mean([[float(x) for x in wers["seed_1"]], [float(x) for x in wers["seed_2"]], [float(x) for x in wers["seed_3"]]],
                        axis=0)
    
    wers_sum = [round(x, 2) for x in wers_sum]
    cers_sum = np.mean([[float(x) for x in cers["seed_1"]], [float(x) for x in cers["seed_2"]], [float(x) for x in cers["seed_3"]]],
                        axis=0)
    cers_sum = [round(x, 2) for x in cers_sum]

    df = pd.DataFrame(data={
                        'Dataset':[dataset]*len(sizes), 
                        'Size': sizes, 
                        'WER_Seed1': wers["seed_1"],
                        'WER_Seed2': wers["seed_2"],
                        'WER_Seed3': wers["seed_3"],
                        'WER_Avg': wers_sum,
                        'CER_Seed1': cers["seed_1"],
                        'CER_Seed2': cers["seed_2"],
                        'CER_Seed3': cers["seed_3"],
                        'CER_Avg': cers_sum
    })
                        
    return df


def combine_tools(datas: List[pd.DataFrame])-> pd.DataFrame:
    """combine results from various tools into one dataframe
    :param datas: result from various tool
    :return: horisontally combined dataframe
    """
    data = {"Size": datas[0]["Size"]}

    combined_df = pd.DataFrame(data)

    for i, df in enumerate(datas):
        for col in ['WER_Seed1', 'WER_Seed2', 'WER_Seed3', 'WER_Avg', 'CER_Seed1', 'CER_Seed2', 'CER_Seed3', 'CER_Avg'] :
            combined_df[f"{shorten_col_name[col]}_t{i}"] = df[col]

    return combined_df


def combine_dataset(dataset_names, datas: List[pd.DataFrame])-> pd.DataFrame:
    """combine results from various datasets into one dataframe
    :param datas: result from various dataset
    :return: vertically combined dataframe
    """
    
    for i in range(len(datas)) :
        datas[i]["Name"] = dataset_names[i]
    
        cols = datas[i].columns.to_list()
        cols = cols[-1:] + cols[:-1]
        datas[i] = datas[i][cols]
        
    result = pd.concat(datas)

    return result


def combine_result(wer:str, cer:str, datas: List[pd.DataFrame])-> pd.DataFrame:
    """combine results from various tools into one dataframe
    :param wer: wer of the original model
    :param cer: cer of the original model
    :param datas: result from various tool
    :return: combined dataframe
    """
    data = {
            "Size": datas[0]["Size"],
            "WER": [wer]*len(datas[0]["Dataset"]),
            "CER": [cer]*len(datas[0]["Dataset"])
            }

    combined_df = pd.DataFrame(data)

    for i, df in enumerate(datas):
        for col in ['WER_Seed1', 'WER_Seed2', 'WER_Seed3', 'WER_Avg', 'CER_Seed1', 'CER_Seed2', 'CER_Seed3', 'CER_Avg'] :
            combined_df[f"{shorten_col_name[col]}_t{i}"] = df[col]

    return combined_df



def get_original_performance(asr:str, dataset:str, type="test"):
    """get wer and cer from the original model
    
    Args:
        asr: asr name
        dataset: dataset name
        type: seed, test, all test cases in the dataset
    """
    
    path_to_log = f"data/l2arctic/processed/{dataset}/manifests/{asr}_outputs/original_{type}_infer_log.txt"

    try:
        WER, CER = analyze_result(path_to_log)
        WER = float(WER)
        CER = float(CER)
    except:
        print(path_to_log)
        WER = -1
        CER = -1
         
    return WER, CER

