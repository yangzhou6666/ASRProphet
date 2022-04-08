import prettytable as pt
import os
import pandas as pd
import numpy as np
import warnings
from typing import List

warnings.filterwarnings("ignore")


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


def gather_result(asr:str, dataset:str, tool:str):
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

            
            
            if tool in ["icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]:
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


def get_original_performance(asr:str, dataset:str):
    """get wer and cer from the original model
    
    Args:
        asr: asr name
        dataset: dataset name
    """
    
    path_to_log = f"data/l2arctic/processed/{dataset}/manifests/{asr}_outputs/original_test_infer_log.txt"

    try:
        WER, CER = analyze_result(path_to_log)
    except:
        # print(path_to_log)
        WER = -1
        CER = -1
         
    return WER, CER


if __name__ == "__main__":

    asrs = ["quartznet"]
    datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    tools = ["icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    # tools = ["icassp_real_mix"]
    # tools = ["asrevolve_error_model_real"]
    # tools = ["word_error_real_mix/no_word_enhance"]
    # tools = ["word_error_real_mix/word_enhance"]
    
    # asrs = ["hubert"]
    # datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    # tools = ["icassp_real_mix"]
    # tools = ["asrevolve_error_model_real"]
    # tools = ["word_error_real_mix/no_word_enhance"]
    # tools = ["word_error_real_mix/word_enhance"]
    
    # asrs = ["wav2vec-base"]
    # datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    # tools = ["icassp_real_mix"]
    # tools = ["asrevolve_error_model_real"]
    # tools = ["word_error_real_mix/no_word_enhance"]
    # tools = ["word_error_real_mix/word_enhance"]
    
    
    for asr in asrs :
        
        dataframes = []
        
        for dataset in datasets :
            wer, cer = get_original_performance(asr, dataset)

            dfs = []

            for tool in tools :
        
                df = gather_result(asr, dataset, tool)
                # print()
                # print("ASR \t\t: ", asr)
                # print("Dataset \t: ", dataset)
                # print("Original WER\t: ", wer)
                # print("Original CER\t: ", cer)
                # print("Tool \t\t: ", tool)
                # print(df)

                os.makedirs("result/RQ2", exist_ok=True)
                df.to_csv(f"result/RQ2/{asr}_{dataset}_{tool.replace('/','_')}.csv")

                dfs.append(df)
            
            combined_df = combine_result(wer, cer, dfs)
            # combined_df = combine_tools(dfs)
            combined_df.drop(columns=["Size"], inplace=True)
            
            # print()
            # print("ASR \t\t: ", asr)
            # print("Dataset \t: ", dataset)
            # print(combined_df.to_string(index=False))
            
            dataframes.append(combined_df)
            
        print()
        print()
        result = combine_dataset(datasets, dataframes)
        
        # select only column with average values
        selected_column = []
        for col in result.columns.to_list() :
            if col.startswith("Wa") or col.startswith("Ca") or col.startswith("WER") or col.startswith("CER") :
                selected_column.append(col)
        result = result[selected_column]
        
        
        print(result.to_string(index=False))
