import prettytable as pt
import os
import pandas as pd
import numpy as np
import warnings
from typing import List

warnings.filterwarnings("ignore")


def analyze_result(log_path):
    if os.path.exists(log_path): 
        with open(log_path) as f:
            contents = f.readlines()
            WER = contents[-6].strip()
            WER = WER.split(" ")[-1]

            CER = contents[-2].strip()
            CER = CER.split(" ")[-1]
    else:
        # print("\033[0;37;41m\t!!Data Missing!!\033[0m")
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

            
            
            if tool in ["icassp_real_mix", "icassp_real_mix_lr1e-5"]:
                path_to_log = f"models/pretrained_checkpoints/{asr}/finetuned/{dataset}/{size}/seed_{seed}/{tool}/test_infer_log.txt"
            elif tool in ["word_error_real_mix", "word_error_real_mix_lr1e-5"] :
                path_to_log = f"models/pretrained_checkpoints/{asr}/finetuned/{dataset}/{size}/seed_{seed}/{tool}/word_enhance/test_infer_log.txt"
            elif tool == "word_error_real_mix_no_seed":
                path_to_log = f"models/pretrained_checkpoints/{asr}/finetuned/{dataset}/{size}/seed_{seed}/word_error_real_mix/word_enhance_no_seed/test_infer_log.txt"
            elif tool == "icassp_real_mix_no_seed":
                path_to_log = f"models/pretrained_checkpoints/{asr}/finetuned/{dataset}/{size}/seed_{seed}/icassp_real_no_seed/test_infer_log.txt"
            else :
                raise ValueError("Undefined tool")

            try:
                WER, CER = analyze_result(path_to_log)
            except:
                # print(path_to_log)
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
            combined_df[f"{col}_t{i}"] = df[col]

    return combined_df


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

    
    # asrs = ["quartznet"]
    # datasets = ["SVBI", "HJK"]
    # tools = [ "icassp_real_mix", "word_error_real_mix"]
    
    # asrs = ["hubert"]
    # datasets = ["ABA", "SKA", "YBAA", "ZHAA", "BWC", "LXC", "NCC", "TXHC", "HJK", "HKK", "YDCK", "YKWK", "ASI", "RRBI", "SVBI", "TNI", "EBVS", "ERMS", "MBMPS", "NJS", "HQTV", "PNV", "THV", "TLV"]
    # datasets = ["BWC", "LXC"]
    # tools = [ "icassp_real_mix", "word_error_real_mix"]
    

    asrs = ["wav2vec"]
    datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    tools = [ "icassp_real_mix_lr1e-5", "word_error_real_mix_lr1e-5"]

    # asrs = ["wavlm"]
    # datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = [ "icassp_real_mix", "word_error_real_mix"]
    
    
    
    for asr in asrs :
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
                df.to_csv(f"result/RQ2/{asr}_{dataset}_{tool}.csv")

                dfs.append(df)
            
            combined_df = combine_result(wer, cer, dfs)
            combined_df.drop(columns=["Size"], inplace=True)
            
            print()
            print("ASR \t\t: ", asr)
            print("Dataset \t: ", dataset)
            print(combined_df.to_string(index=False))