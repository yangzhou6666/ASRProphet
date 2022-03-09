import prettytable as pt
import os
import pandas as pd
import warnings
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
    df = pd.DataFrame(columns=['Dataset', 'Seed', 'Size', 'WER', 'CER'])
    
    for seed in [1, 2, 3]:
        
        for size in [50, 75, 100, 150, 200, 300, 400, 500]:
            size = str(size)
            path_to_log = os.path.join(data_path, size,f"seed_{seed}", "test_out_ori_log.txt")
            try:
                WER, CER = analyze_result(path_to_log)
            except:
                print(path_to_log)
                WER = -1
                CER = -1
        
            df = df.append({
                        "Dataset": dataset,
                        "Seed": seed, 
                        "Size": size, 
                        "WER": WER, 
                        "CER": CER
                        }
                        , ignore_index=True)

    return df


if __name__ == "__main__":
    asr = "deepspeech"
    dataset = "ASI"
    tool = "error_model"
    df = gather_result(asr, dataset, tool)

    print(df)