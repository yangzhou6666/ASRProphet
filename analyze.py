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
    
    sizes = [50, 75, 100, 150, 200, 300, 400, 500]

    wers = {}
    cers = {}

    
    for seed in [1, 2, 3]:
        wers[f"seed_{seed}"] = []
        cers[f"seed_{seed}"] = []
        
        for size in sizes:
            size = str(size)
            
            if tool == "error_model":
                path_to_log = os.path.join(data_path, size,f"seed_{seed}", "test_out_ori_log.txt")
            elif tool == "word_error_predictor_real" :
                path_to_log = os.path.join(data_path, size, "word_enhance", f"seed_{seed}", "test_out_ori_log.txt")
            else :
                raise ValueError("Undefined tool")

            try:
                WER, CER = analyze_result(path_to_log)
            except:
                print(path_to_log)
                WER = -1
                CER = -1
            
            wers[f"seed_{seed}"].append(WER)
            cers[f"seed_{seed}"].append(CER)
        
            # df = df.append({
            #             "Dataset": dataset,
            #             "Seed": seed, 
            #             "Size": size, 
            #             "WER": WER, 
            #             "CER": CER
            #             }
            #             , ignore_index=True)
    
    df = pd.DataFrame(data={
                        'Dataset':[dataset]*len(sizes), 
                        'Size': sizes, 
                        'WER_Seed1': wers["seed_1"],
                        'WER_Seed2': wers["seed_2"],
                        'WER_Seed3': wers["seed_3"],
                        'CER_Seed1': cers["seed_1"],
                        'CER_Seed2': cers["seed_2"],
                        'CER_Seed3': cers["seed_3"]
    })
                        
    return df


if __name__ == "__main__":

    asrs = ["deepspeech"]
    datasets = ["ASI", "RRBI"]
    tools = ["error_model", "word_error_predictor_real"]
    
    asrs = ["deepspeech"]
    datasets = ["ASI"]
    tools = ["error_model"]

    for asr in asrs :
        for dataset in datasets :
            for tool in tools :
        
                print()
                print("ASR \t\t: ", asr)
                print("Dataset \t: ", dataset)
                print("Tool \t\t: ", tool)

                df = gather_result(asr, dataset, tool)
                print(df)

                os.makedirs("result", exist_ok=True)
                df.to_csv(f"result/{asr}_{dataset}_{tool}.csv")