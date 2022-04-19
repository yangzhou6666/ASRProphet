import prettytable as pt
import os
import pandas as pd
import numpy as np
import warnings
from typing import List
warnings.filterwarnings("ignore")

import helper

if __name__ == "__main__":

    asrs = ["quartznet"]
    # datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["random", "pure_diversity", "icassp_without_diversity_enhancing_real_mix", "icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    
    # asrs = ["hubert"]
    # datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["random", "pure_diversity", "icassp_without_diversity_enhancing_real_mix", "icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    
    # asrs = ["wav2vec-base"]
    datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["random", "pure_diversity", "icassp_without_diversity_enhancing_real_mix", "icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    tools = ["triphone_rich"]
    
    
    for asr in asrs :
        
        dataframes = []
        
        for dataset in datasets :
            wer, cer = helper.get_original_performance(asr, dataset)

            dfs = []

            for tool in tools :
        
                df = helper.gather_result_RQ2(asr, dataset, tool)
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
            
            combined_df = helper.combine_result(wer, cer, dfs)
            # combined_df = combine_tools(dfs)
            combined_df.drop(columns=["Size"], inplace=True)
            
            # print()
            # print("ASR \t\t: ", asr)
            # print("Dataset \t: ", dataset)
            # print(combined_df.to_string(index=False))
            
            dataframes.append(combined_df)
            
        print()
        print()
        result = helper.combine_dataset(datasets, dataframes)
        
        # select only column with average values
        selected_column = []
        for col in result.columns.to_list() :
            if col.startswith("Wa") or col.startswith("Ca") or col.startswith("WER") or col.startswith("CER") :
                selected_column.append(col)
        result = result[selected_column]
        
        
        print(result.to_string(index=False))
