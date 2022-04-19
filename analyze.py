from typing import List
import prettytable as pt
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import helper


if __name__ == "__main__":

    ## RQ1 
    # Measure the WER and CER using the original model
    
    asrs = ["quartznet"]
    # datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["error_model_triphone_rich", "error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]
    
    # asrs = ["hubert"]
    datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["error_model_triphone_rich", "error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]
    
    # asrs = ["wav2vec-base"]
    datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    # tools = ["error_model_triphone_rich","error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]
    tools = ["random", "error_model_triphone_rich"]
    
    
    for asr in asrs :
        
        dataframes = []
        
        for dataset in datasets :

            dfs = []

            for tool in tools :
        
                df = helper.gather_result_RQ1(asr, dataset, tool)
                
                # print()
                # print("ASR \t\t: ", asr)
                # print("Dataset \t: ", dataset)
                # print("Tool \t\t: ", tool)
                # print(df)

                os.makedirs("result", exist_ok=True)
                df.to_csv(f"result/{asr}_{dataset}_{tool.replace('/','_')}.csv")

                dfs.append(df)

            combined_df = helper.combine_tools(dfs)
            
            # print()
            # print("ASR \t\t: ", asr)
            # print("Dataset \t: ", dataset)
            # combined_df.drop(columns=["Size"], inplace=True)
            # print(combined_df.to_string(index=False))
            
            dataframes.append(combined_df)
            
        print()
        print()
        result = helper.combine_dataset(datasets, dataframes)
        
        # select only column with average values
        selected_column = []
        for col in result.columns.to_list() :
            if col.startswith("Wa") or col.startswith("Ca") :
                selected_column.append(col)
        result = result[selected_column]
        
        print(result.to_string(index=False))

