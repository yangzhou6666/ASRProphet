from operator import index
import numpy as np
import pandas as pd
import os

import helper
import json


if __name__ == "__main__":
    
    asrs = ["quartznet", "hubert", "wav2vec-base"]
    datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    
    ## RQ1: Measure the WER and CER using the original model
    tools = ["random", "error_model_triphone_rich","error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]

    os.makedirs("result/RQ1", exist_ok=True)
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            raw_results[asr][dataset] = {}
            for tool in tools :
                df = helper.gather_result_RQ1(asr, dataset, tool)                
                fpath = f"result/RQ1/{asr}_{dataset}_{tool.replace('/','_')}.csv"
                raw_results[asr][dataset][tool] = fpath
                df.to_csv(fpath, index=False)
    
    with open('result/RQ1.json', 'w') as fp:
        json.dump(raw_results, fp)    
    
    ## RQ2: Measure the WER and CER using the fine-tuned model            
    tools = ["random", "triphone_rich", "pure_diversity", "icassp_without_diversity_enhancing_real_mix", "icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    
    os.makedirs("result/RQ2", exist_ok=True)
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            raw_results[asr][dataset] = {}
            for tool in tools :                
                df = helper.gather_result_RQ2(asr, dataset, tool)
                fpath = f"result/RQ2/{asr}_{dataset}_{tool.replace('/','_')}.csv"
                df.to_csv(fpath, index=False)
                raw_results[asr][dataset][tool] = fpath

    with open('result/RQ2.json', 'w') as fp:
        json.dump(raw_results, fp)    
    

    ## Measure the WER and CER using the original model            
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            
            raw_results[asr][dataset] = {}
            
            for t in ["all", "seed", "dev", "seed_plus_dev", "test"] :
                wer, cer = helper.get_original_performance(asr, dataset, type=t)
                raw_results[asr][dataset][t] = {"wer":wer, "cer":cer}
    
    
    with open('result/original.json', 'w') as fp:
        json.dump(raw_results, fp)  
      
    