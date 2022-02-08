import os

path = './processed'




for name in os.listdir(path):
    # Get original performance on seed_plus_dev.json
    
    print('>>>>>>>>>>>>>>>>')
    print(name)

    print("------Original Performance------")
    log_path = os.path.join(path, name, 'manifests', 'quartznet_outputs', 'seed_plus_dev_infer_log.txt')
    with open(log_path) as f:
        contents = f.readlines()
        WER = contents[-6].strip()

        CER = contents[-2].strip()
        print(WER)
        print(CER)
    
    print("------Randome Selection------")
    baseline_log = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/random/test_infer_log.txt'
    if os.path.exists(baseline_log): 
        with open(baseline_log) as f:
            contents = f.readlines()
            WER = contents[-6].strip()

            CER = contents[-2].strip()
            print(WER)
            print(CER)
    else:
        print("\033[0;37;41m\t!!Data Missing!!\033[0m")

    print("------ICASSP 2021------")
    baseline_log = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/error_model/test_infer_log.txt'
    if os.path.exists(baseline_log): 
        with open(baseline_log) as f:
            contents = f.readlines()
            WER = contents[-6].strip()

            CER = contents[-2].strip()
            print(WER)
            print(CER)
    else:
        print("\033[0;37;41m\t!!Data Missing!!\033[0m")


    print("------Upper Bound------")
    baseline_log = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/all/test_infer_log.txt'
    if os.path.exists(baseline_log): 
        with open(baseline_log) as f:
            contents = f.readlines()
            WER = contents[-6].strip()

            CER = contents[-2].strip()
            print(WER)
            print(CER)
    else:
        print("\033[0;37;41m\t!!Data Missing!!\033[0m")


    print("------ All TTS------")
    baseline_log = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/all_tts/test_infer_log.txt'
    if os.path.exists(baseline_log): 
        with open(baseline_log) as f:
            contents = f.readlines()
            WER = contents[-6].strip()

            CER = contents[-2].strip()
            print(WER)
            print(CER)
    else:
        print("\033[0;37;41m\t!!Data Missing!!\033[0m")

    print("\n\n")