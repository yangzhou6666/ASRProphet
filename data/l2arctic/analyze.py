import os

path = './processed'

def print_result(log_path):
    if os.path.exists(log_path): 
        with open(log_path) as f:
            contents = f.readlines()
            WER = contents[-6].strip()

            CER = contents[-2].strip()
            print(WER)
            print(CER)
    else:
        print("\033[0;37;41m\t!!Data Missing!!\033[0m")

for name in os.listdir(path):
    # Get original performance on seed_plus_dev.json
    
    print('>>>>>>>>>>>>>>>>')
    print(name)

    print("------Original Performance on Seed + Dev ------")
    log_path = os.path.join(path, name, 'manifests', 'quartznet_outputs', 'seed_plus_dev_infer_log.txt')
    print_result(log_path)

    print("------Original Performance on Synthetic Seed + Dev------")
    log_path = os.path.join(path, name, 'manifests', 'quartznet_outputs', 'seed_plus_dev_infer_log_tts.txt')
    print_result(log_path)

    print("------Original Performance on Test ------")
    log_path = os.path.join(path, name, 'manifests', 'quartznet_outputs', 'original_test_infer_log.txt')
    print_result(log_path)
    
    print("------Original Performance on Synthetic Test ------")
    log_path = os.path.join(path, name, 'manifests', 'quartznet_outputs', 'original_test_infer_log_tts.txt')
    print_result(log_path)

    print("------Randome Selection------")
    log_path = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/random/test_infer_log.txt'
    print_result(log_path)

    print("------Randome TTS------")
    log_path = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/random_tts/test_infer_log.txt'
    print_result(log_path)

    print("------ICASSP 2021------")
    log_path = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/error_model/test_infer_log.txt'
    print_result(log_path)

    print("------Selected TTS------")
    log_path = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/error_model_tts/test_infer_log.txt'
    print_result(log_path)


    print("------Upper Bound------")
    log_path = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/all/test_infer_log.txt'
    print_result(log_path)


    print("------ All TTS------")
    log_path = f'../../models/pretrained_checkpoints/quartznet/finetuned/{name}/200/seed_1/all_tts/test_infer_log.txt'
    print_result(log_path)

    print("\n\n")