import os

path = './processed'




for name in os.listdir(path):
    # Get original performance on seed_plus_dev.json
    log_path = os.path.join(path, name, 'manifests', 'quartznet_outputs', 'seed_plus_dev_infer_log.txt')
    print('>>>>>>>>>>>>>>>>')
    print(name)

    with open(log_path) as f:
        contents = f.readlines()
        WER = contents[-6].strip()

        CER = contents[-2].strip()
        print(WER)
        print(CER)