import multiprocessing
import time
import json
import os


def run_verification(marabou_path, nnet, property, log_dir="./", counter=0):
    nnet_name = nnet.split('/')[-1]
    command = f"{marabou_path} {nnet} {property} > {log_dir}/{nnet_name}.log"
    os.system(command)
    with open(f'{log_dir}/{nnet_name}.log') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if 'Intra' in line:
                count += 1
        print(f"{nnet_name} : {count}")


def main():
    processes_num = 8
    pool = multiprocessing.Pool(processes=processes_num)

    marabou_path = "/home/center/CDCL-Marabou/build/Marabou"
    nnet_folder = "/home/center/CDCL-Marabou/sat_example/prop4"
    property = "/home/center/CDCL-Marabou/sat_example/prop4/acas_property_4.txt"
    result_dir = "/home/center/CDCL-Marabou/sat_example/prop4/result"
    nnets = []

    for home, dirs, files in os.walk(nnet_folder):
        for file in files:
            if file.endswith(".nnet"):
                nnets.append(file)

    counter = 0

    for nnet in nnets:
        nnet_path = os.path.join(nnet_folder, nnet)
        pool.apply_async(run_verification, (marabou_path, nnet_path, property, result_dir, counter))
        counter += 1

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
