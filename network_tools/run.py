import multiprocessing
import time
import json
import os


def worker(id, msg):
    start = time.time()
    time.sleep(5)
    end = time.time()
    print(f'{id} : {msg} use time {end - start}')


def load_json(path):
    with open(path) as f:
        json_data = json.load(f)
        data = json_data['data']
        return data


def divide_splits(json_data, process_num, tmp_dir="./json_tmp"):
    num = len(json_data) // process_num
    for i in range(process_num):
        if i != process_num - 1:
            split_data = json_data[i * num: (i + 1) * num]
        else:
            split_data = json_data[i * num:]
        new_data = {'data': split_data}
        new_path = os.path.join(tmp_dir, f"tmp_json{i}.json")
        with open(f"{new_path}", "w+") as f:
            f.write(json.dumps(new_data))


def clear_tmp(tmp_dir="./json_tmp"):
    command = f"rm -rf {tmp_dir}/*"
    os.system(command)


def run_verification(marabou_path, nnet, property, json_path, tmp_dir="./json_tmp/log", counter=0):
    command = f"{marabou_path} {nnet} {property} --json-load {json_path} > {tmp_dir}/{counter}.log --check"
    os.system(command)


if __name__ == "__main__":
    processes_num = 8
    pool = multiprocessing.Pool(processes=processes_num)

    tmp_dir = "./json_tmp"
    json_data = load_json("/home/center/CDCL-Marabou/example/relu.json")
    divide_splits(json_data, processes_num, tmp_dir)

    marabou_path = "/home/center/CDCL-Marabou/build/Marabou"
    nnet_path = "/home/center/CDCL-Marabou/example/reluBenchmark3.11155605316s_UNSAT.nnet"
    property_path = "/home/center/CDCL-Marabou/example/builtin_property.txt"
    for i in range(processes_num):
        json_path = os.path.join(tmp_dir, f'tmp_json{i}.json')
        log_path = os.path.join(tmp_dir, "log")
        pool.apply_async(run_verification, (marabou_path, nnet_path, property_path, json_path, log_path, i))

    pool.close()
    pool.join()
