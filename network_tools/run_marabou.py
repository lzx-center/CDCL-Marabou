import multiprocessing
import time
import json
import os
import signal

def handler(signum, frame):
    raise TimeoutError("函数运行超时")
def run_verification(marabou_path, nnet, property, log_dir="./"):
    nnet_name = nnet.split('/')[-1]
    property_name = property.split("/")[-1]
    command = f"{marabou_path} {nnet} {property} > {log_dir}/{nnet_name}_{property_name}.log --timeout 1800"
    os.system(command)


def main():
    # processes_num = 6
    # pool = multiprocessing.Pool(processes=processes_num)

    # origin_marabou_path = "/home/center/Marabou/build/oMarabou"
    gurobi_marabou_path = "/home/center/CDCL-Marabou/build/Marabou"

    properties = [
        "/home/center/CDCL-Marabou/sat_example/prop2/acas_property_2.txt"
      # "/home/center/Marabou/resources/properties/acas_property_1.txt",
      # "/home/center/Marabou/resources/properties/acas_property_2.txt",
      # "/home/center/Marabou/resources/properties/acas_property_3.txt",
      # "/home/center/Marabou/resources/properties/acas_property_4.txt"
    ]


    # nnet_folder = "/home/center/Marabou/resources/nnet/acasxu"
    nnet_folder = "/home/center/CDCL-Marabou/sat_example/prop2"
    origin_result_dir = "/home/center/Marabou/resources/nnet/acasxu/origin_marabou"
    gurobi_result_dir = "/home/center/Marabou/resources/nnet/acasxu/gurobi_marabou"
    nnets = []

    for home, dirs, files in os.walk(nnet_folder):
        for file in files:
            if file.endswith(".nnet"):
                nnets.append(file)

    for nnet in nnets:
        nnet_path = os.path.join(nnet_folder, nnet)
        for property_ in properties:
            # run_verification()
            command = f'python ./run_divide.py -n {nnet_path} -p {property_}'
            os.system(command)
          # pool.apply_async(run_verification, (origin_marabou_path, nnet_path, property_, origin_result_dir))
          # pool.apply_async(run_verification, (gurobi_marabou_path, nnet_path, property_, gurobi_result_dir))

    # pool.close()
    # pool.join()


if __name__ == "__main__":
    main()
