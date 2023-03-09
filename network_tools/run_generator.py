from ClauseGenerator import ClauseGenerator
import multiprocessing
import os


def run_verification(marabou_path, nnet, property, log_dir="./"):
    nnet_name = nnet.split('/')[-1]
    # generate clause
    clause_generator = ClauseGenerator(nnet, property)
    json_file = os.path.join(log_dir, f'{nnet_name}.json')

    clause_generator.generate_clause(20, 1000, json_file)

    command = f"{marabou_path} {nnet} {property} > {log_dir}/{nnet_name}.log --check --load-json {json_file}"
    print(command)
    os.system(command)


def main():
    processes_num = 8
    pool = multiprocessing.Pool(processes=processes_num)

    marabou_path = '/home/center/CDCL-Marabou/build/Marabou'

    net_folders = [
        "/home/center/CDCL-Marabou/sat_example/prop2/",
        "/home/center/CDCL-Marabou/sat_example/prop3",
        "/home/center/CDCL-Marabou/sat_example/prop4",
        # "/home/center/CDCL-Marabou/resources/nnet/acasxu",
    ]
    property_paths = [
        "/home/center/CDCL-Marabou/sat_example/prop2/acas_property_2.txt",
        "/home/center/CDCL-Marabou/sat_example/prop3/acas_property_3.txt",
        "/home/center/CDCL-Marabou/sat_example/prop4/acas_property_4.txt",
    ]

    for i in range(len(property_paths)):
        net_folder = net_folders[i]
        for home, dirs, files in os.walk(net_folder):
            for file in files:
                if file.endswith(".nnet"):
                    nnet_path = os.path.join(net_folder, file)
                    result_dir = os.path.join(net_folder, "h_results_marabou")
                    # print(result_dir, nnet_path)
                    # run_verification(marabou_path, nnet_path, property_paths[i], result_dir)
                    pool.apply_async(run_verification, (marabou_path, nnet_path, property_paths[i], result_dir))

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
