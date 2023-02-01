import os

if __name__ == "__main__":
    dir_path = "/home/center/CDCL-Marabou/resources/nnet/acasxu"
    prop_path = "/home/center/CDCL-Marabou/resources/properties/acas_property_1.txt"
    for _, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith("log") or file.endswith("path"):
                continue
            file_path = os.path.join(dir_path, file)
            json_path = os.path.join(os.path.join(dir_path, "paths"), file + ".path")
            log_path = os.path.join(os.path.join(dir_path, "log"), file)
            command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {prop_path} --save-path {json_path} --timeout 3600 > {log_path}.o.log"
            print(command)
            os.system(command)
            command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {prop_path} --load-path {json_path} --check > {log_path}.c.log"
            print(command)
            os.system(command)
    # for _, _, files in os.walk(dir_path):
    #     for file in files:
    #         if file.endswith("log"):
    #             log_path = os.path.join(os.path.join(dir_path, "log"), file)
    #             f = open(log_path)
    #             lines = f.readlines()
    #             print(log_path)
    #             for line in lines:
    #                 if "Presearch tree path" in line or "need extra info" in line:
    #                     print(line.strip("\n"))
