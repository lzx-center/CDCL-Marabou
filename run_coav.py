import os

if __name__ == "__main__":
    dir_path = "/home/center/CDCL-Marabou/resources/nnet/coav"
    prop_path = "/home/center/CDCL-Marabou/resources/properties/builtin_property.txt"
    for _, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith("log") or file.endswith("json"):
                continue
            file_path = os.path.join(dir_path, file)
            json_path = os.path.join(os.path.join(dir_path, "json"), file + ".json")
            log_path = os.path.join(os.path.join(dir_path, "log"), file + ".center.log")
            command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {prop_path} --check > {log_path}"
            print(command)
            os.system(command)
            # command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {prop_path} --load-json {json_path} --check > {log_path}"
            # os.system(command)
            # f = open(log_path)
            # lines = f.readlines()
            # print(lines[-3:-1])
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
