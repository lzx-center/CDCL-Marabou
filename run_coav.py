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
