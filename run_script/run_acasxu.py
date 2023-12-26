import os

import subprocess
import shlex
import time
import signal


def run_command_with_timeout(command, timeout):
    try:
        # 执行命令行程序
        process = subprocess.Popen(command, shell=True)

        # 等待命令行程序结束或超时
        start_time = time.time()
        while time.time() - start_time < timeout and process.poll() is None:
            time.sleep(0.1)

        if process.poll() is None:
            # 超时，发送SIGTERM信号终止进程
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                # 进程仍未结束，发送SIGKILL信号
                process.kill()
                process.wait()

        # 返回命令行程序的返回码和输出信息
        return_code = process.poll()

        return return_code
    except Exception as e:
        print("An error occurred:", e)
        return None, None


def run(timeout=300):
    dir_path = "/home/center/CDCL-Marabou/resources/nnet/acasxu"
    prop_base = "/home/center/CDCL-Marabou/resources/properties/"
    props = ["acas_property_1.txt", "acas_property_2.txt", "acas_property_3.txt", "acas_property_4.txt"]
    for _, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith("log") or file.endswith("json"):
                continue
            for prop in props:
                prop_path = os.path.join(prop_base, prop)
                file_path = os.path.join(dir_path, file)
                center_log_path = os.path.join("/home/center/CDCL-Marabou/run_script/log2",
                                               file + "." + prop + ".log" + ".center.log")
                center_command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {prop_path} --learn-clause " \
                                 f" > {center_log_path} "
                log_path = os.path.join("/home/center/CDCL-Marabou/run_script/log3", file + "." + prop + ".log")
                command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {prop_path}  > {log_path}"

                check_log_path = os.path.join("/home/center/CDCL-Marabou/run_script/log2",
                                              file + "." + prop + ".log" + ".check.log")
                check_command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {prop_path} --check > {check_log_path}"
                print(file, prop_path)
                run_command_with_timeout(command, timeout)
                # run_command_with_timeout(center_command, timeout)
                # run_command_with_timeout(check_command, timeout)
                # process = subprocess.Popen(command, shell=True)


if __name__ == "__main__":
    run()
