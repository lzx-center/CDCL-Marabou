import argparse
import os.path
import time
import datetime
from nnet import NNet
from tqdm import tqdm

def init_parser():
    parser = argparse.ArgumentParser(description='Process marabou input.')
    parser.add_argument('-n', '--net_path', type=str,
                        help='input net file path')
    parser.add_argument('-p', '--prop_path', type=str,
                        help='input property file path')
    return parser


def convert_time_hmsms(seconds):
    """将秒数转换为时:分:秒:毫秒的格式"""
    td = datetime.timedelta(seconds=seconds)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{td.microseconds // 1000:03d}"


def divide(net_path, prop_path):
    net = NNet(net_path)
    net.load_property(prop_path)
    domains = []
    net.recursive_split(net.norm_mins, net.norm_maxes, 10, domains)
    start_time = time.time()
    sat = False
    time_out = 0
    nnet_name = net_path.split('/')[-1]
    property_name = prop_path.split('/')[-1]
    for i in tqdm(range(len(domains)), desc=f'{nnet_name}.{property_name}', leave=True):
        new_prop_folder = prop_path[:-4]
        if not os.path.exists(new_prop_folder):
            os.mkdir(new_prop_folder)
        new_prop_path = f'{prop_path[:-4]}/sub{i}.txt'
        lower, upper = domains[i]
        net.dump_property(new_prop_path, lower, upper)
        # run marabou
        marabou_path = '/home/center/CDCL-Marabou/build/Marabou'
        command = f'{marabou_path} {net_path} {new_prop_path}  > {new_prop_path}.log'
        os.system(command)

        log = open(f'{new_prop_path}.log')
        for line in log.readlines():
            if 'Input assignment:' in line:
                print(f'{new_prop_path}.log')
                sat = True
            elif "Timeout" in line:
                time_out += 1
        if sat:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("------" * 10)
    print(
        f"net: {nnet_name}\nproperty: {property_name}\nTotal time: {elapsed_time:.6f} seonds({convert_time_hmsms(elapsed_time)})\nresults:{'sat' if sat else 'unsat'}")
    print(f"timeout: {time_out}")


if __name__ == "__main__":
    argument_parser = init_parser()
    args = argument_parser.parse_args()
    divide(args.net_path, args.prop_path)
