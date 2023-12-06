import os
import random
import time
from NetAnalysis import deep_poly 
"""
1: Header text. This can be any number of lines so long as they begin with "//"
2: Four values: Number of layers, number of inputs, number of outputs, and maximum layer size
3: A sequence of values describing the network layer sizes. Begin with the input size, then the size of the first layer, second layer, and so on until the output layer size
4: A flag that is no longer used, can be ignored
5: Minimum values of inputs (used to keep inputs within expected range)
6: Maximum values of inputs (used to keep inputs within expected range)
7: Mean values of inputs and one value for all outputs (used for normalization)
8: Range values of inputs and one value for all outputs (used for normalization)
9+: Begin defining the weight matrix for the first layer, followed by the bias vector. The weights and biases for the second layer follow after, until the weights and biases for the output layer are defined.
"""


class TestGenerator:
    def __init__(self) -> None:
        pass

    def generator(self, file_path, property_path=None):
        layers, inputs, outputs, maximum_layer_size = 3, 2, 3, 3
        layer_nums = [inputs,3,3,outputs]

        min_val_for_input, max_val_for_input = -1.0, 1.0
        mean_val_for_input, one_val_for_output = 0.0, 0.0
        range_val_for_input, range_for_output = 1.0, 1.0

        with open(file_path, "w+") as f:

            def printf(*arg, **kwargs):
                print(*arg, **kwargs, file=f)

            print("//auto generated", file=f)
            print(f"{layers},{inputs},{outputs},{maximum_layer_size},", file=f)
            for layer_num in layer_nums:
                print(layer_num, end=",", file=f)
            print(file=f)
            print("0,", file=f)
            # line 5
            for i in range(inputs):
                print(min_val_for_input, end=",", file=f)
            print(file=f)
            # line 6
            for i in range(inputs):
                printf(max_val_for_input, end=",")
            printf()
            # line 7
            for i in range(inputs):
                printf(mean_val_for_input, end=",")
            printf(f"{one_val_for_output},")
            # line 8
            for i in range(inputs):
                printf(range_val_for_input, end=",")
            printf(f"{range_for_output},")
            # matrix
            for i in range(len(layer_nums) - 1):
                for h in range(layer_nums[i + 1]):
                    for l in range(layer_nums[i]):
                        printf(f'{random.uniform(-1, 1):.1f}', end=",")
                    printf()
                for h in range(layer_nums[i + 1]):
                    printf(f"{random.uniform(-1, 1):.1f},")
                if property_path is not None:
                    bound = random.uniform(-2, 2)
                    with open(property_path, "w+") as t:
                        def printt(*arg, **kwargs):
                            print(*arg, **kwargs, file=t)

                        printt(f"y0 >= {bound:.1f}\n+y0 -y2 >= 0")
        
        return run_exp(file_path, property_path)

def run_exp(file_path, property_path):
    command = f"/home/center/CDCL-Marabou/build/Marabou {file_path} {property_path} > ./test.log"
    print(command)
    os.system(command)
    with open("./test.log") as f:
        unsat = False
        for line in f.readlines():
            if "unsat" in line:
                unsat = True
            if "Search path" in line:
                size = int(line.split(" ")[-1].strip("\n").strip("[").strip("]"))
                print(size)
                return size
    return 0

if __name__ == "__main__":
    generator = TestGenerator()
    # while True:
    #     sz = generator.generator("./test.txt", "./test.property")
    #     # time.sleep(0.5)
    #     if sz > 3:
    #         if deep_poly("./test.txt", "./test.property"):
    #             break
    deep_poly("./test.txt", "./test.property")
    run_exp("./test.txt", "./test.property")
