import NetAnalysis
import nnet
import torch
import random
from attack import InputGenerate, attack_single_sample
import json


class ClauseGenerator(NetAnalysis.DeepPolyAnalysis):
    def __init__(self, file_name, property_file=None) -> None:
        super().__init__(file_name)
        if property_file:
            self.load_property(property_file)
        self.candidates = []

    # aid functions
    def check_property(self, input):
        out = self.forward(input)
        for equation in self.property_equation:
            sc = 0
            for index, coeff in equation.lhs.items():
                sc += coeff * out[index]
            if equation.type == nnet.Equation.EquationType.ge:
                if sc < equation.scalar:
                    return True
            else:
                if sc > equation.scalar:
                    return True
        print(f"Already satify! input: {input} output: {out}")
        for i in range(self.num_inputs()):
            print(f"x{i} [{self.norm_mins[i]}, {self.norm_maxes[i]}]")
        exit()
        return False

    def desent_input(self, candidates, target_states, rate=0.1):
        print(candidates, target_states)
        max_step = 20
        while max_step:
            max_step -= 1

    def encode_state_to_num(self, state):
        num_state = 0
        for i in range(len(state)):
            if state[i]:
                num_state |= (1 << i)
        return num_state

    def decode_num_to_state(self, num_state, length):
        target_states = []
        for i in range(length):
            if num_state & (1 << i):
                target_states.append(True)
            else:
                target_states.append(False)
        return target_states

    def find_appear_pattern(self, candidates_index, states):
        candidates_states = [[state[index] for index in candidates_index] for state in states]
        appear_states = set()
        length = len(candidates_index)
        for state in candidates_states:
            num_state = self.encode_state_to_num(state)
            appear_states.add(num_state)
        print("appear states:", appear_states)

        ret = []
        for num_state in appear_states:
            candidates = [self.candidates[index] for index in candidates_index]
            target_states = self.decode_num_to_state(num_state, length)
            assert len(candidates) == len(target_states)
            ret.append([(candidates[i], target_states[i]) for i in range(len(candidates))])
        return ret

    # candidate selector
    def get_layer_candidates_num(self, candidates_num):
        nums = [int(candidates_num / self.numLayers) for _ in range(1, self.numLayers + 1)]
        nums[1] += candidates_num - sum(nums)
        assert (sum(nums) == candidates_num)
        return nums

    def select_candidates_by_grand(self, candidates_num):
        layer_nums = self.get_layer_candidates_num(candidates_num)
        self.candidates.clear()
        for layer in range(1, self.numLayers):
            num = layer_nums[layer]
            print(f'layer {layer}, size {self.layerSizes[layer]}, num {num}')
            # generate a sample
            x = torch.rand(self.num_inputs())
            for i in range(self.num_inputs()):
                x[i] = self.norm_mins[i] + (self.norm_maxes[i] - self.norm_mins[i]) * x[i]
            x.requires_grad_(True)
            # out
            self.check_property(x)
            # sort
            candidates = [(layer, node) for node in range(self.layerSizes[layer])]
            candidates = sorted(candidates, key=lambda pos: self.get_grand(x, pos[0], pos[1])[0])
            self.candidates.extend(candidates[:num])
        print(self.candidates)

    def select_candidates_by_abstraction(self, candidates_num):
        layer_nums = self.get_layer_candidates_num(candidates_num)
        self.candidates.clear()

    def select_candidates_by_impact_score(self, candidates_num):
        candidates = []
        self.init()
        self.deep_poly()
        target_equation = [[1, -1, 0, 0, 0],
                           [1, 0, -1, 0, 0],
                           [1, 0, 0, -1, 0],
                           [1, 0, 0, 0, -1]]
        equation_node = self.form_equation_node(target_equation)
        for layer in [1, 2, 3]:
            for node in range(self.layerSizes[layer]):
                node_info = self.get_node_info(layer, node, NetAnalysis.NodeType.reluBack)
                if node_info.node_status == NetAnalysis.NodeStatus.undefine:
                    score = self.get_impact_score(equation_node, node_info)
                    candidates.append((score, (layer, node)))
        for node in range(self.layerSizes[0]):
            node_info = self.get_node_info(0, node, NetAnalysis.NodeType.input)
            score = self.get_impact_score(equation_node, node_info)
            candidates.append((score, (layer, node)))

        candidates = sorted(candidates, key=lambda x: -x[0])
        for c in candidates:
            print(c)
        return [candidates[i][1] for i in range(candidates_num)]

    def norm_sample(self, sample):
        for i in range(self.num_inputs()):
            if sample[i] > self.maxes[i]:
                sample[i] = self.maxes[i]
            if sample[i] < self.mins[i]:
                sample[i] = self.mins[i]

    # simple method
    def random_sample(self, sample_num):
        samples = []
        for _ in range(sample_num):
            sample = [random.uniform(self.mins[i], self.maxes[i]) for i in range(self.num_inputs())]
            tensor_sample = torch.tensor(sample, dtype=torch.float32, requires_grad=True)
            if self.check_property(tensor_sample):
                samples.append(tensor_sample)
        return samples

    def sample_with_grand(self, sample_num):
        samples = []
        rate = 0.0001
        single_simple_step = 20
        for _ in range(sample_num):
            sample = [random.uniform(self.mins[i], self.maxes[i]) for i in range(self.num_inputs())]
            samples.append(sample)
            for _ in range(single_simple_step):
                tensor_sample = torch.tensor(samples[-1], dtype=torch.float32, requires_grad=True)
                self.check_property(tensor_sample)
                out = self.forward(tensor_sample)
                grand = torch.autograd.grad(inputs=tensor_sample, outputs=out[0])[0]
                sample = [sample[i] - rate * grand[i] for i in range(self.num_inputs())]
                self.norm_sample(sample)
                if torch.norm(torch.tensor(sample, requires_grad=True) - tensor_sample).item() == 0:
                    break
                samples.append(sample)
        return samples

    def sample_with_candidates_and_grand(self, candidates):
        pass

    def calculate_states(self, samples):
        states = []
        for sample in samples:
            state = self.evaluate_state(sample)
            candidates_state = [state[c[0]][c[1]] for c in self.candidates]
            states.append(candidates_state)
        return states

    def save_to_json(self, clauses, json_path="./clauses.json"):
        data = []
        json_content = {'data': data}
        for clause in clauses:
            path = []
            for element in clause:
                (layer, node), state = element
                if state:
                    state = 'Relu active'
                else:
                    state = "Relu inactive"
                path.append({
                    'split': f'({layer}, {node}, {state})',
                    'implied': []
                })
            data.append(path)
        for d in data:
            print("==========" * 10)
            print(d)
        with open(json_path, "w+") as f:
            json.dump(json_content, f)

    def generate_clause(self, candidates_num, sample_num, json_path="./clauses.json"):
        self.init()
        self.deep_poly()
        samples = InputGenerate().inputs_generate(self.norm_mins, self.norm_maxes, sample_num)
        adversarials = []
        for sample in samples:
            is_target = False
            for equation in self.property_equation:
                if equation.type == nnet.EquationType.ge:
                    is_target = True
            adversarials.append(attack_single_sample(self, sample, is_target, True))
        adversarials = sorted(adversarials, key=lambda x: (x[0], x[2]))
        candidates = []
        for i in range(self.layerSizes[1]):
            candidates.append((1, i))
        self.candidates = candidates
        # self.candidates = self.select_candidates_by_impact_score(candidates_num)


        states = self.calculate_states([val[1] for val in adversarials[:30]])
        index = [i for i in range(len(self.candidates))]

        # self.select_candidates_by_grand(candidates_num)
        # samples = self.sample_with_grand(sample_num)
        # states = self.calculate_states(samples)
        # index = [1, 2]
        # print(len(states))
        res = self.find_appear_pattern(index, states)
        print(res)
        self.save_to_json(res, json_path)

        #
        # # grand and candidates
        # for i in range(10):
        #     candidates = random.sample(self.candidates, 2)
        #     self.sample_with_candidates_and_grand(candidates)
        #     break
        # print([val[0] for val in adversarials[:30]])


if __name__ == "__main__":
    net = "/home/center/CDCL-Marabou/sat_example/prop2/ACASXU_experimental_v2a_1_5.nnet"
    prop = "/home/center/CDCL-Marabou/sat_example/prop2/acas_property_2.txt"
    # # clause = ClauseGenerator("/home/center/CDCL-Marabou/sat_example/prop2/ACASXU_experimental_v2a_2_1.nnet")
    clause = ClauseGenerator(net, prop)
    clause.select_candidates_by_impact_score(100)
    #
    # # clause = ClauseGenerator("/home/center/CDCL-Marabou/sat_example/prop2/ACASXU_experimental_v2a_1_2.nnet")
    # clause.generate_clause(20, 1000)
    # # clause.read_property("/home/center/CDCL-Marabou/sat_example/prop2/acas_property_2.txt")
