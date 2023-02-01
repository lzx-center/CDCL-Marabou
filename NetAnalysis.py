import nnet
import json
import numpy as np
import math
import random

class NodeType:
    input = "input"
    output = 'output'
    reluBack = 'relu back'
    reluForward = 'relu forward'
    unknown = 'unknown'
    scalar = 'scalar'

class NodeStatus:
    undefine = "undefine"
    reluActive = 'relu active'
    reluInactive = 'relu inactive '

def dump_bounds(bounds):
    ret = ""
    for info, coeff in bounds.items():
        ret += (f'[{coeff} {info.name}]')
    return ret

class NodeInfo():
    def __init__(self, name, node_type, layer=None, node=None, status=NodeStatus.undefine) -> None:
        self.name = name
        self.node_type = node_type
        self.layer = layer
        self.node = node
        #bound info
        self.lower_bound, self.upper_bound = -np.inf, np.inf
        self.symbolic_upper_bound, self.symbolic_lower_bound = {}, {}
        self.symbolic_upper_bounds, self.symbolic_lower_bounds = {}, {}
        if layer and layer > 0:
            if node_type in {NodeType.reluBack, NodeType.output}:
                self.symbolic_upper_bounds[layer - 1] = self.symbolic_upper_bound
                self.symbolic_lower_bounds[layer - 1] = self.symbolic_lower_bound
            elif node_type == NodeType.reluForward :
                self.symbolic_upper_bounds[layer] = self.symbolic_upper_bound
                self.symbolic_lower_bounds[layer] = self.symbolic_lower_bound
        self.node_status = status
    
    # get bounds
    def get_symbolic_upper_bound(self, layer=0):
        if layer not in self.symbolic_upper_bounds:
            self.propagate_to_layer(layer)
        return self.symbolic_upper_bounds[layer]
        
    def get_symbolic_lower_bound(self, layer=0):
        if layer not in self.symbolic_lower_bounds:
            self.propagate_to_layer(layer)
        return self.symbolic_lower_bounds[layer]

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def propagate_to_layer(self, layer=0):
        # print(self.name)
        if layer > self.layer:
            raise Exception("Can not propagate to layer less than current layer")
        if layer == self.layer and self.node_type == NodeType.reluBack:
            raise Exception("Can not progate to itself")

        repeat_num = max(0, 2 * (self.layer - layer - 1))
        if self.node_type in {NodeType.reluBack, NodeType.reluForward}:
            repeat_num += 1

        sym_lower, sym_upper = self.symbolic_lower_bound, self.symbolic_upper_bound
        for _ in range(repeat_num):
            t_lower, t_upper = {}, {}
            for node_info, coeff in sym_lower.items():
                if coeff < 0:
                    bounds = node_info.symbolic_upper_bound
                else:
                    bounds = node_info.symbolic_lower_bound
                for pre_node_info, pre_coeff in bounds.items():
                    if pre_node_info not in t_lower:
                        t_lower[pre_node_info] = np.float64(0)
                    t_lower[pre_node_info] += coeff * pre_coeff          

            for node_info, coeff in sym_upper.items():
                if coeff < 0:
                    bounds = node_info.symbolic_lower_bound
                else:
                    bounds = node_info.symbolic_upper_bound
                for pre_node_info, pre_coeff in bounds.items():
                    if pre_node_info not in t_upper:
                        t_upper[pre_node_info] = np.float64(0)
                    t_upper[pre_node_info] += coeff * pre_coeff
            sym_lower = t_lower
            sym_upper = t_upper

        self.symbolic_lower_bounds[layer] = sym_lower
        self.symbolic_upper_bounds[layer] = sym_upper

    # to string
    def __str__(self) -> str:
        ret = f"Node name: {self.name}\nNode type: {self.node_type}\nLower bound: {self.lower_bound}\n" \
                f"Upper bound: {self.upper_bound}\n"
        s_lower, s_upper = self.symbolic_lower_bound, self.symbolic_upper_bound
        ret += "Symbolic lower: "
        for node, coeff in s_lower.items():
            ret += f"{coeff} {node.name} "
        ret += "\nSymbolic upper: "
        for node, coeff in s_upper.items():
            ret += f"{coeff} {node.name} "
        ret += f"\nNode status: {self.node_status}"
        return ret

    def dump_symbolic_bound(self, bound):
        ret = ""
        for node, coeff in bound.items():
            ret += f"{coeff} {node.name} "
        return ret

    def __update_status(self):
        if self.lower_bound > self.upper_bound:
            print(self)
            low, up = self.get_symbolic_lower_bound(), self.get_symbolic_upper_bound()
            print(f"====={self.name}====\nlower: {dump_bounds(low)}\nupper: {dump_bounds(up)}")
            print(self.lower_bound, self.upper_bound)
            raise Exception("Lower bound largerer than upper bound")
        if self.lower_bound >= 0:
            self.node_status = NodeStatus.reluActive
        elif self.upper_bound <= 0:
            self.node_status = NodeStatus.reluInactive

    def clac_bounds_via_layer(self, layer=0):
        sym_lower = self.get_symbolic_lower_bound(layer)
        sym_upper = self.get_symbolic_upper_bound(layer)
        self.lower_bound, self.upper_bound = np.float64(0), np.float64(0)
        if sym_lower is not None:
            for pre_node, coeff in sym_lower.items():
                if coeff < 0:
                    self.lower_bound += pre_node.upper_bound * coeff
                    continue
                self.lower_bound += pre_node.lower_bound * coeff
        if sym_upper is not None:
            for pre_node, coeff in sym_upper.items():
                if coeff < 0:
                    self.upper_bound += pre_node.lower_bound * coeff
                    continue
                self.upper_bound += pre_node.upper_bound * coeff
                
        self.__update_status()

class DeepPolyAnalysis(nnet.NNet):
    def __init__(self, filename):
        super().__init__(filename)
        self.node_names = {}
        self.node_infos = {}
        self.__init_node_info()

    def get_node_name(self, layer, node, type):
        query = (layer, node, type)
        if query in self.node_names:
            return self.node_names[query]
        name = f'({layer},{node},{type})'
        self.node_names[query] = name
        return name

    def get_node_info_by_name(self, name):
        if name in self.node_infos:
            return self.node_infos[name]
        raise Exception("Invalid name!")

    def __init_node_info(self):
        # scalar
        scalar_name = 'scalar'
        sc = NodeInfo(scalar_name, node_type=NodeType.scalar, layer=0, node=0)
        sc.lower_bound = np.float64(1)
        sc.upper_bound = np.float64(1)
        sc.symbolic_lower_bound[sc] = np.float64(1)
        sc.symbolic_upper_bound[sc] = np.float64(1)
        sc.symbolic_lower_bounds = {layer: sc.symbolic_lower_bound for layer in range(self.numLayers)}
        sc.symbolic_upper_bounds = {layer: sc.symbolic_upper_bound for layer in range(self.numLayers)}
        self.node_infos[scalar_name] = sc

        # input
        for node in range(self.layerSizes[0]):
            node_name = self.get_node_name(0, node, NodeType.input)
            info = NodeInfo(node_name, NodeType.input, 0, node)
            info.lower_bound = np.float64(self.mins[node])
            info.upper_bound = np.float64(self.maxes[node])
            # inputs' lower and upper bounds is their self
            info.symbolic_lower_bound[info] = np.float64(1)
            info.symbolic_upper_bound[info] = np.float64(1)
            self.node_infos[node_name] = info
     
        # relu
        for layer in range(1, self.numLayers):
            for node in range(self.layerSizes[layer]):
                # process back and pre forward
                weight = self.weights[layer - 1][node]
                node_name_back = self.get_node_name(layer, node, NodeType.reluBack)
                back_info = NodeInfo(node_name_back, node_type=NodeType.reluBack, layer=layer, node=node)
                for pre_node in range(self.layerSizes[layer - 1]):
                    w = weight[pre_node]
                    if w == 0:
                        continue
                    if layer == 1:
                        pre_node_name = self.get_node_name(layer - 1, pre_node, NodeType.input)
                    else:
                        pre_node_name = self.get_node_name(layer - 1, pre_node, NodeType.reluForward)
                    pre_node_info = self.get_node_info_by_name(pre_node_name)
                    back_info.symbolic_lower_bound[pre_node_info] = w
                    back_info.symbolic_upper_bound[pre_node_info] = w
                back_info.symbolic_lower_bound[sc] = self.biases[layer - 1][node]
                back_info.symbolic_upper_bound[sc] = self.biases[layer - 1][node]
                back_info.clac_bounds_via_layer()
                self.node_infos[node_name_back] = back_info
                # process back and forward
                node_name_forward = self.get_node_name(layer, node, NodeType.reluForward)
                forward_info = NodeInfo(node_name_forward, node_type=NodeType.reluForward, layer=layer, node=node)
                if back_info.lower_bound > 0:
                    forward_info.symbolic_lower_bound[back_info] = np.float64(1)
                    forward_info.symbolic_upper_bound[back_info] = np.float64(1)
                    forward_info.node_status = NodeStatus.reluActive
                elif back_info.upper_bound <= 0:
                    forward_info.node_status = NodeStatus.reluInactive
                else:
                    l, u = back_info.lower_bound, back_info.upper_bound
                    a = u / (u - l)
                    forward_info.symbolic_upper_bound[back_info] = np.float64(a)
                    forward_info.symbolic_upper_bound[sc] = np.float64(-l) * np.float64(a)

                forward_info.clac_bounds_via_layer()
                self.node_infos[node_name_forward] = forward_info
                # print(back_info)
                # print(forward_info)
        # output
        layer = self.numLayers
        for node in range(self.layerSizes[layer]):
        # process back and pre forward
            weight = self.weights[layer - 1][node]
            node_name_back = self.get_node_name(layer, node, NodeType.output)
            back_info = NodeInfo(node_name_back, NodeType.output, layer=layer, node=node)
            for pre_node in range(self.layerSizes[layer - 1]):
                w = weight[pre_node]
                if w == 0:
                    continue
                if layer == 1:
                    pre_node_name = self.get_node_name(layer - 1, pre_node, NodeType.input)
                else:
                    pre_node_name = self.get_node_name(layer - 1, pre_node, NodeType.reluForward)
                pre_node_info = self.get_node_info_by_name(pre_node_name)
                back_info.symbolic_lower_bound[pre_node_info] = w
                back_info.symbolic_upper_bound[pre_node_info] = w
            back_info.symbolic_lower_bound[sc] = self.biases[layer - 1][node]
            back_info.symbolic_upper_bound[sc] = self.biases[layer - 1][node]
            back_info.clac_bounds_via_layer()
            self.node_infos[node_name_back] = back_info
    
    def _generate_split_data(self, node_info_set):
        data = []
        for node_info in node_info_set:
            data.append(
                {
                    "split" : node_info,
                    "implied" : []
                }
            )
        return data

    def __dump_splits(self, split_data, file_path):
        with open(file_path, "w+") as f:
            data = []
            for node_info_set in split_data:
                data.append(self._generate_split_data(node_info_set))
            text = json.dumps({'data' : data})
            f.write(text)

    def clause_predict(self, candidate_num, sample_num = 100000, generated_num = 50):
        layer = self.numLayers

        layer_candidates_num = [ 0 for _ in range(self.numLayers)]
        n = self.numLayers - 1
        start_num = math.floor(candidate_num / n) - n + 1
        for i in range(1, self.numLayers):
            layer_candidates_num[i] = start_num + 2 * (self.numLayers - i - 1)
        layer_candidates_num[1] += candidate_num - sum(layer_candidates_num)
        print("select num", layer_candidates_num)

        # ndoe selection
        candidate = []

        for layer in range(self.numLayers):
            if layer and layer_candidates_num[layer]:
                node_num = self.layerSizes[layer]
                nodes = []
                for node in range(node_num):
                    for type in {NodeType.reluBack}:
                        node_name = self.get_node_name(layer, node, type)
                        node_info = self.get_node_info_by_name(node_name)
                        if node_info.node_status == NodeStatus.undefine:
                            nodes.append(node_info)
                nodes = sorted(nodes, key=lambda x: abs(x.lower_bound - x.upper_bound) / min(abs(x.lower_bound), abs(x.upper_bound)))
                candidate.extend(nodes[:node_num])
        
        print("Candidates:")
        for node in candidate:
            print(node.name, end=", ")
        print("")
        # sample
        states = []
        def trivial_sample():
            for i in range(sample_num):
                input = [random.uniform(self.mins[i], self.maxes[i]) for i in range(self.num_inputs())]
                # print(input)
                state = self.evaluate_state(input)
                states.append([state[node.layer][node.node] for node in candidate])
            print(f"{len(self.calc_states)} test Genrate done")
        
        def interval_sample():
            total = 0
            for i in range(self.num_inputs()):
                total += self.maxes[i] - self.mins[i]
            interval = [(self.maxes[i] - self.mins[i]) / total for i in range(self.num_inputs())]
            interval = [int(i * sample_num) for i in interval]
            print(interval)
            index_set = set()
            for _ in range(sample_num):
                select = (random.randint(0, interval[i]) for i in range(self.num_inputs()))
                while select in index_set:
                    select = (random.randint(0, interval[i]) for i in range(self.num_inputs()))
                index_set.add(select)
                lists = [*select]
                input = [self.mins[i] + (self.maxes[i] - self.mins[i]) / interval[i] * lists[i] for i in range(self.num_inputs())]
                state = self.evaluate_state(input)
                states.append([state[node.layer][node.node] for node in candidate])
        
        interval_sample()
        # clause generate
        clauses = []
        clause_length = 5
        indexes = [_ for _ in range(len(candidate))]
        while generated_num != 0:
            index = random.sample(indexes, clause_length)
            index = sorted(index)

            index_states = [[state[i] for i in index] for state in states]
            state_sets = set()
            for state in index_states:
                state_number = 0
                for i in range(len(state)):
                    if state[i]:
                        state_number = state_number | (1 << i)
                state_sets.add(state_number)
            
            def count_one(v):
                num = 0
                while v:
                    v &= v - 1
                    num += 1
                return num  
            
            min_diff = clause_length
            val = -1
            for i in range(1 << clause_length):
                if i not in state_sets:
                    count = abs(count_one(i) - int((clause_length + 1) / 2))
                    if min_diff > count:
                        min_diff = count
                        val = i
            if val != -1:
                clause = []
                for i in range(clause_length):
                    info = candidate[index[i]]
                    if val & (1 << i):
                        clause.append(f"({info.layer}, {info.node}, Relu active)")
                    else:
                        clause.append(f"({info.layer}, {info.node}, Relu inactive)")
                clauses.append(clause)
            generated_num -= 1

        self.__dump_splits(clauses, "./g.json")

if __name__ == "__main__":
    analysis = DeepPolyAnalysis("/home/center/CDCL-Marabou/example/reluBenchmark3.11155605316s_UNSAT.nnet")
    print("======="*20)
    analysis.clause_predict(30, sample_num=10000)