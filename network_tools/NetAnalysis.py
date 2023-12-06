from itertools import combinations

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
        ret += f'[{coeff} {info.name}]'
    return ret


class NodeInfo:
    def __init__(self, name, node_type, layer=None, node=None, status=NodeStatus.undefine) -> None:
        self.name = name
        self.node_type = node_type
        self.layer = layer
        self.node = node
        # bound info
        self.lower_bound, self.upper_bound = -np.inf, np.inf
        if node_type is NodeType.reluForward:
            self.lower_bound = 0
        self.symbolic_upper_bound, self.symbolic_lower_bound = {}, {}
        self.symbolic_upper_bounds, self.symbolic_lower_bounds = {}, {}
        if layer and layer > 0:
            if node_type in {NodeType.reluBack, NodeType.output}:
                self.symbolic_upper_bounds[layer - 1] = self.symbolic_upper_bound
                self.symbolic_lower_bounds[layer - 1] = self.symbolic_lower_bound
            elif node_type == NodeType.reluForward:
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

    def propagate_to_layer(self, layer=0, node_type=NodeType.input):
        if layer and layer > self.layer:
            raise Exception("Can not propagate to layer less than current layer")
        if layer and layer == self.layer and self.node_type == NodeType.reluBack:
            raise Exception("Can not progate to itself")

        sym_lower, sym_upper = self.symbolic_lower_bound, self.symbolic_upper_bound
        print("*********"*10)
        print("Before Calculating Node name: ", self.name)
        print("Dump Symbolic lower: ", self.dump_symbolic_bound(sym_lower))
        print("Dump Symbolic upper: ", self.dump_symbolic_bound(sym_upper))
        print("*********"*10)
        while True:
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
            print("*********"*10)
            print("Calculating Node name: ", self.name)
            print("Dump Symbolic lower: ", self.dump_symbolic_bound(sym_lower))
            print("Dump Symbolic upper: ", self.dump_symbolic_bound(sym_upper))
            print("*********"*10)
            self.calc_bounds_via_symbolic(t_lower, t_upper)

            def end_process(bound):
                for node_info, coeff in bound.items():
                    if node_info.node_type == node_type and node_info.layer == layer:
                        return True
                if len(bound) == 1:
                    for node_info, coeff in bound.items():
                        if node_info.node_type != NodeType.scalar:
                            return False
                    return True
                elif len(bound) == 0:
                    return True
                return False

            # end propagate judgement
            if end_process(sym_lower) and end_process(sym_upper):
                break

        return sym_lower, sym_upper

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
        elif self.upper_bound <= 0:
            self.node_status = NodeStatus.reluInactive
        if self.lower_bound > 0:
            self.node_status = NodeStatus.reluActive

    # def calc_bounds_via_layer(self, sym_lower, sym_up):
    def calc_bounds_via_symbolic(self, sym_lower, sym_upper):
        lower_bound, upper_bound = np.float64(0), np.float64(0)
        if sym_lower is not None:
            for pre_node, coeff in sym_lower.items():
                if coeff < 0:
                    lower_bound += pre_node.upper_bound * coeff
                    continue
                lower_bound += pre_node.lower_bound * coeff
        if sym_upper is not None:
            for pre_node, coeff in sym_upper.items():
                if coeff < 0:
                    upper_bound += pre_node.lower_bound * coeff
                    continue
                upper_bound += pre_node.upper_bound * coeff
        self.lower_bound = max(self.lower_bound, lower_bound)
        self.upper_bound = min(self.upper_bound, upper_bound)

    def clac_bounds_via_layer(self, layer=0):
        sym_lower, sym_upper = self.propagate_to_layer()
        self.calc_bounds_via_symbolic(sym_lower, sym_upper)
        self.__update_status()


class DeepPolyAnalysis(nnet.NNet):
    def __init__(self, filename):
        super().__init__(filename)
        self.node_names = {}
        self.node_infos = {}
        self.init()

    def init(self):
        self.node_names.clear()
        self.node_infos.clear()
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

    def get_node_name(self, layer, node, type):
        query = (layer, node, type)
        if query in self.node_names:
            return self.node_names[query]
        name = f'({layer},{node},{type})'
        self.node_names[query] = name
        return name

    def get_node_info(self, layer, node, type) -> NodeInfo:
        return self.get_node_info_by_name(self.get_node_name(layer, node, type))

    def get_node_info_by_name(self, name):
        if name in self.node_infos:
            return self.node_infos[name]
        raise Exception(f"Invalid name! {name}")

    def deep_poly(self):
        sc = self.node_infos["scalar"]
        # input
        for node in range(self.layerSizes[0]):
            node_name = self.get_node_name(0, node, NodeType.input)
            info = NodeInfo(node_name, NodeType.input, 0, node)
            info.lower_bound = np.float64(self.norm_mins[node])
            info.upper_bound = np.float64(self.norm_maxes[node])
            # inputs' lower and upper bounds is their self
            info.symbolic_lower_bound[info] = np.float64(1)
            info.symbolic_upper_bound[info] = np.float64(1)
            self.node_infos[node_name] = info

        # relu
        for layer in range(1, self.numLayers):
            # print("----" * 20)
            # print("Analysis layer", layer, "size:", self.layerSizes[layer])
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
                    if u > -l:
                        forward_info.symbolic_lower_bound[back_info] = np.float64(1)

                forward_info.clac_bounds_via_layer()
                self.node_infos[node_name_forward] = forward_info
                print("----" * 10)
                print(back_info)
                print("----" * 10)
                print(forward_info)
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
            print("----" * 10)
            print(back_info)

    def _generate_split_data(self, node_info_set):
        data = []
        for node_info in node_info_set:
            data.append(
                {
                    "split": node_info,
                    "implied": []
                }
            )
        return data

    def __dump_splits(self, split_data, file_path):
        with open(file_path, "w+") as f:
            data = []
            for node_info_set in split_data:
                data.append(self._generate_split_data(node_info_set))
            text = json.dumps({'data': data})
            f.write(text)

    def clause_predict(self, candidate_num, sample_num=100000, generated_num=50):
        layer = self.numLayers

        layer_candidates_num = [0 for _ in range(self.numLayers)]
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
                nodes = sorted(nodes, key=lambda x: abs(x.lower_bound - x.upper_bound) / min(abs(x.lower_bound),
                                                                                             abs(x.upper_bound)))
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
                input = [self.mins[i] + (self.maxes[i] - self.mins[i]) / interval[i] * lists[i] for i in
                         range(self.num_inputs())]
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

    def dependency_analysis(self):
        intra_dependencies = self.intra_depends_analysis()
        inter_dependencies = self.inter_depends_analysis()
        print("Intra dependencies:")
        for intra in intra_dependencies:
            print(intra)
        print("Inter dependencies:")
        for inter in inter_dependencies:
            print(inter)

        return intra_dependencies
        

    def inter_depends_analysis(self):
        ret = []
        for layer in range(2, self.numLayers - 1):
            for index in range(self.layerSizes[layer]):
                node = self.get_node_info_by_name(self.get_node_name(layer, index, NodeType.reluBack))
                sym_lower = node.symbolic_upper_bound
                for info, coe in sym_lower.items():
                    if info.node_status is not NodeStatus.undefine:
                        continue
                    if info.node_type is NodeType.scalar:
                        continue
                    if node.lower_bound - info.upper_bound * coe >= 0:
                        ret.append(((info.name, "inactive"), (node.name, "active")))
                    if node.upper_bound - info.upper_bound * coe <= 0:
                        ret.append(((info.name, "inactive"), (node.name, "inactive")))
        return ret

    def intra_depends_analysis(self):
        res = []
        for pre_layer in range(self.numLayers - 1):
            cur_layer = pre_layer + 1
            print(f"layer {pre_layer}: {self.layerSizes[pre_layer]}, layer {cur_layer}: {self.layerSizes[cur_layer]}")
            for (index1, index2) in list(combinations(range(self.layerSizes[cur_layer]), 2)):
                node_name1 = self.get_node_name(cur_layer, index1, NodeType.reluBack)
                node_name2 = self.get_node_name(cur_layer, index2, NodeType.reluBack)

                # get pre bound
                lower_bounds, upper_bounds = [], []
                for index in range(self.layerSizes[pre_layer]):
                    n = self.get_node_info_by_name(self.get_node_name(pre_layer, index, NodeType.reluForward
                    if pre_layer else NodeType.input))
                    lower_bounds.append(n.lower_bound)
                    upper_bounds.append(n.upper_bound)
                bounds = (np.array(lower_bounds), np.array(upper_bounds))

                n1, n2 = self.get_node_info_by_name(node_name1), self.get_node_info_by_name(node_name2)
                min0, max0 = self.calc_intra_neural_depends(n1, n2, bounds)
                min1, max1 = self.calc_intra_neural_depends(n2, n1, bounds)
                if min0 is None or min1 is None:
                    continue
                result = None
                if max0 < 0 and max1 < 0:
                    result = "active", "inactive"

                elif min0 > 0 and min1 > 0:
                    result = "inactive", "active"

                elif max0 < 0 < min1:
                    result = "active", "active"

                elif min0 > 0 > max1:
                    result = "inactive", "inactive"
                if result:
                    if n1.node_status == NodeStatus.undefine and n2.node_status == NodeStatus.undefine:
                        res.append(((node_name1, result[0]), (node_name2, result[1])))
                        # print(f"{n1.name} {result[0]} ==> {n2.name} {result[1]}")
                        # print(f"{n1.name} [{min0}, {max0}]")
                        # print(f"{n2.name} [{min1}, {max1}]")
        return res

    def calc_intra_neural_depends(self, neural1, neural2, bounds):
        lower_bounds, upper_bounds = bounds

        w1 = self.weights[neural1.layer - 1][neural1.node, :]
        w2 = self.weights[neural1.layer - 1][neural2.node, :]

        b1 = self.biases[neural1.layer - 1][neural1.node]
        b2 = self.biases[neural2.layer - 1][neural2.node]

        nonzero_index = 0
        while w1[nonzero_index] == 0 or w2[nonzero_index] == 0:
            nonzero_index += 1
            if nonzero_index == len(w1):
                return None, None

        wp = w1 - (w1[nonzero_index] / w2[nonzero_index]) * w2
        bp = b1 - (w1[nonzero_index] / w2[nonzero_index]) * b2
        weights_plus = np.clip(wp, 0, np.inf)
        weights_minus = np.clip(wp, -np.inf, 0)

        _min = _max = 0
        _min += weights_plus.dot(lower_bounds) + weights_minus.dot(upper_bounds) + bp
        _max += weights_plus.dot(upper_bounds) + weights_minus.dot(lower_bounds) + bp
        # print("=====" * 10)
        # print("weight plus: ", weights_plus)
        # print("weight minus: ", weights_minus)
        # print("lower:", lower_bounds)
        # print("wp:", wp)
        # print('upper:', upper_bounds)
        # print("bp: ", bp, "b1: ", b1, "b2: ", b2)
        # print(_min, _max)
        return _min, _max

    def get_certain_node_num(self):
        counter = 0
        for layer in range(1, self.numLayers):
            for index in range(self.layerSizes[layer]):
                node = self.get_node_info(layer, index, NodeType.reluForward)
                if node.node_status is not NodeStatus.undefine:
                    counter += 1
        return counter

    def get_total_node_num(self):
        return sum(self.layerSizes)

    def form_equation_node(self, out_puts):
        """

        Args:
            out_puts: a list of size: out_put_num * n

        Returns:
            a node with upper bound and lower bound as target equation
        """
        node_name = self.get_node_name(self.numLayers + 1, 0, NodeType.reluBack)
        node_info = NodeInfo(node_name, node_type=NodeType.reluBack, layer=self.numLayers + 1, node=0)
        for equation in out_puts:
            for i in range(len(equation)):
                out_info = self.get_node_info(self.numLayers, i, NodeType.output)
                if node_info in node_info.symbolic_lower_bound:
                    node_info.symbolic_lower_bound[out_info] += equation[i]
                    node_info.symbolic_upper_bound[out_info] += equation[i]
                    continue
                node_info.symbolic_lower_bound[out_info] = equation[i]
                node_info.symbolic_upper_bound[out_info] = equation[i]
        return node_info

    def get_impact_score(self, equation_node: NodeInfo, target_node: NodeInfo):
        sym_low, sym_up = equation_node.propagate_to_layer(target_node.layer, target_node.node_type)
        if target_node not in sym_low:
            sym_low = {target_node: 0}
        if target_node not in sym_up:
            sym_up = {target_node: 0}

        coe = sym_up[target_node] - sym_low[target_node]
        tmp_node = NodeInfo("tmp", NodeType.reluBack)
        tmp_node.symbolic_upper_bound[target_node] = coe
        tmp_node.symbolic_lower_bound[target_node] = coe
        tmp_node.clac_bounds_via_layer()
        score = tmp_node.upper_bound - tmp_node.lower_bound
        return score


def recursive_split(analysis, input_lower, input_upper, certain_node, domains=[]):
    analysis.norm_mins = input_lower
    analysis.norm_maxes = input_upper
    analysis.init()
    analysis.deep_poly()
    certain = analysis.get_certain_node_num()
    total = analysis.get_total_node_num()
    ratio = (certain - certain_node) / total
    print(f"certain node: {certain}")
    if ratio < 0.05:
        domains.append(([val for val in input_lower], [val for val in input_upper]))
        return

    max_interval, index = input_upper[0] - input_lower[0], 0
    for i in range(1, len(input_lower)):
        interval = input_upper[i] - input_lower[i]
        if max_interval < interval:
            max_interval = interval
            index = i
    lower = [val for val in input_lower]
    upper = [val for val in input_upper]

    mid = (lower[index] + upper[index]) / 2
    upper[index] = mid
    recursive_split(analysis, lower, upper, certain, domains)
    upper[index] = input_upper[index]
    lower[index] = mid
    recursive_split(analysis, lower, upper, certain, domains)


def split_input(net_path, property_path):
    analysis = DeepPolyAnalysis(net_path)
    analysis.load_property(property_path)
    input_lower, input_upper = analysis.norm_mins, analysis.norm_maxes
    domains = []
    recursive_split(analysis, input_lower, input_upper, 0, domains)
    for lower, upper in domains:
        analysis.norm_mins = lower
        analysis.norm_maxes = upper
        analysis.init()
        analysis.deep_poly()
        print(lower, upper)
        analysis.dependency_analysis()


def deep_poly(net_path, property_path):
    analysis = DeepPolyAnalysis(net_path)
    analysis.load_property(property_path)
    analysis.init()
    analysis.deep_poly()

    # target_equation = [[1, -1, 0, 0, 0],
    #                    [1, 0, -1, 0, 0],
    #                    [1, 0, 0, -1, 0],
    #                    [1, 0, 0, 0, -1]]
    #
    # equation_node = analysis.form_equation_node(target_equation)
    # for i in range(analysis.layerSizes[2]):
    #     target_node = analysis.get_node_info(2, i, NodeType.reluBack)
    #     if target_node.node_status == NodeStatus.undefine:
    #         score = analysis.get_impact_score(equation_node, target_node)
    #         print(target_node.name, score, target_node.node_status)

    intra = analysis.dependency_analysis()
    if len(intra) > 0:
        return True
    return False

if __name__ == "__main__":
    net = "/home/center/CDCL-Marabou/example/test.nnet"
    prop = "/home/center/CDCL-Marabou/example/test.txt"
    deep_poly(net, prop)
