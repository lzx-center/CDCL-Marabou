import numpy as np
import json
# from graphviz import Digraph, Graph

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import torch.nn as nn
import torch


class EquationType:
    ge = ">="
    le = "<="


class Equation:

    def __init__(self, type) -> None:
        self.type = type
        self.lhs = {}
        self.scalar = None

    def __str__(self) -> str:
        s = ""
        for index, coeff in self.lhs.items():
            s += f"({coeff} y{index})+"
        s = s.strip("+")
        s += self.type
        s += f" {self.scalar}"
        return s


class NNet(nn.Module):
    """
    Class that represents a fully connected ReLU network from a .nnet file
    
    Args:
        filename (str): A .nnet file to load
    
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        weights (list of numpy arrays): Weight matrices in network
        biases (list of numpy arrays): Bias vectors in network
    """

    def __init__(self, filename):
        super(NNet, self).__init__()
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line = f.readline()
                cnt += 1
            # numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()

            # input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            weights = []
            biases = []
            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum + 1]
                weights.append([])
                biases.append([])
                weights[layernum] = np.zeros((currentLayerSize, previousLayerSize))
                for i in range(currentLayerSize):
                    line = f.readline()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    for j in range(previousLayerSize):
                        weights[layernum][i, j] = aux[j]
                # biases
                biases[layernum] = np.zeros(currentLayerSize)
                for i in range(currentLayerSize):
                    line = f.readline()
                    x = float(line.strip().split(",")[0])
                    biases[layernum][i] = x

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges
            self.weights = weights
            self.biases = biases

            self.splits_set = None
            self.calc_states = []
            self.eliminated_constraint = []
            self.__init_tensor_param()
            self.norm_mins = [(self.mins[i] - self.means[i]) / self.ranges[i] for i in range(self.num_inputs())]
            self.norm_maxes = [(self.maxes[i] - self.means[i]) / self.ranges[i] for i in range(self.num_inputs())]
            self.property_equation = []

    def __init_tensor_param(self):
        self.torch_layers = [None]
        self.calc_info = [None]
        for i in range(self.numLayers):
            current_size = self.layerSizes[i + 1]
            pre_size = self.layerSizes[i]
            linear = nn.Linear(pre_size, current_size)
            linear.weight = nn.Parameter(torch.tensor(self.weights[i], requires_grad=True).to(torch.float32))
            linear.bias = nn.Parameter(torch.tensor(self.biases[i]).to(torch.float32))
            self.torch_layers.append(linear)
            self.calc_info.append(None)

    def satisfy_property(self, input_vec):
        out = self.evaluate_network(input_vec)
        sat = True
        for equation in self.property_equation:
            res = 0
            for index, coe in equation.lhs.items():
                res += out[index] * coe
            if equation.type == EquationType.le:
                if res > equation.scalar:
                    sat = False
                    break
            else:
                if res < equation.scalar:
                    sat = False
                    break
        return sat

    def forward(self, input):
        self.calc_info[0] = input
        for i in range(1, self.numLayers + 1):
            if i == self.numLayers:
                self.calc_info[i] = self.torch_layers[i](self.calc_info[i - 1])
            else:
                self.calc_info[i] = nn.functional.relu(self.torch_layers[i](self.calc_info[i - 1]))
        out = self.calc_info[self.numLayers]
        return out

    def get_grand(self, input, layer, node):
        self.forward(input)
        out = self.calc_info[layer][node]
        grand = torch.autograd.grad(inputs=input, outputs=out)[0]
        # print(f"Out: {out}, Grand {grand}")
        return grand

    def get_fn_grand(self, input, function):
        return torch.autograd.grad(inputs=input, outputs=function)[0]

    def evaluate_network(self, inputs, normalize_input=False, undo_normalize_output=False):
        '''
        Evaluate network using given inputs
        
        Args:
            inputs (numpy array of floats): Network inputs to be evaluated
            
        Returns:
            (numpy array of floats): Network output
        '''
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights

        # Prepare the inputs to the neural network
        inputsNorm = np.zeros(inputSize)
        for i in range(inputSize):
            inputsNorm[i] = inputs[i]
            if normalize_input:
                if inputs[i] < self.mins[i]:
                    inputsNorm[i] = (self.mins[i] - self.means[i]) / self.ranges[i]
                elif inputs[i] > self.maxes[i]:
                    inputsNorm[i] = (self.maxes[i] - self.means[i]) / self.ranges[i]
                else:
                    inputsNorm[i] = (inputs[i] - self.means[i]) / self.ranges[i]

                    # Evaluate the neural network
        for layer in range(numLayers - 1):
            inputsNorm = np.maximum(np.dot(weights[layer], inputsNorm) + biases[layer], 0)
        outputs = np.dot(weights[-1], inputsNorm) + biases[-1]

        # Undo output normalization
        if undo_normalize_output:
            for i in range(outputSize):
                outputs[i] = outputs[i] * self.ranges[-1] + self.means[-1]
        return outputs

    def evaluate_network_multiple(self, inputs):
        '''
        Evaluate network using multiple sets of inputs
        
        Args:
            inputs (numpy array of floats): Array of network inputs to be evaluated.
            
        Returns:
            (numpy array of floats): Network outputs for each set of inputs
        '''

        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights
        inputs = np.array(inputs).T

        # Prepare the inputs to the neural network
        numInputs = inputs.shape[1]
        inputsNorm = np.zeros((inputSize, numInputs))
        for i in range(inputSize):
            for j in range(numInputs):
                if inputs[i, j] < self.mins[i]:
                    inputsNorm[i, j] = (self.mins[i] - self.means[i]) / self.ranges[i]
                elif inputs[i, j] > self.maxes[i]:
                    inputsNorm[i, j] = (self.maxes[i] - self.means[i]) / self.ranges[i]
                else:
                    inputsNorm[i, j] = (inputs[i, j] - self.means[i]) / self.ranges[i]

        # Evaluate the neural network
        for layer in range(numLayers - 1):
            inputsNorm = np.maximum(np.dot(weights[layer], inputsNorm) + biases[layer].reshape((len(biases[layer]), 1)),
                                    0)
        outputs = np.dot(weights[-1], inputsNorm) + biases[-1].reshape((len(biases[-1]), 1))

        # Undo output normalization
        for i in range(outputSize):
            for j in range(numInputs):
                outputs[i, j] = outputs[i, j] * self.ranges[-1] + self.means[-1]
        return outputs.T

    def num_inputs(self):
        ''' Get network input size'''
        return self.inputSize

    def num_outputs(self):
        ''' Get network output size'''
        return self.outputSize

    def get_node_relation(self, layer, node):
        return [self.weights[layer - 1][node, i] for i in range(self.layerSizes[layer - 1])]

    def load_json(self, json_path):
        f = open(json_path)
        json_dict = json.load(f)
        json_data = json_dict['data']

        def load_triple(triple, mark, order=None):
            triple = triple.strip('(').strip(')').split(',')
            layer, node, type = int(triple[0].strip(' ')), int(triple[1].strip(' ')), triple[2].strip(" ")
            return layer, node, type, mark, order

        splits_set = []
        for path in json_data:
            splits = []
            num = 1
            for element in path:
                splits.append(load_triple(element['split'], "assertion", num))
                for imply in element['implied']:
                    splits.append(load_triple(imply, "implied", num))
                num += 1
            splits_set.append(splits)

            # parse eliminated
        eliminated = json_dict['eliminated']
        for constraint in eliminated:
            self.eliminated_constraint.append(load_triple(constraint['split'], "eliminated", 0))

        self.splits_set = splits_set

    def load_property(self, property_path):
        with open(property_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n")
                if line.startswith('x'):
                    var, op, bound = line.split(" ")
                    index, bound = int(var[1:]), float(bound)
                    if op == "<=":
                        self.norm_maxes[index] = min(self.norm_maxes[index], bound)
                    elif op == ">=":
                        self.norm_mins[index] = max(self.norm_mins[index], bound)
                elif line.startswith(('+', '-', 'y')):
                    if "<=" in line:
                        lhs, scalar = line.split("<=")
                        equation = Equation(EquationType.le)
                    elif ">=" in line:
                        lhs, scalar = line.split(">=")
                        equation = Equation(EquationType.ge)
                    equation.scalar = float(scalar)
                    lhs = lhs.strip(" ").split(" ")
                    for lh in lhs:
                        op = "+"
                        if lh.startswith("-"):
                            op = "-"
                            lh = lh[1:]
                        else:
                            if lh.startswith('+'):
                                lh = lh[1:]
                        index = int(lh[1:])
                        equation.lhs[index] = 1 if op == "+" else -1
                    self.property_equation.append(equation)

    def dump_property(self, prop_path, lower=None, upper=None):
        if lower is None:
            lower = self.norm_mins
        if upper is None:
            upper = self.norm_maxes
        with open(prop_path, "w+") as f:
            for i in range(len(lower)):
                print(f"x{i} >= {lower[i]}", file=f)
                print(f"x{i} <= {upper[i]}", file=f)
            for equation in self.property_equation:
                s = ''
                if len(equation.lhs) == 1:
                    for index, coeff in equation.lhs.items():
                        if coeff > 0:
                            s += f"y{index} "
                        else:
                            s += f'-y{index} '
                else:
                    for index, coeff in equation.lhs.items():
                        if coeff > 0:
                            s += f"+y{index} "
                        else:
                            s += f'-y{index} '
                s += equation.type
                s += ' '
                s += str(equation.scalar)
                print(s, file=f)

    def evaluate_state(self, inputs, normalize_input=False, undo_normalize_output=False):
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights

        # Prepare the inputs to the neural network
        inputsNorm = np.zeros(inputSize)
        for i in range(inputSize):
            inputsNorm[i] = inputs[i]
            if normalize_input:
                if inputs[i] < self.mins[i]:
                    inputsNorm[i] = (self.mins[i] - self.means[i]) / self.ranges[i]
                elif inputs[i] > self.maxes[i]:
                    inputsNorm[i] = (self.maxes[i] - self.means[i]) / self.ranges[i]
                else:
                    inputsNorm[i] = (inputs[i] - self.means[i]) / self.ranges[i]

                    # Evaluate the neural network
        state = [[]]
        for layer in range(numLayers - 1):
            inputsNorm = np.maximum(np.dot(weights[layer], inputsNorm) + biases[layer], 0)
            state.append((inputsNorm > 0).tolist())
        outputs = np.dot(weights[-1], inputsNorm) + biases[-1]
        # Undo output normalization
        if undo_normalize_output:
            for i in range(outputSize):
                outputs[i] = outputs[i] * self.ranges[-1] + self.means[-1]
        return state

    def visual(self, pic_path="visualize/test.visul.jpg"):
        # input
        dot = Digraph(comment="nnet")
        dot.graph_attr['rankdir'] = 'LR'
        dot.graph_attr['splines'] = 'line'
        dot.graph_attr['ranksep'] = '5'

        with dot.subgraph(name='input') as input:
            for i in range(self.num_inputs()):
                input.node(f'({0},{i})', shape='circle')

        for layer in range(self.numLayers - 1):
            with dot.subgraph(name=f'layer{layer + 1}') as layer_graph:
                for node in range(self.layerSizes[layer + 1]):
                    node_name = f'({layer + 1},{node})'
                    layer_graph.node(node_name, shape='circle')
                    weight = self.weights[layer][node]
                    for pre_node in range(self.layerSizes[layer]):
                        w = weight[pre_node]
                        if w:
                            pre_node_name = f'({layer},{pre_node})'
                            dot.edge(pre_node_name, node_name)

        with dot.subgraph(name='output') as output:
            for i in range(self.num_outputs()):
                node_name = f'({self.numLayers},{i})'
                output.node(node_name, shape='circle')
                for node in range(self.layerSizes[self.numLayers]):
                    weight = self.weights[self.numLayers - 1][node]
                    for pre_node in range(self.layerSizes[self.numLayers - 1]):
                        w = weight[pre_node]
                        if w:
                            pre_node_name = f'({self.numLayers - 1},{pre_node})'
                            dot.edge(pre_node_name, node_name)

        # dot.draw('visualize/output.png', args='-Gsize=10 -Gratio=1.4', prog='dot')
        dot.render(pic_path, format='jpg')

    def visualize_search_path(self, name="test"):
        step = len(self.splits_set) // 10
        for i in range(0, len(self.splits_set), step):
            pic_path = f"visualize/{name}_{i}_{len(self.splits_set[i])}"
            self.visualize_single_path(self.splits_set[i], pic_path)

    def visualize_single_path(self, splits, pic_path="visualize/visual.jpg"):

        type_param = {
            "Relu active": {
                'assertion': {'style': 'filled', 'fillcolor': '/greens7/6', 'fontsize': '20'},
                'implied': {'style': 'filled', 'fillcolor': '/greens7/4'},
                'eliminated': {'style': 'filled', 'fillcolor': '/greens7/1'}
            },
            "Relu inactive": {
                'assertion': {'style': 'filled', 'fillcolor': '/oranges7/6', 'fontsize': '20'},
                'implied': {'style': 'filled', 'fillcolor': '/oranges7/4'},
                'eliminated': {'style': 'filled', 'fillcolor': '/oranges7/1'}
            }
        }

        def get_node_name(layer, node):
            return f'({layer},{node})'

        node_state = {}
        for layer, node, phase, _, _ in self.eliminated_constraint:
            name = get_node_name(layer, node)
            node_state[name] = {
                'state': 'eliminated',
                'phase': phase,
                'order': 0
            }

        disjunction_set = set()

        for split in splits:
            layer, node, phase, state, order = split
            name = get_node_name(layer, node)
            if layer == 0:
                if name not in node_state:
                    node_state[name] = {
                        'phase': [f'{phase}-{order}'],
                        'state': state,
                        'order': [order]
                    }
                else:
                    node_state[name]['phase'].append(f'{phase}-{order}')
                    node_state[name]['order'].append(order)
                disjunction_set.add(order)
            else:
                node_state[name] = {
                    'phase': phase,
                    'state': state,
                    'order': order
                }

        # input
        dot = Digraph(comment="nnet")
        dot.graph_attr['rankdir'] = 'LR'
        dot.graph_attr['splines'] = 'line'
        dot.graph_attr['ranksep'] = '15'
        dot.graph_attr['ordering'] = 'out'

        with dot.subgraph(name='input') as input:
            for i in range(self.num_inputs()):
                node_name = get_node_name(0, i)
                if node_name in node_state:
                    label = "\n".join(node_state[node_name]['phase'])
                    input.node(node_name, shape='circle', label=label, style='filled', fillcolor='yellow')
                    continue
                input.node(node_name, shape='circle')

        for layer in range(self.numLayers - 1):
            with dot.subgraph(name=f'layer{layer + 1}') as layer_graph:
                for node in range(self.layerSizes[layer + 1]):
                    node_name = get_node_name(layer + 1, node)
                    if node_name in node_state:
                        state = node_state[node_name]
                        label = f"Order: {state['order']}" if state['order'] != 0 else 'e'
                        if state['order'] in disjunction_set:
                            label = str(state['order'])
                        layer_graph.node(node_name, shape='circle', label=label,
                                         **type_param[state['phase']][state['state']])
                    else:
                        layer_graph.node(node_name, shape='circle')
                    weight = self.weights[layer][node]
                    for pre_node in range(self.layerSizes[layer]):
                        w = weight[pre_node]
                        if w != 0:
                            pre_node_name = get_node_name(layer, pre_node)
                            if pre_node_name in node_state and node_state[pre_node_name]['phase'] == 'Relu inactive':
                                # dot.edge(pre_node_name, node_name, color='white')
                                pass
                            else:
                                dot.edge(pre_node_name, node_name)
                            # dot.edge(pre_node_name, node_name)

        with dot.subgraph(name='output') as output:
            for i in range(self.num_outputs()):
                node_name = get_node_name(self.numLayers, i)
                output.node(node_name, shape='circle')
                for node in range(self.layerSizes[self.numLayers]):
                    weight = self.weights[self.numLayers - 1][node]
                    for pre_node in range(self.layerSizes[self.numLayers - 1]):
                        w = weight[pre_node]
                        if w:
                            pre_node_name = get_node_name(self.numLayers - 1, pre_node)
                            if pre_node_name in node_state and node_state[pre_node_name]['phase'] == 'Relu inactive':
                                # dot.edge(pre_node_name, node_name, color='white')
                                pass
                            else:
                                dot.edge(pre_node_name, node_name)

        # dot.draw('visualize/output.png', args='-Gsize=10 -Gratio=1.4', prog='dot')
        dot.render(pic_path, format='jpg')

    def deep_poly_analysis(self):
        class NodeType:
            input = "input"
            output = 'output'
            reluBack = 'relu back'
            reluForward = 'relu forward'

        node_names = {}

        def get_node_name(layer, node, type):
            query = (layer, node, type)
            if query in node_names:
                return node_names[query]
            name = f'({layer}, {node}, {type})'
            node_names[query] = name
            return name

        # node name to coefficient
        node_infos = {}

        class NodeInfo:
            def __init__(self, name) -> None:
                self.name = name
                self.lower_bound, self.upper_bound = None, None
                self.symbolic_lower_bound, self.symbolic_upper_bound = {}, {}
                self.symbolic_lower_bound_of_input = {}
                self.symbolic_upper_bound_of_input = {}
                self.bias = np.float64(0)

            def __str__(self) -> str:
                ret = f"Node name: {self.name}\nLower bound: {self.lower_bound}\n" \
                      f"Upper bound: {self.upper_bound}\n"
                ret += "Symbolic lower: "
                for name, coeff in self.symbolic_lower_bound.items():
                    ret += f"{coeff} {name} "
                ret += "\nSymbolic upper: "
                for name, coeff in self.symbolic_upper_bound.items():
                    ret += f"{coeff} {name} "
                ret += "\n"
                ret += "Symbolic lower to input: "
                for name, coeff in self.symbolic_lower_bound_of_input.items():
                    ret += f"{coeff} {name} "
                ret += "\nSymbolic upper to input: "
                for name, coeff in self.symbolic_upper_bound_of_input.items():
                    ret += f"{coeff} {name} "
                ret += f"\nBias: {self.bias}\n"
                return ret

            def back_propagate_to(self):
                for name, coeff in self.symbolic_lower_bound.items():
                    pre_info = node_infos[name]
                    if coeff < 0:
                        for input, input_coeff in pre_info.symbolic_upper_bound_of_input.items():
                            if input not in self.symbolic_lower_bound_of_input:
                                self.symbolic_lower_bound_of_input[input] = np.float64(0)
                            self.symbolic_lower_bound_of_input[input] += input_coeff * coeff
                        continue
                    for input, input_coeff in pre_info.symbolic_lower_bound_of_input.items():
                        if input not in self.symbolic_lower_bound_of_input:
                            self.symbolic_lower_bound_of_input[input] = np.float64(0)
                        self.symbolic_lower_bound_of_input[input] += input_coeff * coeff

                for name, coeff in self.symbolic_upper_bound.items():
                    pre_info = node_infos[name]
                    if coeff < 0:
                        for input, input_coeff in pre_info.symbolic_lower_bound_of_input.items():
                            if input not in self.symbolic_upper_bound_of_input:
                                self.symbolic_upper_bound_of_input[input] = np.float64(0)
                            self.symbolic_upper_bound_of_input[input] += input_coeff * coeff
                        continue
                    for input, input_coeff in pre_info.symbolic_upper_bound_of_input.items():
                        if input not in self.symbolic_upper_bound_of_input:
                            self.symbolic_upper_bound_of_input[input] = np.float64(0)
                        self.symbolic_upper_bound_of_input[input] += input_coeff * coeff

            def clac_bounds_via_symbol(self, back_to_input=True):
                sym_lower, sym_upper = self.symbolic_lower_bound, self.symbolic_upper_bound
                if back_to_input:
                    if len(self.symbolic_lower_bound_of_input) == 0:
                        self.back_propagate_to()
                    sym_lower = self.symbolic_lower_bound_of_input
                    sym_upper = self.symbolic_upper_bound_of_input

                self.lower_bound, self.upper_bound = self.bias, self.bias
                if sym_lower is not None:
                    for node, coeff in sym_lower.items():
                        pre_node = node_infos[node]
                        if coeff < 0:
                            self.lower_bound += pre_node.upper_bound * coeff
                            continue
                        self.lower_bound += pre_node.lower_bound * coeff
                if sym_upper is not None:
                    for node, coeff in sym_upper.items():
                        pre_node = node_infos[node]
                        if coeff < 0:
                            self.upper_bound += pre_node.lower_bound * coeff
                            continue
                        self.upper_bound += pre_node.upper_bound * coeff

        # add scalar node
        scalar_name = 'scalar'
        sc = NodeInfo(scalar_name)
        sc.lower_bound, sc.upper_bound = np.float64(1), np.float64(1)
        sc.symbolic_lower_bound[scalar_name], sc.symbolic_upper_bound[scalar_name] = np.float64(1), np.float64(1)
        sc.symbolic_lower_bound_of_input[scalar_name], sc.symbolic_upper_bound_of_input[scalar_name] = np.float64(
            1), np.float64(1)
        node_infos[scalar_name] = sc

        # process input
        for node in range(self.layerSizes[0]):
            node_name = get_node_name(0, node, NodeType.input)
            info = NodeInfo(node_name)
            info.lower_bound = np.float64(self.mins[node])
            info.upper_bound = np.float64(self.maxes[node])
            info.symbolic_lower_bound[node_name] = np.float64(1)
            info.symbolic_upper_bound[node_name] = np.float64(1)
            info.symbolic_lower_bound_of_input[node_name] = np.float64(1)
            info.symbolic_upper_bound_of_input[node_name] = np.float64(1)
            node_infos[node_name] = info

            # inital coefficient
        for layer in range(1, self.numLayers):
            for node in range(self.layerSizes[layer]):
                # process back and pre forward
                weight = self.weights[layer - 1][node]
                node_name_back = get_node_name(layer, node, NodeType.reluBack)
                back_info = NodeInfo(node_name_back)
                for pre_node in range(self.layerSizes[layer - 1]):
                    w = weight[pre_node]
                    if w == 0:
                        continue
                    if layer == 1:
                        pre_node_name = get_node_name(layer - 1, pre_node, NodeType.input)
                    else:
                        pre_node_name = get_node_name(layer - 1, pre_node, NodeType.reluForward)

                    back_info.symbolic_lower_bound[pre_node_name] = w
                    back_info.symbolic_upper_bound[pre_node_name] = w
                back_info.bias = self.biases[layer - 1][node]
                back_info.clac_bounds_via_symbol()
                node_infos[node_name_back] = back_info

                # process back and forward
                node_name_forward = get_node_name(layer, node, NodeType.reluForward)
                forward_info = NodeInfo(node_name_forward)
                if back_info.lower_bound > 0:
                    forward_info.symbolic_lower_bound[node_name_back] = np.float64(1)
                    forward_info.symbolic_upper_bound[node_name_back] = np.float64(1)
                elif back_info.upper_bound <= 0:
                    forward_info.symbolic_lower_bound[node_name_back] = np.float64(0)
                    forward_info.symbolic_upper_bound[node_name_back] = np.float64(0)
                else:
                    forward_info.symbolic_lower_bound[node_name_back] = np.float64(0)
                    l, u = back_info.lower_bound, back_info.upper_bound
                    a = u / (u - l)
                    forward_info.symbolic_upper_bound[node_name_back] = np.float64(a)
                    forward_info.symbolic_upper_bound[scalar_name] = np.float64(-l) * np.float64(a)

                forward_info.clac_bounds_via_symbol()
                node_infos[node_name_forward] = forward_info

                print(back_info)
                print(forward_info)
        # output foefficient
        layer = self.numLayers
        for node in range(self.layerSizes[layer]):
            # process back and pre forward
            weight = self.weights[layer - 1][node]
            node_name_back = get_node_name(layer, node, NodeType.output)
            back_info = NodeInfo(node_name_back)
            for pre_node in range(self.layerSizes[layer - 1]):
                w = weight[pre_node]
                if w == 0:
                    continue
                if layer == 1:
                    pre_node_name = get_node_name(layer - 1, pre_node, NodeType.input)
                else:
                    pre_node_name = get_node_name(layer - 1, pre_node, NodeType.reluForward)

                back_info.symbolic_lower_bound[pre_node_name] = w
                back_info.symbolic_upper_bound[pre_node_name] = w
            back_info.bias = self.biases[layer - 1][node]
            back_info.clac_bounds_via_symbol()
            node_infos[node_name_back] = back_info
            print(back_info)

    def recursive_split(self, input_lower, input_upper, depth=10, domains=[]):
        if depth == 0:
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
        self.recursive_split(lower, upper, depth - 1, domains)
        upper[index] = input_upper[index]
        lower[index] = mid
        self.recursive_split(lower, upper, depth - 1, domains)


if __name__ == "__main__":
    nnet_path = "/home/center/CDCL-Marabou/sat_example/prop4/ACASXU_experimental_v2a_1_7.nnet"
    nnet = NNet(nnet_path)
    nnet.eval()
    x = [
        -0.298553, -0.003250, 0.000000, 0.318182, 0.103698
    ]
    x = torch.tensor(x, requires_grad=True)
    y = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)

    print(nnet.satisfy_property(x))

    # print(nnet.get_fn_grand(x, nnet(x)[0]))
    # print(nnet.get_fn_grand(x, nnet(x)[1]))
    # print(nnet.get_fn_grand(x, nnet(x)[2]))
    # print(nnet.get_fn_grand(x, nnet(x)[3]))
    # print(nnet.get_fn_grand(x, nnet(x)[4]))
    x_fgm = projected_gradient_descent(nnet, x, 0.1, 0.1, 2, np.inf, y=y, targeted=True)
    # print(x_fgm, nnet.forward(x_fgm))
