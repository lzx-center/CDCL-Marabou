import numpy as np
import json
import os
import random
from graphviz import Digraph, Graph
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

class NNet():
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
    def __init__ (self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline() 
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            weights=[]
            biases = []
            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                weights.append([])
                biases.append([])
                weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
                for i in range(currentLayerSize):
                    line=f.readline()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    for j in range(previousLayerSize):
                        weights[layernum][i,j] = aux[j]
                #biases
                biases[layernum] = np.zeros(currentLayerSize)
                for i in range(currentLayerSize):
                    line=f.readline()
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
                if inputs[i]<self.mins[i]:
                    inputsNorm[i] = (self.mins[i]-self.means[i])/self.ranges[i]
                elif inputs[i]>self.maxes[i]:
                    inputsNorm[i] = (self.maxes[i]-self.means[i])/self.ranges[i] 
                else:
                    inputsNorm[i] = (inputs[i]-self.means[i])/self.ranges[i] 

        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer],0)
        outputs = np.dot(weights[-1],inputsNorm)+biases[-1]

        # Undo output normalization
        if undo_normalize_output:
            for i in range(outputSize):
                outputs[i] = outputs[i]*self.ranges[-1]+self.means[-1]
        return outputs

    def evaluate_network_multiple(self,inputs):
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
        inputsNorm = np.zeros((inputSize,numInputs))
        for i in range(inputSize):
            for j in range(numInputs):
                if inputs[i,j]<self.mins[i]:
                    inputsNorm[i,j] = (self.mins[i]-self.means[i])/self.ranges[i]
                elif inputs[i,j] > self.maxes[i]:
                    inputsNorm[i,j] = (self.maxes[i]-self.means[i])/self.ranges[i] 
                else:
                    inputsNorm[i,j] = (inputs[i,j]-self.means[i])/self.ranges[i]

        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer].reshape((len(biases[layer]),1)),0)
        outputs = np.dot(weights[-1],inputsNorm)+biases[-1].reshape((len(biases[-1]),1))

        # Undo output normalization
        for i in range(outputSize):
            for j in range(numInputs):
                outputs[i,j] = outputs[i,j]*self.ranges[-1]+self.means[-1]
        return outputs.T

    def num_inputs(self):
        ''' Get network input size'''
        return self.inputSize

    def num_outputs(self):
        ''' Get network output size'''
        return self.outputSize
 
    def get_node_relation(self, layer, node):
        return [self.weights[layer - 1][node, i] for i in range(self.layerSizes[layer-1])]

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
                if inputs[i]<self.mins[i]:
                    inputsNorm[i] = (self.mins[i]-self.means[i])/self.ranges[i]
                elif inputs[i]>self.maxes[i]:
                    inputsNorm[i] = (self.maxes[i]-self.means[i])/self.ranges[i] 
                else:
                    inputsNorm[i] = (inputs[i]-self.means[i])/self.ranges[i] 

        # Evaluate the neural network
        state = [[]]
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer],0)
            state.append((inputsNorm > 0).tolist())
        outputs = np.dot(weights[-1],inputsNorm)+biases[-1]
        # Undo output normalization
        if undo_normalize_output:
            for i in range(outputSize):
                outputs[i] = outputs[i]*self.ranges[-1]+self.means[-1]
        self.calc_states.append(state)
        return outputs

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
        dot.render(pic_path,format='jpg')  

    def visualize_search_path(self, name="test"):
        step = len(self.splits_set) // 10
        for i in range(0, len(self.splits_set), step):
            pic_path = f"visualize/{name}_{i}_{len(self.splits_set[i])}"
            self.visualize_single_path(self.splits_set[i], pic_path)
            
            

    def visualize_single_path(self, splits, pic_path="visualize/visual.jpg"):

        type_param = {
            "Relu active" : {
                'assertion' : {'style' : 'filled', 'fillcolor' : '/greens7/6', 'fontsize' : '20'},
                'implied'   : {'style' : 'filled', 'fillcolor' : '/greens7/4'},
                'eliminated'   : {'style' : 'filled', 'fillcolor' : '/greens7/1'}
            },
            "Relu inactive" : {
                'assertion' : {'style' : 'filled', 'fillcolor' : '/oranges7/6', 'fontsize' : '20'},
                'implied'   : {'style' : 'filled', 'fillcolor' : '/oranges7/4'},
                'eliminated'   : {'style' : 'filled', 'fillcolor' : '/oranges7/1'}
            }
        }


        def get_node_name(layer, node):
            return f'({layer},{node})'

        node_state = {}
        for layer, node, phase, _, _ in self.eliminated_constraint:
            name = get_node_name(layer, node)
            node_state[name] = {
                'state' : 'eliminated',
                'phase' : phase,
                'order' : 0
            }

        disjunction_set = set()

        for split in splits:
            layer, node, phase, state, order = split
            name = get_node_name(layer, node)
            if layer == 0:
                if name not in node_state:
                    node_state[name] = {
                        'phase' : [f'{phase}-{order}'],
                        'state' : state,
                        'order' : [order]
                    }
                else:
                    node_state[name]['phase'].append(f'{phase}-{order}')
                    node_state[name]['order'].append(order)
                disjunction_set.add(order)
            else:
                node_state[name] = {
                    'phase' : phase,
                    'state' : state,
                    'order' : order
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
                        layer_graph.node(node_name, shape='circle', label=label, **type_param[state['phase']][state['state']])
                    else:
                        layer_graph.node(node_name, shape='circle')
                    weight = self.weights[layer][node]
                    for pre_node in range(self.layerSizes[layer]):
                        w = weight[pre_node]
                        if w != 0:
                            pre_node_name = get_node_name(layer,pre_node)
                            if pre_node_name in node_state and node_state[pre_node_name]['phase'] == 'Relu inactive':
                                # dot.edge(pre_node_name, node_name, color='white')
                                pass
                            else:
                                dot.edge(pre_node_name, node_name)
                            # dot.edge(pre_node_name, node_name)

        with dot.subgraph(name='output') as output:
            for i in range(self.num_outputs()):
                node_name = get_node_name(self.numLayers,i)
                output.node(node_name, shape='circle')
                for node in range(self.layerSizes[self.numLayers]):
                    weight = self.weights[self.numLayers - 1][node]
                    for pre_node in range(self.layerSizes[self.numLayers - 1]):
                        w = weight[pre_node]
                        if w:
                            pre_node_name = get_node_name(self.numLayers - 1,pre_node)
                            if pre_node_name in node_state and node_state[pre_node_name]['phase'] == 'Relu inactive':
                                # dot.edge(pre_node_name, node_name, color='white')
                                pass
                            else:
                                dot.edge(pre_node_name, node_name)

        # dot.draw('visualize/output.png', args='-Gsize=10 -Gratio=1.4', prog='dot')
        dot.render(pic_path,format='jpg')  

class TestGenerator:
    def __init__(self) -> None:
        pass

    def generator(file_path, property_path=None):
        layers, inputs, outputs, maximum_layer_size = 3, 1, 1, 2
        layer_nums = [inputs, 3, 2, outputs]

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
                        printf(random.uniform(-1, 1), end=",")
                    printf()
                for h in range(layer_nums[i + 1]):
                    printf("0.0,")
                bound = random.uniform(-2, 2)
                with open(property_path, "w+") as t:
                    def printt(*arg, **kwargs):
                        print(*arg, **kwargs, file=t)

                    printt(f"y0 >= {bound}")
        command = f"/home/center/Delta-Marabou/Marabou {file_path} {property_path} > ./test.log"
        print(command)
        os.system(command)
        with open("./test.log") as f:
            unsat = False
            for line in f.readlines():
                if "unsat" in line:
                    unsat = True
                if "Search Tree" in line:
                    size = int(line.split(" ")[-1].strip("\n"))
                    print(size)
                    if unsat:
                        return size
        return 0


if __name__ == "__main__":
    nnet_path = "example/ACASXU_experimental_v2a_1_2.nnet"
    nnet = NNet(nnet_path)
    # nnet.evaluate_state(inputs=[0.2])
    nnet.load_json("build/acas1_2.json")
    nnet.visualize_search_path('acas1_2')
    # nnet.visual()
