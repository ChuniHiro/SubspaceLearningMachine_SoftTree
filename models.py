""" Different models """
import warnings
import torch
import ops
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ops import get_leaf_nodes, get_past_leaf_nodes, get_path_to_root
import numpy as np
import math
from einops import rearrange
class Tree(nn.Module):
    """ Adaptive Neural Tree module. """
    def __init__(self,
                 tree_struct, tree_modules,
                 split=False,
                 node_split=None, child_left=None, child_right=None,
                 extend=False,
                 node_extend=None, child_extension=None,
                 cuda_on=True,
                 breadth_first=True,
                 soft_decision=True):
        """ Initialise the class.

        Args:
            tree_struct (list): List of dictionaries each of which contains
                meta information about each node of the tree.
            tree_modules (list): List of dictionaries, each of which contains
                modules (nn.Module) of each node in the tree and takes the form
                module = {'edger': transformer_module (nn.Module),
                          'classifier': solver_module (nn.Module),
                          'router': router_module (nn.Module) }
            split (bool): Set True if the model is testing 'split' growth option
            node_split (int): Index of the node that is being split
            child_left (dict): Left child of the node node_split and takes the
                form of {'edger': transformer_module (nn.Module),
                          'classifier': solver_module (nn.Module),
                          'router': router_module (nn.Module) }
            child_right (dict): Right child of the node node_split and takes the
                form of {'transform': transformer_module (nn.Module),
                          'classifier': solver_module (nn.Module),
                          'router': router_module (nn.Module) }
            extend (bool): Set True if the model is testing 'extend'
                growth option
            node_extend (int): Index of the node that is being extended
            child_extension (dict): The extra node used to extend node
                node_extend.
            cuda_on (bool): Set True to train on a GPU.
            breadth_first (bool): Set True to perform bread-first forward pass.
                If set to False, depth-first forward pass is performed.
            soft_decision (bool): Set True to perform multi-path inference,
                which computes the predictive distribution as the mean
                of the conditional distributions from all the leaf nodes,
                weighted by the corresponding reaching probabilities.

                If set to False, inference based on "hard" decisions is
                performed. If the routers are defined with
                stochastic=True, then the stochastic single-path inference
                is used. Otherwise, the greedy single-path inference is carried
                out whereby the input sample traverses the tree in the
                directions of the highest confidence of routers.
        """
        super(Tree, self).__init__()

        assert not(split and extend)  # the node can only be split or extended
        self.soft_decision = soft_decision
        self.cuda_on = cuda_on
        self.split = split
        self.extend = extend
        self.tree_struct = tree_struct
        self.node_split = node_split
        self.node_extend = node_extend
        self.breadth_first = breadth_first

        # get list of leaf nodes:
        self.leaves_list = get_leaf_nodes(tree_struct)
        # for each leaf predictor, get the list of all nodes (indices) on
        # their paths to the root and the corresponding lef-child-status
        # (boolean) on all edges i.e. edge = True if the child is on the left
        # branch of its parent. Each element in self.paths_list is a tuple
        # (nodes, edges) which contains these two lists.
        self.paths_list = [
            get_path_to_root(i, tree_struct) for i in self.leaves_list]

        self.tree_modules = nn.ModuleList()
        for i, node in enumerate(tree_modules):
            node_modules = nn.Sequential()
            node_modules.add_module('transform', node["transform"])
            node_modules.add_module('classifier', node["classifier"])
            node_modules.add_module('router', node["router"])
            self.tree_modules.append(node_modules)

        # add children nodes:
        # case (1): splitting
        if split:
            self.child_left = nn.Sequential()
            self.child_left.add_module('transform', child_left["transform"])
            self.child_left.add_module('classifier', child_left["classifier"])
            self.child_left.add_module('router', child_left["router"])
            self.child_right = nn.Sequential()
            self.child_right.add_module('transform', child_right["transform"])
            self.child_right.add_module('classifier', child_right["classifier"])
            self.child_right.add_module('router', child_right["router"])
        
        # case (2): making deeper
        if extend:
            self.child_extension = nn.Sequential()
            self.child_extension.add_module(
                'transform', child_extension["transform"],
            )
            self.child_extension.add_module(
                'classifier', child_extension["classifier"],
            )
            self.child_extension.add_module(
                'router', child_extension["router"],
            )

    def forward(self, input):
        """Choose breadth-first/depth-first inference"""
        if self.breadth_first:
            return self.forward_breadth_first(input)
        else:
            return self.forward_depth_first(input)

    def forward_depth_first(self, input):
        """ Depth first forward pass.
        Args:
            input: A tensor of size (batch, channels, width, height)
        Return:
            log soft-max probabilities (tensor) of size (batch, classes)
            If self.training = True, it also returns the probability of reaching
                the last node.
        """
        y_pred = 0.0
        prob_last = None 

        for (nodes, edges) in self.paths_list:
            # split the node and perform prediction
            if self.split and nodes[-1] == self.node_split:
                y_tmp, prob_last = self.node_pred_split(input, nodes, edges)
                y_pred += y_tmp
            elif self.extend and nodes[-1] == self.node_extend:
                y_tmp, prob_last = self.node_pred_extend(input, nodes, edges)
                y_pred += y_tmp
            else:
                y_pred += self.node_pred(input, nodes, edges)
        
        if self.training:
            return torch.log(1e-10 + y_pred), prob_last
        else:
            return torch.log(1e-10 + y_pred)

    def forward_breadth_first(self, input):
        """ Breadth first forward pass.

        Notes:
            In the current implementation, tree_struct is constructed level
            by level. So, sequentially iterating tree_struct naturally leads
            to breadth first inference.
        """
        t_list = [self.tree_modules[0].transform(input)]  # transformed inputs
        r_list = [1.0]  # list of reaching probabilities
        s_list = []  # list of classifier outputs on the transformed inputs
        prob_last = 1.0

        for node in self.tree_struct:
            inp = t_list.pop(0)
            ro = r_list.pop(0)

            # if the node is the target
            if self.split and node['index'] == self.node_split:
                s_list.append(self.child_left.classifier(self.child_left.transform(inp)))
                s_list.append(self.child_right.classifier(self.child_right.transform(inp)))
            
                p_left = self.tree_modules[node['index']].router(inp)
                p_left = torch.unsqueeze(p_left, 1)
                prob_last = p_left

                r_list.append(ro * p_left)
                r_list.append(ro * (1.0 - p_left))

            elif self.extend and node['index'] == self.node_extend:
                s_list.append(self.child_extension.classifier(self.child_extension.transform(inp)))          
                p_left = 1.0
                r_list.append(ro * p_left)
            
            # if the node is a leaf node,
            elif node['is_leaf']: 
                s_list.append(self.tree_modules[node['index']].classifier(inp))
                r_list.append(ro)
            elif node['extended']:
                t_list.append(self.tree_modules[node['left_child']].transform(inp))
                p_left = self.tree_modules[node['index']].router(inp)
                r_list.append(ro * p_left)
            else:
                t_list.append(self.tree_modules[node['left_child']].transform(inp))
                t_list.append(self.tree_modules[node['right_child']].transform(inp))
                p_left = self.tree_modules[node['index']].router(inp)
                p_left = torch.unsqueeze(p_left, 1)
                r_list.append(ro * p_left)
                r_list.append(ro * (1.0 - p_left))

        # combine and perform inference:
        y_pred = 0.0
        # print("debug")
        # print("tlist")
        # print(t_list)
        # print("rlist")
        # print(r_list)
        # print("slist")
        # print(s_list)
        for r, s in zip(r_list, s_list):
            
            # print("inside loop")
            # print(r)
            # print(r.shape)
            # print(s.shape)
            # print(torch.exp(s).shape)
            # print(r * torch.exp(s))
            # print(y_pred)
            # print()
            y_pred += r * torch.exp(s)

        out = torch.log(1e-10 + y_pred)

        if self.training:
            return out, prob_last
        else:
            return out

    def node_pred(self, input, nodes, edges):
        """ Perform prediction on a given node given its path on the tree.
        e.g.
        nodes = [0, 1, 4, 10]
        edges = [True, False, False]
        """
        # Transform data and compute probability of reaching
        # the last node in path
        prob = 1.0
        for node, state in zip(nodes[:-1], edges):
            input = self.tree_modules[node].transform(input)
            if state:
                prob = prob * self.tree_modules[node].router(input)
            else:
                prob = prob * (1.0 - self.tree_modules[node].router(input))

        if not (isinstance(prob, float)):
            print("prob before:", prob)
            prob = torch.unsqueeze(prob, -1)
            print("prob after:", prob)

        node_final = nodes[-1]
        input = self.tree_modules[node_final].transform(input)

        # Perform classification with the last node:
        y_pred = prob * torch.exp(
            self.tree_modules[node_final].classifier(input))

        return y_pred

    def node_pred_split(self, input, nodes, edges):
        """ Perform prediction on a split node given its path on the tree.
        Here, the last node in the  list "nodes" is assumed to be split.
        e.g.
        nodes = [0, 1, 4, 10]
        edges = [True, False, False]
        then, node 10 is assumed to be split.

        Args:
            input (torch.Variable): input images
            nodes (list): list of all nodes (index) on the path between root
                and given node
            edges (list): list of left-child-status (boolean) of each edge
                between nodes in the list 'nodes'
        Returns:
            y_pred (torch.Variable): predicted label
            prob_last (torch.Variable): output of the parent router
            (if self.training=True)
        """

        # Transform data and compute prob of reaching the last node in path
        prob = 1.0
        for node, state in zip(nodes[:-1], edges):
            input = self.tree_modules[node].transform(input)
            if state:
                prob = prob * self.tree_modules[node].router(input)
            else:
                prob = prob * (1.0 - self.tree_modules[node].router(input))

        if not (isinstance(prob, float)):
            prob = torch.unsqueeze(prob, 1)

        node_final = nodes[-1]
        input = self.tree_modules[node_final].transform(input)

        # Perform classification with the last node:
        prob_last = torch.unsqueeze(
            self.tree_modules[node_final].router(input), 1,
        )

        # Split the last node:
        y_pred = prob * (prob_last * torch.exp(
            self.child_left.classifier(self.child_left.transform(input)))
                         + (1.0 - prob_last) * torch.exp(
            self.child_right.classifier(self.child_right.transform(input)))
                         )
        return y_pred, prob_last

    def node_pred_extend(self, input, nodes, edges):
        """ Perform prediction on an extended node given its path on the tree.
        Here, the last node in the  list "nodes" is assumed to be split.
        e.g.
        nodes = [0, 1, 4, 10]
        edges = [True, False, False]
        then, node 10 is assumed to be split.

        Args:
            input (torch.Variable): input images
            nodes (list): list of all nodes (index) on the path between root and given node
            edges (list): list of left-child-status (boolean) of each edge between nodes in
                          the list 'nodes'
        Return:
            y_pred (torch.Variable): predicted label
            prob_last (torch.Variable): output of the parent router (if self.training=True)
        """

        # Transform data and compute probability of
        # reaching the last node in path
        prob = 1.0
        for node, state in zip(nodes[:-1], edges):
            input = self.tree_modules[node].transform(input)
            if state:
                prob = prob * self.tree_modules[node].router(input)
            else:
                prob = prob * (1.0 - self.tree_modules[node].router(input))

        if not (isinstance(prob, float)):
            prob = torch.unsqueeze(prob, 1)

        # TODO: need to make prob_last a vector of ones instead of a scaler?
        prob_last = 1.0
        node_final = nodes[-1]
        input = self.tree_modules[node_final].transform(input)

        # Perform classification with the last node:
        y_pred = prob * torch.exp(self.child_extension.classifier(
            self.child_extension.transform(input)))

        return y_pred, prob_last

    def compute_routing_probabilities(self, input):
        """ Compute routing probabilities for all nodes in the tree.

        Return:
            routing probabilities tensor (tensor) : torch tensor (N, num_nodes)
        """
        for i, (nodes, edges) in enumerate(self.paths_list):
            # compute probabilities for the given branch
            prob = 1.0
            for node, state in zip(nodes[:-1], edges):
                input = self.tree_modules[node].transform(input)
                if state:
                    prob = prob * self.tree_modules[node].router(input)
                else:
                    prob = prob * (1.0 - self.tree_modules[node].router(input))

            if not (isinstance(prob, float)):
                prob = torch.unsqueeze(prob, 1)

            # account for the split at the last node
            if self.split and nodes[-1] == self.node_split:
                node_final = nodes[-1]
                input = self.tree_modules[node_final].transform(input)
                prob_last = torch.unsqueeze(self.tree_modules[node_final].router(input), 1)
                prob = torch.cat((prob_last*prob, (1.0-prob_last)*prob), dim=1)

            # concatenate
            if i == 0:
                prob_tensor = prob
            else:
                prob_tensor = torch.cat((prob_tensor, prob), dim=1)

        return prob_tensor

    def compute_routing_probability_specificnode(self, input, node_idx):
        """ Compute the probability of reaching a selected node.
        If a batch is provided, then the sum of probabilities is computed.
        """ 
        
        nodes, edges = get_path_to_root(node_idx, self.tree_struct)
        prob = 1.0

        for node, edge in zip(nodes[:-1], edges):
            input = self.tree_modules[node].transform(input)
            if edge:
                prob = prob * self.tree_modules[node].router(input)
            else:
                prob = prob * (1.0 - self.tree_modules[node].router(input))

        if not (isinstance(prob, float)):
            prob = torch.unsqueeze(prob, 1)
            prob_sum = prob.sum(dim=0)
            return prob_sum.data[0]
        else:
            return prob*input.size(0)

    def compute_routing_probabilities_uptonode(self, input, node_idx):
        """ Compute the routing probabilities up to a node.

        Return:
            routing probabilities tensor (tensor) : torch tensor (N, nodes)

        """
        leaves_up_to_node = get_past_leaf_nodes(self.tree_struct, node_idx)

        # for each leaf predictor, get the list of all nodes (indices) on
        # their paths to the root and the corresponding lef-child-status
        # (boolean) on all edges i.e. edge = True if the child is on the left
        # branch of its parent. Each element in self.paths_list is a tuple
        # (nodes, edges) which contains these two lists.
        paths_list_up_to_node = [get_path_to_root(i, self.tree_struct)
                                 for i in leaves_up_to_node]
        
        for i, (nodes, edges) in enumerate(paths_list_up_to_node):
            # compute probabilities for the given branch
            # if len(nodes)>1:
            #     prob = 1.0
            # else: # if it's just a root node
            dtype = torch.cuda.FloatTensor if self.cuda_on else torch.FloatTensor
            prob = Variable(torch.ones(input.size(0)).type(dtype))
            output = input.clone()

            for node, state in zip(nodes[:-1], edges):
                output = self.tree_modules[node].transform(output)
                if state:
                    prob = prob * self.tree_modules[node].router(output)
                else:
                    prob = prob * (1.0 - self.tree_modules[node].router(output))

            if not (isinstance(prob, float)):
                prob = torch.unsqueeze(prob, 1)

            # account for the split at the last node
            if self.split and nodes[-1] == self.node_split:
                node_final = nodes[-1]
                output = self.tree_modules[node_final].transform(output)
                prob_last = torch.unsqueeze(
                    self.tree_modules[node_final].router(output), 1)
                prob = torch.cat((prob_last*prob, (1.0-prob_last)*prob), dim=1)

            # concatenate
            if i == 0:
                prob_tensor = prob
            else:
                prob_tensor = torch.cat((prob_tensor, prob), dim=1)

        return prob_tensor, leaves_up_to_node

    def update_tree_modules(self):
        """
        Return tree_modules (list) with the current parameters.
        """
        tree_modules_new=[]
        for node_module in self.tree_modules:
            node = {'transform' :node_module.transform,
                    'classifier':node_module.classifier,
                    'router': node_module.router}
            tree_modules_new.append(node)
        return tree_modules_new

    def update_children(self):
        assert self.split or self.extend
        if self.split:
            child_left= {'transform' : self.child_left.transform,
                        'classifier': self.child_left.classifier,
                        'router': self.child_left.router}
            child_right= {'transform' :self.child_right.transform,
                        'classifier':self.child_right.classifier,
                        'router': self.child_right.router}
            print("returning left and right children")
            return child_left, child_right
        elif self.extend:
            child_extension= {'transform' : self.child_extension.transform,
                              'classifier': self.child_extension.classifier,
                              'router': self.child_extension.router}
            print("returning an extended child")
            return child_extension


# ############################ Building blocks  ##############################
# ########################### (1) Transformers ###############################
class Identity(nn.Module):
    def __init__(self,input_nc, input_width, input_height, **kwargs):
        super(Identity, self).__init__()
        self.outputshape = (1, input_nc, input_width, input_height)

    def forward(self, x):
        return x


class JustConv(nn.Module):
    """ 1 convolution """
    def __init__(self, input_nc, input_width, input_height, 
                 ngf=6, kernel_size=3, stride=1, **kwargs):
        super(JustConv, self).__init__()

        # print("check just conv:", stride)
        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)

        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size, stride=stride, padding=kernel_size//2)
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the transformer to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out


class ConvPool(nn.Module):
    """ 1 convolution + 1 max pooling """
    def __init__(self, input_nc, input_width, input_height, 
                 ngf=6, kernel_size=5, downsample=True, **kwargs):
        super(ConvPool, self).__init__()
        self.downsample = downsample

        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)
            self.downsample = False

        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size)
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the transformer to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        if self.downsample:
            return F.max_pool2d(out, 2)
        else:
            return out


class ResidualTransformer(nn.Module):
    """ Bottleneck without batch-norm
    Got the base codes from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, input_nc, input_width, input_height, 
                 ngf=6, stride=1, **kwargs):
        super(ResidualTransformer, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            ngf, ngf, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.conv3 = nn.Conv2d(ngf, input_nc, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)
        
    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the transformer to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)

        return self.forward(x).data.numpy().shape

    def forward(self, x):

        print("using bottleneck residual transformer")
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        out = self.relu(out)
        return out


class VGG13ConvPool(nn.Module):
    """ n convolution + 1 max pooling """
    def __init__(self, input_nc, input_width, input_height, 
                 ngf=64, kernel_size=3, batch_norm=True, downsample=True,
                 **kwargs):
        super(VGG13ConvPool, self).__init__()
        self.downsample = downsample        
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(
            input_nc, ngf, kernel_size=kernel_size, padding=(kernel_size-1)/2,
        )
        self.conv2 = nn.Conv2d(ngf, ngf, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(ngf)
            self.bn2 = nn.BatchNorm2d(ngf)
            # self.bn3 = nn.BatchNorm2d(ngf)

        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the transformer to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()

    def forward(self, x):
        if self.batch_norm:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
        else:
            out = self.relu(self.conv1(x))
            out = self.relu(self.conv2(out))
        
        if self.downsample:
            return F.max_pool2d(out, 2)
        else:
            return out

# Mobilenet v2 : reverse residual block for Edge

def make_divisible(x, divisible_by=8):
    
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidualv3(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidualv3, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    # def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Edge_MBV2(nn.Module):

    def __init__(self, input_nc, input_width, input_height, num_classes = 10, width_mult = 1.0,stride = 1, ngf = 32, expansion_rate = 2, **kwargs):
        super(Edge_MBV2, self).__init__()
        block = InvertedResidual
        input_channel = input_nc

        interverted_residual_setting = [
            # t, c, n, s
            [expansion_rate, ngf, 1, stride],
        ]

        # print("using MBV2")
        # print("check settings")
        # print(interverted_residual_setting)

        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # print("chekc MBv2 model")
        # print(self.features)
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the edger to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()


class Root_MBV2(nn.Module):

    def __init__(self, input_nc, input_width, input_height, num_classes = 10, expansion_rate = 6, width_mult = 1.0,stride = 1, ngf = 32, **kwargs):
        super(Root_MBV2, self).__init__()
        block = InvertedResidual
        input_channel = input_nc

        # interverted_residual_setting = [
        #     # standard setting for mobilev2
        #     # t, c, n, s
        #     [1, 16, 1, 1],
        #     [6, 24, 2, 2],
        #     [6, 32, 3, 2],
        #     [6, 64, 4, 2],
        #     [6, 96, 3, 1],
        #     [6, 160, 3, 2],
        #     [6, 320, 1, 1],
        #     [6, 1280, 1, 1],
        #     [6, 1280, 1, 1],
        # ]
        
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [expansion_rate, 24, 2, 2],
            [expansion_rate, 32, 3, 2],
            [expansion_rate, 64, 4, 2],
            [expansion_rate, 96, 3, 1],
            [expansion_rate, 160, 3, 2],
            [expansion_rate, 320, 1, 1],
        ]
        # print("using MBV2")
        # print("check settings")
        # print(interverted_residual_setting)

        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # print("chekc MBv2 model")
        # print(self.features)
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the edger to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()

class Root_MBV2light(nn.Module):

    def __init__(self, input_nc, input_width, input_height, num_classes = 10, width_mult = 1.0,stride = 1, ngf = 32,  expansion_rate = 6, **kwargs):
        super(Root_MBV2light, self).__init__()
        block = InvertedResidual
        input_channel = input_nc
        
        print("check width mult", width_mult)
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expansion_rate, 24, 2, 2],
            [expansion_rate, 32, 2, 2],
            [expansion_rate, 64, 3, 2],
            [expansion_rate, 96, 2, 1],
            [expansion_rate, 160, 2, 2],
        ]

        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # print("chekc MBv2 model")
        # print(self.features)
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the edger to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()

class Root_MBV2tiny(nn.Module):

    def __init__(self, input_nc, input_width, input_height, num_classes = 10, width_mult = 1.0,stride = 1, ngf = 32,  expansion_rate = 6, **kwargs):
        super(Root_MBV2tiny, self).__init__()
        block = InvertedResidual
        input_channel = input_nc
        
        print("check width mult", width_mult)
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expansion_rate, 32, 2, 2],
            [expansion_rate, 64, 2, 2],
            [expansion_rate, 96, 2, 2],
            [expansion_rate, 160, 1, 2],
        ]

        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # print("chekc MBv2 model")
        # print(self.features)
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the edger to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()

class Root_MobileNetV3(nn.Module):
    def __init__(self, input_nc, input_width, input_height, mode='small', num_classes=10, width_mult=1., **kwargs):
        super(Root_MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        assert mode in ['large', 'small']

        settings = {

            'small': [
            # # k, t, c, SE, HS, s 
            [3,    1,  16, 1, 0, 2],
            [3,  4.5,  24, 0, 0, 2],
            [3, 3.67,  24, 0, 0, 1],
            [5,    4,  40, 1, 1, 2],
            [5,    6,  40, 1, 1, 1],
            [5,    6,  40, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    6,  96, 1, 1, 2],
            [5,    6,  96, 1, 1, 1],
            [5,    6,  96, 1, 1, 1],
            ],
            # 'small': [
            # # k, t, c, SE, HS, s 
            # [3,    1,  16, 1, 0, 2],
            # [3,  4.5,  24, 0, 0, 2],
            # [3, 3.67,  24, 0, 0, 1],
            # [5,    4,  40, 1, 1, 2],
            # [5,    6,  40, 1, 1, 1],
            # [5,    6,  40, 1, 1, 1],
            # [5,    3,  48, 1, 1, 1],
            # [5,    3,  48, 1, 1, 1],
            # [5,    6,  96, 1, 1, 2],
            # [5,    6,  96, 1, 1, 1],
            # ],
            'large' : [
            # k, t, c, SE, HS, s 
            [3,   1,  16, 0, 0, 1],
            [3,   4,  24, 0, 0, 2],
            [3,   3,  24, 0, 0, 1],
            [5,   3,  40, 1, 0, 2],
            [5,   3,  40, 1, 0, 1],
            [5,   3,  40, 1, 0, 1],
            [3,   6,  80, 0, 1, 2],
            [3, 2.5,  80, 0, 1, 1],
            [3, 2.3,  80, 0, 1, 1],
            [3, 2.3,  80, 0, 1, 1],
            [3,   6, 112, 1, 1, 1],
            [3,   6, 112, 1, 1, 1],
            [5,   6, 160, 1, 1, 2],
            [5,   6, 160, 1, 1, 1],
            [5,   6, 160, 1, 1, 1]
            ]
        }
        
        # building first layer
        input_channel = make_divisible(16 * width_mult, 8)
        # print("input_nc, input_width, input_height", input_nc, input_width, input_height)
        layers = [conv_3x3_bn(input_nc, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidualv3
        for k, t, c, use_se, use_hs, s in settings[mode]:
            output_channel = make_divisible(c * width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the edger to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(2, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()


## vision transformer block

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class Edge_MBVIT(nn.Module):

    def __init__(self, input_nc, input_width, input_height, num_classes = 10, width_mult = 1.0,stride = 1, ngf = 32, expansion_rate = 2, **kwargs):
        super(Edge_MBVIT, self).__init__()
        block = MobileViTBlock
        input_channel = input_nc

        # self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        dims = ngf
        channels = input_nc
        L = 2
        kernel_size = 3
        patch_size = (2,2)

        # print("using MBVIT")

        self.features = MobileViTBlock(dims, L, channels, kernel_size, patch_size, int(dims*2))
        self.outputshape = self.get_outputshape(input_nc, input_width, input_height)

    def forward(self, x):
        x = self.features(x)
        return x


    def get_outputshape(self, input_nc, input_width, input_height ):
        """ Run a single forward pass through the edger to get the 
        output size
        """
        dtype = torch.FloatTensor
        x = Variable(
            torch.randn(1, input_nc, input_width, input_height).type(dtype),
            requires_grad=False)
        return self.forward(x).size()


# ########################### (2) Routers ##################################
class One(nn.Module):
    """Route all data points to the left branch branch """
    def __init__(self):
        super(One, self).__init__()
        
    def forward(self, x):
        return 1.0


class Router(nn.Module):
    """Convolution + Relu + Global Average Pooling + Sigmoid"""
    def __init__(self, input_nc,  input_width, input_height,
                 kernel_size=28,
                 soft_decision=True,
                 stochastic=False,
                 **kwargs):
        super(Router, self).__init__()
        self.soft_decision = soft_decision
        self.stochastic=stochastic

        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)

        self.conv1 = nn.Conv2d(input_nc, 1, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # convolution
        # TODO: x = F.relu(self.conv1(x))
        x = self.conv1(x)
        # spatial averaging
        x = x.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
        # get probability of "left" or "right"
        x = self.output_controller(x)
        return x
                
    def output_controller(self, x):

        # soft decision
        if self.soft_decision:
            return self.sigmoid(x)

        # stochastic hard decision:
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        
        # soft greedy decision:
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)


class RouterGAP(nn.Module):
    """ Convolution + Relu + Global Average Pooling + FC + Sigmoid """

    def __init__(self, input_nc, input_width, input_height, 
                 ngf=5,
                 kernel_size=7,
                 soft_decision=True,
                 stochastic=False,
                 **kwargs):

        super(RouterGAP, self).__init__()
        self.ngf = ngf
        self.soft_decision = soft_decision
        self.stochastic = stochastic

        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)

        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=kernel_size)
        self.linear1 = nn.Linear(ngf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # convolution
        x = self.conv1(x)

        # spatial averaging and fully connected layer
        if self.ngf == 1:
            x = x.mean(dim=-1).mean(dim=-1).squeeze()
        else:
            x = F.relu(x)
            x = x.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
            x = self.linear1(x).squeeze()

        # get probability of "left" or "right"
        output = self.sigmoid(x)

        if self.soft_decision:
            return output

        if self.stochastic:
            return ops.ST_StochasticIndicator()(output)
        else:
            return ops.ST_Indicator()(output)


class RouterGAPwithDoubleConv(nn.Module):
    """ 2 x (Convolution + Relu) + Global Average Pooling + FC + Sigmoid """

    def __init__(self, input_nc, input_width, input_height, 
                 ngf=32,
                 kernel_size=3,
                 soft_decision=True,
                 stochastic=False,
                 **kwargs):

        super(RouterGAPwithDoubleConv, self).__init__()
        self.ngf = ngf
        self.soft_decision = soft_decision
        self.stochastic = stochastic

        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)
            if max(input_width, input_height)%2 ==0:
                kernel_size += 1
 
        padding = (kernel_size-1)/2 
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(ngf, ngf, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(ngf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # convolution
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        # spatial averaging and fully connected layer
        out = out.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
        out = self.linear1(out).squeeze()
        # get probability of "left" or "right"
        out = self.output_controller(out)
        return out

    def output_controller(self, x):
        # soft decision
        if self.soft_decision:
            return self.sigmoid(x)

        # stochastic hard decision:
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)


class Router_MLP_h1(nn.Module):
    """  MLP with 1 hidden layer """
    def __init__(self, input_nc,  input_width, input_height,
                 kernel_size=28,
                 soft_decision=True,
                 stochastic=False,
                 reduction_rate=2,
                 **kwargs):
        super(Router_MLP_h1, self).__init__()
        self.soft_decision = soft_decision
        self.stochastic=stochastic

        width = input_nc*input_width*input_height
        self.fc1 = nn.Linear(width, int(width/reduction_rate) + 1)
        self.fc2 = nn.Linear(int(width/reduction_rate) + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 2 fc layers:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze()
        # get probability of "left" or "right"
        x = self.output_controller(x)
        return x
                
    def output_controller(self, x):
        # soft decision
        if self.soft_decision:
            return self.sigmoid(x)

        # stochastic hard decision:
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)

class Router_MLP_h2(nn.Module):
    """  MLP with 2 hidden layer """
    def __init__(self, input_nc,  input_width, input_height,
                 kernel_size=28,
                 soft_decision=True,
                 stochastic=False,
                 reduction_rate=2,
                 **kwargs):
        super(Router_MLP_h2, self).__init__()
        self.soft_decision = soft_decision
        self.stochastic=stochastic

        width = input_nc*input_width*input_height
        self.fc1 = nn.Linear(width, int(width/reduction_rate) + 1)
        self.fc2 = nn.Linear(int(width/reduction_rate) + 1, int(width/(reduction_rate*2)) + 1)
        self.fc3 = nn.Linear(int(width/(reduction_rate*2)) + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 2 fc layers:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        # get probability of "left" or "right"
        x = self.output_controller(x)
        # print("within MLP_h2")
        # print(x.shape)
        return x
                
    def output_controller(self, x):
        
        # print("self.soft_decision",self.soft_decision)
        # print("self.stochastic", self.stochastic)
        # soft decision
        if self.soft_decision:
            return self.sigmoid(x)

        # stochastic hard decision:
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)


class RouterGAP_TwoFClayers(nn.Module):
    """ Routing function:
    GAP + fc1 + fc2 
    """
    def __init__(self, input_nc,  input_width, input_height,
                 kernel_size=28,
                 soft_decision=True,
                 stochastic=False,
                 reduction_rate = 2,
                 dropout_prob=0.0,
                 **kwargs):
        super(RouterGAP_TwoFClayers, self).__init__()
        self.soft_decision = soft_decision
        self.stochastic=stochastic
        self.dropout_prob = dropout_prob
    
        self.fc1 = nn.Linear(input_nc, int(input_nc/reduction_rate) + 1)
        self.fc2 = nn.Linear(int(input_nc/reduction_rate) + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # spatial averaging
        x = x.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
        # 2 fc layers:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x).squeeze()
        # get probability of "left" or "right"
        x = self.output_controller(x)
        return x
                
    def output_controller(self, x):
        # soft decision
        if self.soft_decision:
            return self.sigmoid(x)

        # stochastic hard decision:
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)


class RouterGAPwithConv_TwoFClayers(nn.Module):
    """ Routing function:
    Conv2D + GAP + fc1 + fc2 
    """
    def __init__(self, input_nc,  input_width, input_height,
                 ngf=10,
                 kernel_size=3,
                 soft_decision=True,
                 stochastic=False,
                 reduction_rate = 2,
                 dropout_prob=0.0,
                 **kwargs):
        super(RouterGAPwithConv_TwoFClayers, self).__init__()
        self.ngf = ngf
        self.soft_decision = soft_decision
        self.stochastic=stochastic
        self.dropout_prob = dropout_prob

        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)

        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=kernel_size)
        self.fc1 = nn.Linear(ngf, int(ngf/reduction_rate) + 1)
        self.fc2 = nn.Linear(int(ngf/reduction_rate) + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # convolution:
        x = F.relu(self.conv1(x))
        # spatial averaging
        x = x.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
        # 2 fc layers:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x).squeeze()
        # get probability of "left" or "right"
        x = self.output_controller(x)
        return x
                
    def output_controller(self, x):
        # soft decision
        if self.soft_decision:
            return self.sigmoid(x)

        # stochastic hard decision:
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)


class InnerGAPwithMBv2Conv_TwoFClayers(nn.Module):

    def __init__(self, input_nc, input_width, input_height,
                width_mult = 1.0,stride = 1, ngf = 32, 
                soft_decision=True,
                stochastic=False,
                dropout_prob=0.0,
                expansion_rate = 2, 
                reduction_rate = 2,
                **kwargs):
        
        super(InnerGAPwithMBv2Conv_TwoFClayers, self).__init__()
        block = InvertedResidual
        input_channel = input_nc
        self.ngf = ngf
        self.soft_decision = soft_decision
        self.stochastic=stochastic
        self.dropout_prob = dropout_prob

        interverted_residual_setting = [
            # t, c, n, s
            [expansion_rate, ngf, 1, stride],
        ]

        # print("using MBV2")
        # print("check settings")
        # print(interverted_residual_setting)

        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.fc1 = nn.Linear(ngf, int(ngf/reduction_rate) + 1)
        self.fc2 = nn.Linear(int(ngf/reduction_rate) + 1, 1)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
        # 2 fc layers:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x).squeeze()
        # get probability of "left" or "right"
        x = self.output_controller(x)
        return x

    def output_controller(self, x):
        # soft decision
        if self.soft_decision:
            return self.sigmoid(x)

        # stochastic hard decision:
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# ############################ (3) Solvers ####################################
class LR(nn.Module):
    """ Logistinc regression
    """
    def __init__(self, input_nc, input_width, input_height, no_classes=10, **kwargs):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_nc*input_width*input_height, no_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x))


class MLP_LeNet(nn.Module):
    """ The last fully-connected part of LeNet
    """
    def __init__(self, input_nc, input_width, input_height, no_classes=10, **kwargs):
        super(MLP_LeNet, self).__init__()
        assert input_nc*input_width*input_height > 120
        self.fc1 = nn.Linear(input_nc*input_width*input_height, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, no_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out)


class MLP_LeNetMNIST(nn.Module):
    """ The last fully connected part of LeNet MNIST:
    https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    """
    def __init__(self, input_nc, input_width, input_height, dropout_prob=0.1, **kwargs):
        super(MLP_LeNetMNIST, self).__init__()
        self.dropout_prob = dropout_prob
        ngf = input_nc*input_width*input_height
        self.fc1 = nn.Linear(ngf, int(round(ngf/1.6)))
        self.fc2 = nn.Linear(int(round(ngf/1.6)), 10)
       
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        return F.log_softmax(x)


class Solver_GAP_TwoFClayers(nn.Module):
    """ GAP + fc1 + fc2 """
    def __init__(self, input_nc, input_width, input_height, 
                 dropout_prob=0.1, reduction_rate=2,num_classes = 10, **kwargs):
        super(Solver_GAP_TwoFClayers, self).__init__()
        # print("Solver_GAP_TwoFClayers- num_classes", num_classes)
        self.dropout_prob = dropout_prob
        self.reduction_rate = reduction_rate
        
        self.fc1 = nn.Linear(input_nc, int(input_nc/reduction_rate) + 1)
        self.fc2 = nn.Linear(int(input_nc/reduction_rate ) + 1, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # spatial averaging
        x = x.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
        # 2 fc layers:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x).squeeze()
        return F.log_softmax(x)


class MLP_AlexNet(nn.Module):
    """ The last fully connected part of LeNet MNIST:
    https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    """
    def __init__(self, input_nc, input_width, input_height, dropout_prob=0.1, **kwargs):
        super(MLP_AlexNet, self).__init__()
        self.dropout_prob = dropout_prob
        ngf = input_nc * input_width * input_height
        self.fc1 = nn.Linear(ngf, 128)
        self.fc2 = nn.Linear(128, 10)
       
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        return F.log_softmax(x)


class Solver_GAP_OneFClayers(nn.Module):
    """ GAP + fc1 """
    def __init__(self, input_nc, input_width, input_height, 
                 dropout_prob=0.0, reduction_rate=2, num_classes = 10, **kwargs):
        super(Solver_GAP_OneFClayers, self).__init__()
        self.dropout_prob = dropout_prob
        self.reduction_rate = reduction_rate

        self.fc1 = nn.Linear(input_nc, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # spatial averaging
        x = x.mean(dim=-1).mean(dim=-1).squeeze()  
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc1(x)
        return F.log_softmax(x)


class Child_MBV2GAP(nn.Module):

    def __init__(self, input_nc, input_width, input_height, 
                 num_classes = 10, 
                 width_mult = 1.0,
                 stride = 1, 
                 ngf = 32, 
                 dropout_prob=0.0,
                 reduction_rate = 2,
                 expansion_rate = 2, 
                 **kwargs):
        super(Child_MBV2GAP, self).__init__()
        block = InvertedResidual
        input_channel = input_nc

        interverted_residual_setting = [
            # t, c, n, s
            [expansion_rate, ngf, 1, stride],
        ]
        self.dropout_prob = dropout_prob
        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.fc1 = nn.Linear(ngf, int(ngf/reduction_rate) + 1)
        
        # print("check num_classes in MBv2 solver", num_classes)
        
        self.fc2 = nn.Linear(int(ngf/reduction_rate) + 1, num_classes)
        self._initialize_weights()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=-1).mean(dim=-1).squeeze()  # global average pooling
        # 2 fc layers:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        return F.log_softmax(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
class Child_MBV2(nn.Module):

    def __init__(self, input_nc, input_width, input_height, 
                 num_classes = 10, 
                 width_mult = 1.0,
                 stride = 1, 
                 ngf = 32, 
                 dropout_prob=0.0,
                 reduction_rate = 2,
                 expansion_rate = 2, 
                 **kwargs):
        super(Child_MBV2, self).__init__()
        block = InvertedResidual
        input_channel = input_nc

        interverted_residual_setting = [
            # t, c, n, s
            [expansion_rate, ngf, 1, stride],
        ]
        self.dropout_prob = dropout_prob
        self.features = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.fc_input = output_channel * input_height * input_width // (stride * stride)
        self.fc1 = nn.Linear(self.fc_input, int(self.fc_input/reduction_rate) + 1)
        
        # print("check num_classes in MBv2 solver", num_classes)
        
        self.fc2 = nn.Linear(int(self.fc_input/reduction_rate) + 1, num_classes)
        self._initialize_weights()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        print("check x shape in MBv2 solver", x.shape)
        # 2 fc layers:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        return F.log_softmax(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()