import numpy as np
from itertools import groupby
from operator import itemgetter

class Node:
    def __init__(self, name):
        self.__name = name
        self.__parents = []
        self.__children = []
        self.__domain = None
        self.__maps = {}
    @property
    def name(self):
        return self.__name
    @name.setter
    def name(self, name):
        self.__name = name
    def addParent(self, parents):
        if not isinstance(parents, (list, tuple)):
            parents = [parents]
        self.__parents.extend(parents)
    def addChild(self, child):
        self.__children.append(child)
    @property
    def parents(self):
        return self.__parents
    @parents.setter
    def parents(self, parents):
        self.__parents = parents
    @property
    def children(self):
        return self.__children
    @children.setter
    def children(self, children):
        self.__children = children
    @property
    def domain(self):
        return self.__domain
    @domain.setter
    def domain(self, domain):
        self.__domain = domain
        self.__maps = dict(zip(domain, range(len(domain))))
    @property
    def maps(self):
        return self.__maps

class Factor:
    def __init__(self, node):
        self.ptable = node.ptable
        self.scopes = np.array(node.parents + [node.name])
    def reduce(self, evidences):
        same_evidences = list(filter(lambda x: x in evidences, self.scopes))
        for evidence in same_evidences:
            idx = np.flatnonzero(self.scopes == evidence)[0]
            self.ptable = self.ptable[np.flatnonzero(self.ptable[:, idx] == evidences[evidence])]

    def multiply(self, factor):
        common_scopes = np.array(list(set(self.scopes) & set(factor.scopes)))
        common_scopes_size = len(common_scopes)
        self.sort_evidence(common_scopes)
        factor.sort_evidence(common_scopes)

        all_nodes = np.append(self.scopes, factor.scopes[common_scopes_size:])
        sizes = {}
        for i, node in enumerate(self.scopes):
            sizes[node] = np.unique(self.ptable[:, i]).shape[0]
        for i, node in enumerate(factor.scopes[common_scopes_size: ]):
            sizes[node] = np.unique(factor.ptable[:, i + common_scopes_size]).shape[0]
        size = np.prod(list(sizes.values()))
        new_ptable = []

        for f_prob in self.ptable:
            for s_prob in factor.ptable:
                if (f_prob[:common_scopes_size] == s_prob[:common_scopes_size]).all():
                    new_row = np.hstack((f_prob[: - 1], s_prob[common_scopes_size : -1], np.array([f_prob[-1] * s_prob[-1]])))
                    new_ptable.append(new_row)
        self.scopes = all_nodes
        self.ptable = np.vstack((new_ptable))
        assert self.ptable.shape[0] == size, "[error]: wrong len"
        assert self.ptable.shape[1] == len(all_nodes) + 1, "[error]: wrong number of columns"
        return self.ptable

    def margin_prob(self, remove_nodes):
        if not isinstance(remove_nodes, (tuple, list)):
            remove_nodes = (remove_nodes, )
        assert set(remove_nodes) <= set(self.scopes), "[error]: exist a remove node out of scope in factor"
        for node in remove_nodes:
            idx = np.flatnonzero(self.scopes == node)[0]
            self.ptable = np.delete(self.ptable, idx, axis = 1)
        self.scopes = np.array(list(filter(lambda x: x not in remove_nodes, self.scopes)))
        values, indices = np.unique(self.ptable[:, : -1], axis = 0, return_index = True)
        new_ptable = np.hstack((values, np.zeros((values.shape[0], 1))))
        for row in self.ptable:
            new_ptable[np.flatnonzero(np.all(values == row[:-1], axis = 1))[0]][-1] += row[-1]
        del self.ptable
        self.ptable = new_ptable
        return self.ptable


    def sort_evidence(self, nodes):
        for i, node in enumerate(nodes):
            idx = np.flatnonzero(self.scopes == node)[0]
            if i != idx:
                self.scopes[[i, idx]] = self.scopes[[idx, i]]
                self.ptable[:, [i, idx]] = self.ptable[:, [idx, i]]
        ptable = self.ptable.tolist()
        ptable.sort(key = lambda x: x[: -1])
        self.ptable = np.array(ptable)
    def get_prob_at_row(self, variables):
        prob = self.ptable.copy()
        for i, scope in enumerate(self.scopes):
            value = variables[scope]
            prob = prob[prob[:, i] == value]
        return prob[0][-1]


    
class BayesianNetwork:
    def __init__(self, filename):
        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()
        self.graph = {}
        self.orders = []
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)
            # YOUR CODE HERE
            name = node
            node = Node(name)
            self.orders.append(name)

            node.addParent(parents)
            for parent in parents:
                self.graph[parent].addChild(name)
            node.domain = np.array(domain)

            if not isinstance(shape, (list, tuple)):
                shape = (shape,)

            size = np.prod(shape)
            tmp = []
            for i, dim in enumerate(shape):
                t = []
                n = size // (dim * (2 ** i))
                s = size // n
                for j in range(s):
                    t.extend([j % dim] * n)
                tmp.append(np.array(t))
            tmp.append(np.array(probabilities).ravel())
            node.ptable = np.column_stack(tmp)
            self.graph[name] = node

        f.close()
    
    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        elimination_variables = list(filter(lambda x: x not in query_variables and x not in evidence_variables, self.orders))
        for evidence, value in evidence_variables.items():
            evidence_variables[evidence] = self.graph[evidence].maps[value]
        for query, value in query_variables.items():
            query_variables[query] = self.graph[query].maps[value]
        all_factors = set()
        for name, node in self.graph.items():
            factor = Factor(node)
            factor.reduce(evidence_variables)
            all_factors.add(factor)

        for elimination in elimination_variables:
            factor_list = [factor for factor in all_factors if elimination in factor.scopes]
            m_factor = None
            for i, factor in enumerate(factor_list):
                if i == 0:
                    m_factor = factor
                else:
                    m_factor.multiply(factor)
            m_factor.margin_prob(elimination)
            all_factors = (all_factors - set(factor_list))| set([m_factor])


        for i, factor in enumerate(all_factors):
            if i == 0:
                final_factor = factor
            else:
               final_factor.multiply(factor)
        sum_prob = final_factor.ptable[:, -1].sum()
        prob = final_factor.get_prob_at_row({**evidence_variables, **query_variables})
        result = prob / sum_prob
        
                

        f.close()
        return result

    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        # YOUR CODE HERE


        f.close()
        return result

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        probabilities = np.array(eval(parts[4])).reshape(shape)
        return node, parents, domain, shape, probabilities

    def __extract_query(self, line):
        parts = line.split(';')

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            query_variables[lst[0]] = lst[1]

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            evidence_variables[lst[0]] = lst[1]
        return query_variables, evidence_variables
