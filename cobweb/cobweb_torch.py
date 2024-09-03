"""
The Cobweb module contains the :class:`CobwebTree` and :class:`CobwebNode`
classes which are used to achieve the basic Cobweb functionality.
"""
import json
import math
from random import shuffle
from random import random
from math import log
from math import isclose
from collections import defaultdict
from collections import Counter
import heapq

import torch


class CobwebTorchTree(object):
    """
    The CobwebTree contains the knoweldge base of a partiucluar instance of the
    cobweb algorithm and can be used to fit and categorize instances.
    """

    def __init__(self, shape, acuity_cutoff=False, hidden_label=True,
                 cu_type="info", covar_type="diag", covar_from="root",
                 prior_var=None, alpha=1e-8, device=None):
        """
        The tree constructor.
        """
        self.device = device
        self.hidden_label = hidden_label
        self.cu_type = cu_type
        self.covar_type = covar_type
        self.covar_from = covar_from
        self.acuity_cutoff = acuity_cutoff
        self.shape = shape
        self.alpha = torch.tensor(alpha, dtype=torch.float, device=self.device,
                                  requires_grad=False)
        self.pi_tensor = torch.tensor(math.pi, dtype=torch.float,
                                      device=self.device, requires_grad=False)

        self.prior_var = prior_var
        if prior_var is None:
            self.prior_var = 1 / (2 * math.e * self.pi_tensor)

        self.clear()

    def clear(self):
        """
        Clears the concepts of the tree.
        """
        self.root = CobwebTorchNode(shape=self.shape,
                                    full_covar=self.covar_type=="full",
                                    device=self.device)
        self.root.tree = self
        self.labels = {}
        self.reverse_labels = {}

    def __str__(self):
        return str(self.root)

    def dump_json(self):
        # only save reverse labels, because regular labels get converted into
        # strings regardless of type and we know the type of the indices.
        tree_params = {
                'cu_type': self.cu_type,
                'covar_type': self.covar_type,
                'covar_from': self.covar_from,
                'hidden_label': self.hidden_label,
                'acuity_cutoff': self.acuity_cutoff,
                'shape': self.shape.tolist() if isinstance(self.shape, torch.Tensor) else self.shape,
                'alpha': self.alpha.item(),
                'prior_var': self.prior_var.item(),
                'reverse_labels': self.reverse_labels}

        json_output = json.dumps(tree_params)[:-1]
        json_output += ', "root": '
        json_output += self.root.iterative_output_json()
        json_output += "}"

        return json_output

    def load_json_helper(self, node_data_json):
        node = CobwebTorchNode(shape=self.shape, 
                               full_covar=self.covar_type=="full",
                               device=self.device)
        node.count = torch.tensor(node_data_json['count'], dtype=torch.float,
                                  device=self.device, requires_grad=False)
        node.mean = torch.tensor(node_data_json['mean'], dtype=torch.float,
                                 device=self.device, requires_grad=False)
        node.meanSq = torch.tensor(node_data_json['meanSq'], dtype=torch.float,
                                   device=self.device, requires_grad=False)
        node.label_counts = torch.tensor(node_data_json['label_counts'],
                                         dtype=torch.float, device=self.device,
                                         requires_grad=False)
        node.total_label_count = node.label_counts.sum()
        return node

    def load_json(self, json_string):
        data = json.loads(json_string)

        self.cu_type = data['cu_type']
        self.covar_type = data['covar_type']
        self.covar_from = data['covar_from']
        self.hidden_label = data['hidden_label']
        self.acuity_cutoff = data['acuity_cutoff']
        self.shape = data['shape']
        self.alpha = torch.tensor(data['alpha'], dtype=torch.float,
                                  device=self.device, requires_grad=False)
        self.prior_var = torch.tensor(data['prior_var'], dtype=torch.float,
                                      device=self.device, requires_grad=False)
        self.reverse_labels = {int(attr): data['reverse_labels'][attr] for attr in data['reverse_labels']}
        self.labels = {self.reverse_labels[attr]: attr for attr in self.reverse_labels}
        self.root = self.load_json_helper(data['root'])
        self.root.tree = self

        queue = [(self.root, c) for c in data['root']['children']]

        while len(queue) > 0:
            parent, curr_data = queue.pop()
            curr = self.load_json_helper(curr_data)
            curr.tree = self
            curr.parent = parent
            parent.children.append(curr)

            for c in curr_data['children']:
                queue.append((curr, c))

    def ifit(self, instance, label=None):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The instance is passed down the cobweb tree and updates each node to
        incorporate the instance. **This process modifies the tree's
        knowledge** for a non-modifying version of labeling use the
        :meth:`CobwebTree.categorize` function.

        :param instance: An instance to be categorized into the tree.
        :type instance:  :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        if label is not None and label not in self.labels:
            idx = len(self.labels)
            self.labels[label] = idx
            self.reverse_labels[idx] = label

        with torch.no_grad():
            return self.cobweb(instance, label)

    def fit(self, instances, labels=None, iterations=1, randomize_first=True):
        """
        Fit a collection of instances into the tree.

        This is a batch version of the ifit function that takes a collection of
        instances and categorizes all of them. The instances can be
        incorporated multiple times to burn in the tree with prior knowledge.
        Each iteration of fitting uses a randomized order but the first pass
        can be done in the original order of the list if desired, this is
        useful for initializing the tree with specific prior experience.

        :param instances: a collection of instances
        :type instances:  [:ref:`Instance<instance-rep>`,
            :ref:`Instance<instance-rep>`, ...]
        :param iterations: number of times the list of instances should be fit.
        :type iterations: int
        :param randomize_first: whether or not the first iteration of fitting
            should be done in a random order or in the list's original order.
        :type randomize_first: bool
        """
        if labels is None:
            labels = [None for i in range(len(instances))]

        instance_labels = [(inst, labels[i]) for i, inst in
                           enumerate(instances)]

        for x in range(iterations):
            if x == 0 and randomize_first:
                shuffle(instances)
            for inst, label in instances:
                self.ifit(inst, label)
            shuffle(instances)

    def cobweb(self, instance, label):
        """
        The core cobweb algorithm used in fitting and categorization.

        In the general case, the cobweb algorithm entertains a number of
        sorting operations for the instance and then commits to the operation
        that maximizes the :meth:`category utility
        <CobwebNode.category_utility>` of the tree at the current node and then
        recurses.

        At each node the alogrithm first calculates the category utility of
        inserting the instance at each of the node's children, keeping the best
        two (see: :meth:`CobwebNode.two_best_children
        <CobwebNode.two_best_children>`), and then calculates the
        category_utility of performing other operations using the best two
        children (see: :meth:`CobwebNode.get_best_operation
        <CobwebNode.get_best_operation>`), commiting to whichever operation
        results in the highest category utility. In the case of ties an
        operation is chosen at random.

        In the base case, i.e. a leaf node, the algorithm checks to see if
        the current leaf is an exact match to the current node. If it is, then
        the instance is inserted and the leaf is returned. Otherwise, a new
        leaf is created.

        .. note:: This function is equivalent to calling
            :meth:`CobwebTree.ifit` but its better to call ifit because it is
            the polymorphic method siganture between the different cobweb
            family algorithms.

        :param instance: an instance to incorporate into the tree
        :type instance: :ref:`Instance<instance-rep>`
        :return: a concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
        """
        current = self.root

        while current:
            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance, label) or
                                         current.count == 0):
                # print("leaf match")
                current.increment_counts(instance, label)
                break

            elif not current.children:
                # print("fringe split")
                new = CobwebTorchNode(shape=self.shape,
                                      full_covar=self.covar_type=="full",
                                      device=self.device,
                                      otherNode=current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                new.increment_counts(instance, label)
                current = new.create_new_child(instance, label)
                break

            else:
                best1_pu, best1, best2 = current.two_best_children(instance,
                                                                   label)
                _, best_action = current.get_best_operation(instance, label, best1,
                                                            best2, best1_pu)

                # print(best_action)
                if best_action == 'best':
                    current.increment_counts(instance, label)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance, label)
                    current = current.create_new_child(instance, label)
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance, label)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current

    def _cobweb_categorize(self, instance, label, use_best, greedy, max_nodes):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        queue = []
        heapq.heappush(queue, (-self.root.log_prob(instance, label), 0.0, random(), self.root))
        nodes_visited = 0

        best = self.root
        best_score = float('-inf')

        while len(queue) > 0:
            neg_score, neg_curr_ll, _, curr = heapq.heappop(queue)
            score = -neg_score # the heap sorts smallest to largest, so we flip the sign
            curr_ll = -neg_curr_ll # the heap sorts smallest to largest, so we flip the sign
            nodes_visited += 1
            curr.update_label_count_size()

            if score > best_score:
                best = curr
                best_score = score

            if use_best and best_score > curr_ll:
                # if best_score is greater than curr_ll, then we know we've
                # found the best and can stop early.
                break

            if greedy:
                queue = []

            if nodes_visited >= max_nodes:
                break

            if len(curr.children) > 0:
                ll_children_unnorm = torch.zeros(len(curr.children))
                for i, c in enumerate(curr.children):
                    log_prob = c.log_prob(instance, label)
                    ll_children_unnorm[i] = (log_prob + math.log(c.count) - math.log(curr.count))
                log_p_of_x = torch.logsumexp(ll_children_unnorm, dim=0)

                for i, c in enumerate(curr.children):
                    child_ll = ll_children_unnorm[i] - log_p_of_x + curr_ll
                    child_ll_inst = c.log_prob(instance, label)
                    score = child_ll + child_ll_inst # p(c|x) * p(x|c)
                    # score = child_ll # p(c|x)
                    heapq.heappush(queue, (-score, -child_ll, random(), c))

        return best if use_best else curr

    def categorize(self, instance, label=None, use_best=True, greedy=False, max_nodes=float('inf')):
        """
        Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the categorization tree according to the
        normal cobweb algorithm except using only the best operator and without
        modifying nodes' probability tables. **This process does not modify the
        tree's knowledge** for a modifying version of labeling use the
        :meth:`CobwebTree.ifit` function

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        with torch.no_grad():
            return self._cobweb_categorize(instance, label, use_best, greedy, max_nodes)

    def old_categorize(self, instance, label):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        current = self.root

        while True:
            if (len(current.children) == 0):
                return current

            parent = current
            current = None
            best_score = None

            for child in parent.children:
                score = child.log_prob_class_given_instance(instance, label)

                if ((current is None) or ((best_score is None) or (score > best_score))):
                    best_score = score
                    current = child

    def predict_probs(self, instance, label=None, greedy=False,
                      max_nodes=float('inf')):
        with torch.no_grad():
            return self._predict_probs(instance, label, greedy, max_nodes)

    def _predict_probs(self, instance, label, greedy, max_nodes):
        queue = []
        heapq.heappush(queue, (-self.root.log_prob(instance, label), 0.0, random(), self.root))
        nodes_visited = 0

        log_weighted_scores = []
        # total_w = 0

        while len(queue) > 0:
            neg_score, neg_curr_ll, _, curr = heapq.heappop(queue)
            score = -neg_score # the heap sorts smallest to largest, so we flip the sign
            curr_ll = -neg_curr_ll # the heap sorts smallest to largest, so we flip the sign
            nodes_visited += 1

            # w = math.exp(score)
            # total_w += w

            curr.update_label_count_size()
            log_weighted_scores.append(score + torch.log(curr.label_counts) -
                                       torch.log(curr.label_counts.sum()))

            # p_label = {self.reverse_labels[i]: v.item()
            #            for i, v in enumerate(p_label)}

            # for label in p_label:
            #     pred[label].append(w + p_label[label])

            if greedy:
                queue = []

            if nodes_visited >= max_nodes:
                break

            if len(curr.children) > 0:
                ll_children_unnorm = torch.zeros(len(curr.children))
                for i, c in enumerate(curr.children):
                    log_prob = c.log_prob(instance, label)
                    ll_children_unnorm[i] = (log_prob + math.log(c.count) - math.log(curr.count))
                log_p_of_x = torch.logsumexp(ll_children_unnorm, dim=0)

                for i, c in enumerate(curr.children):
                    child_ll = ll_children_unnorm[i] - log_p_of_x + curr_ll
                    child_ll_inst = c.log_prob(instance, label)
                    score = child_ll + child_ll_inst
                    heapq.heappush(queue, (-score, -child_ll, random(), c))

        log_weighted_scores = torch.stack(log_weighted_scores)
        ll = torch.logsumexp(log_weighted_scores, 0) - torch.logsumexp(log_weighted_scores.flatten(), 0)
        pred = {self.reverse_labels[i]: v.item() for i, v in enumerate(torch.exp(ll))}

        # for label in pred:
        #     pred[label] /= total_w

        return pred

    def compute_var(self, meanSq, count):
        # return torch.ones_like(meanSq)
        if self.covar_type == "full":
            return meanSq / count + torch.diag(torch.ones(meanSq.shape[0]) * self.prior_var)
        elif self.acuity_cutoff:
            return torch.clamp(meanSq / count, self.prior_var) # with cutoff
        else:
            return meanSq / count + self.prior_var # with adjustment

    def compute_score(self, mu1, var1, p_label1, mu2, var2, p_label2):

        if self.cu_type == "info":
            if self.covar_type == "full":
                if self.covar_from == "root":
                    var_r = self.root.var
                    L = torch.linalg.cholesky(var_r)
                    y = torch.cholesky_solve((mu1 - mu2).reshape(1, -1).T, L)
                    score = torch.linalg.vector_norm(y)
                elif self.covar_from == "parent":
                    L = torch.linalg.cholesky(var2)
                    y = torch.cholesky_solve((mu1 - mu2).reshape(1, -1).T, L)
                    score = torch.linalg.vector_norm(y)
                else:
                    raise NotImplemented("full self covar is not implemented")

            elif self.covar_type == "diag":
                if self.covar_from == "root":
                    var_r = self.root.var
                    score = (torch.pow(mu1 - mu2, 2)/var_r).sum()
                    score /= 2
                elif self.covar_from == "parent":
                    score = (torch.pow(mu1 - mu2, 2)/var2).sum()
                    score /= 2
                else:
                    score = 0.5 * (torch.log(var2) - torch.log(var1)).sum()

            elif self.covar_type == "const":
                if self.covar_from == "root":
                    var_r = self.root.var
                    score = (torch.pow(mu1 - mu2, 2)/var_r.mean()).sum()
                    score /= 2
                elif self.covar_from == "parent":
                    score = (torch.pow(mu1 - mu2, 2)/var2.mean()).sum()
                    score /= 2
                else:
                    score = 0.5 * (torch.log(var2.mean()) - torch.log(var1.mean()))
                    # raise NotImplemented("constant self covar is not implemented")

            if not self.hidden_label and p_label1 is not None:
                score += (p_label1 * (torch.log(p_label1) -
                                      torch.log(p_label2))).sum()

        elif self.cu_type == "p_sum":
            child_score = (1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var1))).sum()
            parent_score = (1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var2))).sum()

            if not self.hidden_label and p_label1 is not None:
                child_score += (p_label1 * p_label1).sum()
                parent_score += (p_label2 * p_label2).sum()

            score = child_score - parent_score

        # OLD
        # if (self.use_info):
        #     if (self.use_kl):
        #         score = (torch.log(var2) - torch.log(var1)).sum()
        #         score += ((var1 + torch.pow(mu1 - mu2, 2))/(var2)).sum()
        #         score -= mu1.numel()
        #         score /= 2

        #     else:
        #         score = 0.5 * (torch.log(var2) - torch.log(var1)).sum()
        # else:
        #     score = -(1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var1))).sum()
        #     score += (1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var2))).sum()

        # if p_label1 is not None:
        #     if (self.use_info):
        #         if (self.use_kl):
        #             score += (-p_label1 * torch.log(p_label1)).sum()
        #             score -= (-p_label1 * torch.log(p_label2)).sum()
        #         else:
        #             score += (-p_label1 * torch.log(p_label1)).sum()
        #             score -= (-p_label2 * torch.log(p_label2)).sum()
        #     else:
        #         score += (-p_label1 * p_label1).sum()
        #         score -= (-p_label2 * p_label2).sum()

        return score


class CobwebTorchNode(object):
    """
    A CobwebNode represents a concept within the knoweldge base of a particular
    :class:`CobwebTree`. Each node contains a probability table that can be
    used to calculate the probability of different attributes given the concept
    that the node represents.

    In general the :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
    functions should be used to initially interface with the Cobweb knowledge
    base and then the returned concept can be used to calculate probabilities
    of certain attributes or determine concept labels.

    This constructor creates a CobwebNode with default values. It can also be
    used as a copy constructor to "deepcopy" a node, including all references
    to other parts of the original node's CobwebTree.

    :param otherNode: Another concept node to deepcopy.
    :type otherNode: CobwebNode
    """
    # a counter used to generate unique concept names.
    _counter = 0

    def __init__(self, shape, full_covar=False, device=None, otherNode=None):
        """Create a new CobwebNode"""
        self.concept_id = self.gensym()
        self.count = torch.tensor(0.0, dtype=torch.float, device=device,
                                  requires_grad=False)
        self.mean = torch.zeros(shape, dtype=torch.float, device=device,
                                requires_grad=False)
        if full_covar:
            self.meanSq = torch.zeros((shape.numel(), shape.numel()),
                                      dtype=torch.float, device=device,
                                      requires_grad=False)
        else:
            self.meanSq = torch.zeros(shape, dtype=torch.float, device=device,
                                      requires_grad=False)

        self.label_counts = torch.tensor([], dtype=torch.float, device=device,
                                         requires_grad=False)
        self.total_label_count = torch.tensor(0, dtype=torch.float,
                                              device=device,
                                              requires_grad=False)
        self.children = []
        self.parent = None
        self.tree = None

        if otherNode:
            self.tree = otherNode.tree
            self.parent = otherNode.parent
            self.update_counts_from_node(otherNode)

            for child in otherNode.children:
                self.children.append(CobwebTorchNode(shape=self.tree.shape, full_covar=full_covar, device=self.tree.device, otherNode=child))

    def update_label_count_size(self):
        if self.label_counts.shape[0] < len(self.tree.labels):
            num_new = len(self.tree.labels) - self.label_counts.shape[0]
            new_counts = (self.tree.alpha + torch.zeros(num_new,
                                                        dtype=torch.float,
                                                        device=self.tree.device))
            self.label_counts = torch.cat((self.label_counts, new_counts))
            self.total_label_count += new_counts.sum()

    def increment_counts(self, instance, label):
        """
        Increment the counts at the current node according to the specified
        instance.

        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`
        """
        self.count += 1
        delta = instance - self.mean
        self.mean += delta / self.count

        if self.tree.covar_type == "full":
            delta = delta.flatten()
            self.meanSq += (self.count - 1) /self.count * torch.outer(delta, delta)
        else:
            # self.meanSq += delta * (instance - self.mean)
            self.meanSq += (self.count - 1) / self.count * delta * delta

        self.update_label_count_size()

        if label is not None:
            self.label_counts[self.tree.labels[label]] += 1
            self.total_label_count += 1

    def update_counts_from_node(self, other):
        """
        Increments the counts of the current node by the amount in the
        specified node.

        This function is used as part of copying nodes and in merging nodes.

        :param node: Another node from the same CobwebTree
        :type node: CobwebNode
        """
        delta = other.mean - self.mean
        if self.tree.covar_type == "full":
            delta = delta.flatten()
            self.meanSq = self.meanSq + other.meanSq + torch.outer(delta, delta) * ((self.count * other.count) / (self.count + other.count))
        else:
            self.meanSq = (self.meanSq + other.meanSq + delta * delta *
                           ((self.count * other.count) / (self.count + other.count)))
        self.mean = ((self.count * self.mean + other.count * other.mean) /
                     (self.count + other.count))
        self.count += other.count

        self.update_label_count_size()
        other.update_label_count_size()

        # alpha is added to all the counts, so need to subtract from other to
        # not double count
        self.label_counts += (other.label_counts - self.tree.alpha)
        self.total_label_count += (other.total_label_count -
                                   other.label_counts.shape[0] * self.tree.alpha)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def var(self):
        return self.tree.compute_var(self.meanSq, self.count)

    def log_prob_class_given_instance(self, instance, label):
        log_prob = self.log_prob(instance, label)
        log_prob += torch.log(self.count) - torch.log(self.tree.root.count)
        return log_prob

    def log_prob(self, instance, label):

        log_prob = 0

        var = self.var
        if self.tree.covar_from == "root" and self.tree.root:
            var = self.tree.root.var
        elif self.tree.covar_from == "parent" and self.parent:
            var = self.parent.var

        if self.tree.covar_type == "full":
            mvn_dist = torch.distributions.MultivariateNormal(self.mean.flatten(), var)
            log_prob += mvn_dist.log_prob(instance.flatten())
        if self.tree.covar_type == "diag":
            # n_dist = torch.distributions.Normal(self.mean, torch.sqrt(var))
            # log_prob += n_dist.log_prob(instance).sum()
            log_prob += -(0.5 * torch.log(var) + 0.5 * torch.log(2 * self.tree.pi_tensor) +
                          0.5 * torch.square(instance - self.mean) / var).sum()
        elif self.tree.covar_type == "const":
            var = torch.ones_like(self.mean) * var.mean()
            # n_dist = torch.distributions.Normal(self.mean, torch.sqrt(var))
            # log_prob = n_dist.log_prob(instance).sum()
            log_prob += -(0.5 * torch.log(var) + 0.5 * torch.log(2 * self.tree.pi_tensor) +
                          0.5 * torch.square(instance - self.mean) / var).sum()

        self.update_label_count_size()
        if label is not None:
            if label not in self.tree.labels:
                log_prob += (torch.log(self.tree.alpha) -
                             torch.log(self.total_label_count +
                                       self.tree.alpha))
            elif self.total_label_count > 0:
                log_prob += (torch.log(self.label_counts[self.tree.labels[label]]) -
                             torch.log(self.total_label_count))

        return log_prob

    def get_basic(self):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_cu = self.category_utility()

        while curr.parent:
            curr = curr.parent
            curr_cu = curr.category_utility()
            if curr_cu > best_cu:
                best = curr
                best_cu = curr_cu

        return best

    def get_best(self, instance, label=None):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_ll = self.log_prob_class_given_instance(instance, label)

        while curr.parent:
            curr = curr.parent
            curr_ll = curr.log_prob_class_given_instance(instance, label)
            if curr_ll > best_ll:
                best = curr
                best_ll = curr_ll

        return best

    def category_utility(self):
        p_of_c = self.count / self.tree.root.count
        root_mean, root_var, root_p_label = self.tree.root.mean_var_plabel()
        curr_mean, curr_var, curr_p_label = self.mean_var_plabel()

        return p_of_c * self.tree.compute_score(curr_mean, curr_var,
                                                curr_p_label, root_mean,
                                                root_var, root_p_label)

    def mean_var_plabel_new(self, instance, label):
        label_counts = (self.tree.alpha + torch.zeros(len(self.tree.labels),
                                                      dtype=torch.float,
                                                      device=self.tree.device))
        total_label_count = self.tree.alpha * len(self.tree.labels)

        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        if total_label_count > 0:
            p_label = label_counts / total_label_count
        else:
            p_label = None

        var = torch.zeros(self.tree.shape, dtype=torch.float,
                          device=self.tree.device)
        var += self.tree.prior_var

        return instance, var, p_label

    def mean_var_plabel(self):
        self.update_label_count_size()
        if self.total_label_count > 0:
            p_label = self.label_counts / self.total_label_count
        else:
            p_label = None

        return self.mean, self.var, p_label

    def mean_var_plabel_insert(self, instance, label):
        count = self.count + 1
        delta = instance - self.mean
        mean = self.mean + delta / count

        if self.tree.covar_type == "full":
            delta = delta.flatten()
            meanSq = self.meanSq + (count - 1) / count * torch.outer(delta, delta)
        else:
            # meanSq = self.meanSq + delta * (instance - mean)
            meanSq = self.meanSq + (count - 1) / count * delta * delta

        # hopefully cheap if already updated.
        self.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()
        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        var = self.tree.compute_var(meanSq, count)

        if total_label_count > 0:
            p_label = label_counts / total_label_count
        else:
            p_label = None

        return mean, var, p_label

    def mean_var_plabel_merge(self, other, instance, label):
        delta = other.mean - self.mean

        if self.tree.covar_type == "full":
            delta = delta.flatten()
            meanSq = self.meanSq + other.meanSq + torch.outer(delta, delta) * ((self.count * other.count) / (self.count + other.count))
        else:
            meanSq = (self.meanSq + other.meanSq + delta * delta *
                           ((self.count * other.count) / (self.count + other.count)))

        mean = ((self.count * self.mean + other.count * other.mean) /
                (self.count + other.count))
        count = self.count + other.count

        count = count + 1
        delta = instance - mean
        mean += delta / count

        if self.tree.covar_type == "full":
            delta = delta.flatten()
            meanSq += (count - 1) / count * torch.outer(delta, delta)
        else:
            meanSq += (count - 1) / count * delta * delta
            # meanSq += delta * (instance - mean)

        # hopefully cheap if already updated.
        self.update_label_count_size()
        other.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()

        label_counts += (other.label_counts - self.tree.alpha)
        total_label_count += (other.total_label_count -
                              other.label_counts.shape[0] * self.tree.alpha)

        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        var = self.tree.compute_var(meanSq, count)

        if total_label_count > 0:
            p_label = label_counts / total_label_count
        else:
            p_label = None

        return mean, var, p_label

    def partition_utility(self):
        """
        Return the category utility of a particular division of a concept into
        its children.

        Category utility is always calculated in reference to a parent node and
        its own children. This is used as the heuristic to guide the concept
        formation process. Category Utility is calculated as:

        .. math::

            CU(\\{C_1, C_2, \\cdots, C_n\\}) = \\frac{1}{n} \\sum_{k=1}^n
            P(C_k) \\left[ \\sum_i \\sum_j P(A_i = V_{ij} | C_k)^2 \\right] -
            \\sum_i \\sum_j P(A_i = V_{ij})^2

        where :math:`n` is the numer of children concepts to the current node,
        :math:`P(C_k)` is the probability of a concept given the current node,
        :math:`P(A_i = V_{ij} | C_k)` is the probability of a particular
        attribute value given the concept :math:`C_k`, and :math:`P(A_i =
        V_{ij})` is the probability of a particular attribute value given the
        current node.

        In general this is used as an internal function of the cobweb algorithm
        but there may be times when it would be useful to call outside of the
        algorithm itself.

        :return: The category utility of the current node with respect to its
                 children.
        :rtype: float
        """
        if len(self.children) == 0:
            return 0.0

        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel()

        for child in self.children:
            p_of_child = child.count / self.count
            child_mean, child_var, child_p_label = child.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        return score / len(self.children)

    def get_best_operation(self, instance, label, best1, best2, best1_pu):
        """
        Given an instance, the two best children based on category utility and
        a set of possible operations, find the operation that produces the
        highest category utility, and then return the category utility and name
        for the best operation. In the case of ties, an operator is randomly
        chosen.

        Given the following starting tree the results of the 4 standard Cobweb
        operations are shown below:

        .. image:: images/Original.png
            :width: 200px
            :align: center

        * **Best** - Categorize the instance to child with the best category
          utility. This results in a recurisve call to :meth:`cobweb
          <concept_formation.cobweb.CobwebTree.cobweb>`.

            .. image:: images/Best.png
                :width: 200px
                :align: center

        * **New** - Create a new child node to the current node and add the
          instance there. See: :meth:`create_new_child
          <concept_formation.cobweb.CobwebNode.create_new_child>`.

            .. image:: images/New.png
                :width: 200px
                :align: center

        * **Merge** - Take the two best children, create a new node as their
          mutual parent and add the instance there. See: :meth:`merge
          <concept_formation.cobweb.CobwebNode.merge>`.

            .. image:: images/Merge.png
                    :width: 200px
                    :align: center

        * **Split** - Take the best node and promote its children to be
          children of the current node and recurse on the current node. See:
          :meth:`split <concept_formation.cobweb.CobwebNode.split>`

            .. image:: images/Split.png
                :width: 200px
                :align: center

        Each operation is entertained and the resultant category utility is
        used to pick which operation to perform. The list of operations to
        entertain can be controlled with the possible_ops parameter. For
        example, when performing categorization without modifying knoweldge
        only the best and new operators are used.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :param best1: A tuple containing the relative cu of the best child and
            the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best1: (float, CobwebNode)
        :param best2: A tuple containing the relative cu of the second best
            child and the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best2: (float, CobwebNode)
        :param possible_ops: A list of operations from ["best", "new", "merge",
            "split"] to entertain.
        :type possible_ops: ["best", "new", "merge", "split"]
        :return: A tuple of the category utility of the best operation and the
            name of the best operation.
        :rtype: (cu_bestOp, name_bestOp)
        """
        if not best1:
            raise ValueError("Need at least one best child.")

        operations = []

        operations.append((best1_pu, random(), "best"))
        operations.append((self.pu_for_new_child(instance, label), random(), 'new'))
        if len(self.children) > 2 and best2:
            operations.append((self.pu_for_merge(best1, best2, instance, label),
                               random(), 'merge'))
        if len(best1.children) > 0:
            operations.append((self.pu_for_split(best1), random(), 'split'))

        operations.sort(reverse=True)
        # print(operations)
        best_op = (operations[0][0], operations[0][2])
        # print(best_op)
        return best_op

    def two_best_children(self, instance, label):
        """
        Calculates the category utility of inserting the instance into each of
        this node's children and returns the best two. In the event of ties
        children are sorted first by category utility, then by their size, then
        by a random value.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility and indices for the two best children
            (the second tuple will be ``None`` if there is only 1 child).
        :rtype: ((cu_best1,index_best1),(cu_best2,index_best2))
        """
        if len(self.children) == 0:
            raise Exception("No children!")

        relative_pus = []
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for child in self.children:
            p_of_c = (child.count + 1) / (self.count + 1)
            mean, var, p_label = child.mean_var_plabel_insert(instance, label)
            score_gain = p_of_c * self.tree.compute_score(mean, var, p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

            p_of_c = child.count / (self.count + 1)
            mean, var, p_label = child.mean_var_plabel()
            score_gain -= p_of_c * self.tree.compute_score(mean, var, p_label,
                                                           parent_mean,
                                                           parent_var,
                                                           parent_p_label)

            relative_pus.append((score_gain, child.count, random(), child))

        relative_pus.sort(reverse=True)

        best1 = relative_pus[0][3]
        best1_pu = self.pu_for_insert(best1, instance, label)

        best2 = None
        if len(relative_pus) > 1:
            best2 = relative_pus[1][3]

        return best1_pu, best1, best2

    def pu_for_insert(self, child, instance, label):
        """
        Compute the category utility of adding the instance to the specified
        child.

        This operation does not actually insert the instance into the child it
        only calculates what the result of the insertion would be. For the
        actual insertion function see: :meth:`CobwebNode.increment_counts` This
        is the function used to determine the best children for each of the
        other operations.

        :param child: a child of the current node
        :type child: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to the given node
        :rtype: float

        .. seealso:: :meth:`CobwebNode.two_best_children` and
            :meth:`CobwebNode.get_best_operation`

        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for c in self.children:
            if c == child:
                p_of_child = (c.count + 1) / (self.count + 1)
                child_mean, child_var, child_p_label = c.mean_var_plabel_insert(instance, label)
            else:
                p_of_child = (c.count) / (self.count + 1)
                child_mean, child_var, child_p_label = c.mean_var_plabel()

            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        return score / len(self.children)

    def create_new_child(self, instance, label):
        """
        Create a new child (to the current node) with the counts initialized by
        the *given instance*.

        This is the operation used for creating a new child to a node and
        adding the instance to it.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The new child
        :rtype: CobwebNode
        """
        new_child = CobwebTorchNode(shape=self.tree.shape,
                                    full_covar=self.tree.covar_type=="full",
                                    device=self.tree.device)
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance, label)
        self.children.append(new_child)
        return new_child

    def pu_for_new_child(self, instance, label):
        """
        Return the category utility for creating a new child using the
        particular instance.

        This operation does not actually create the child it only calculates
        what the result of creating it would be. For the actual new function
        see: :meth:`CobwebNode.create_new_child`.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to a new child.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for c in self.children:
            p_of_child = c.count / (self.count + 1)
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        # score for new
        p_of_child = 1.0 / (self.count + 1)
        child_mean, child_var, child_p_label = c.mean_var_plabel_new(instance, label)
        score += p_of_child * self.tree.compute_score(child_mean, child_var,
                                                      child_p_label,
                                                      parent_mean, parent_var,
                                                      parent_p_label)

        return score / (len(self.children) + 1)

    def merge(self, best1, best2):
        """
        Merge the two specified nodes.

        A merge operation introduces a new node to be the merger of the the two
        given nodes. This new node becomes a child of the current node and the
        two given nodes become children of the new node.

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :return: The new child node that was created by the merge
        :rtype: CobwebNode
        """
        new_child = CobwebTorchNode(shape=self.tree.shape,
                                    full_covar=self.tree.covar_type=="full",
                                    device=self.tree.device)
        new_child.parent = self
        new_child.tree = self.tree

        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        best1.parent = new_child
        best2.parent = new_child
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def pu_for_merge(self, best1, best2, instance, label):
        """
        Return the category utility for merging the two best children.

        This does not actually merge the two children it only calculates what
        the result of the merge would be. For the actual merge operation see:
        :meth:`CobwebNode.merge`

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The category utility that would result from merging best1 and
            best2.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for c in self.children:
            if c == best1 or c == best2:
                continue

            p_of_child = c.count / (self.count + 1)
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        p_of_child = (best1.count + best2.count + 1) / (self.count + 1)
        child_mean, child_var, child_p_label = best1.mean_var_plabel_merge(best2, instance, label)
        score += p_of_child * self.tree.compute_score(child_mean, child_var,
                                                      child_p_label,
                                                      parent_mean, parent_var,
                                                      parent_p_label)

        return score / (len(self.children) - 1)


    def split(self, best):
        """
        Split the best node and promote its children

        A split operation removes a child node and promotes its children to be
        children of the current node. Split operations result in a recursive
        call of cobweb on the current node so this function does not return
        anything.

        :param best: The child node to be split
        :type best: CobwebNode
        """
        self.children.remove(best)
        for child in best.children:
            child.parent = self
            child.tree = self.tree
            self.children.append(child)

    def pu_for_split(self, best):
        """
        Return the category utility for splitting the best child.

        This does not actually split the child it only calculates what the
        result of the split would be. For the actual split operation see:
        :meth:`CobwebNode.split`. Unlike the category utility calculations for
        the other operations split does not need the instance because splits
        trigger a recursive call on the current node.

        :param best: The child of the current node with the best category
            utility
        :type best: CobwebNode
        :return: The category utility that would result from splitting best
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel()

        for c in self.children:
            if c == best:
                continue
            p_of_child = c.count / self.count
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        for c in best.children:
            p_of_child = c.count / self.count
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        return (score / (len(self.children) - 1 + len(best.children)))

    def is_exact_match(self, instance, label):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        self.update_label_count_size()

        if label is not None and self.total_label_count == 0:
            return False

        if label is None and self.total_label_count > 0:
            return False

        if self.total_label_count > 0:
            label_counts = self.label_counts - self.tree.alpha
            p_labels = label_counts / label_counts.sum()

            if not math.isclose(p_labels[self.tree.labels[label]].item(), 1.0):
                return False

        if not torch.isclose(self.meanSq, torch.zeros_like(self.meanSq, device=self.tree.device)).all():
            return False

        return torch.isclose(instance, self.mean).all()

    def __hash__(self):
        """
        The basic hash function. This hashes the concept name, which is
        generated to be unique across concepts.
        """
        return hash("CobwebNode" + str(self.concept_id))

    def gensym(self):
        """
        Generate a unique id and increment the class _counter.

        This is used to create a unique name for every concept. As long as the
        class _counter variable is never externally altered these keys will
        remain unique.

        """
        self.__class__._counter += 1
        return self.__class__._counter

    def __str__(self):

        """
        Call :meth:`CobwebNode.pretty_print`
        """
        return self.pretty_print()

    def pretty_print(self, depth=0):
        """
        Print the categorization tree

        The string formatting inserts tab characters to align child nodes of
        the same depth.

        :param depth: The current depth in the print, intended to be called
                      recursively
        :type depth: int
        :return: a formated string displaying the tree and its children
        :rtype: str
        """
        ret = str(('\t' * depth) + "|-" + str(self.mean) + ":" +
                  str(self.count) + '\n')

        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def depth(self):
        """
        Returns the depth of the current node in its tree

        :return: the depth of the current node in its tree
        :rtype: int
        """
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def is_parent(self, other_concept):
        """
        Return True if this concept is a parent of other_concept

        :return: ``True`` if this concept is a parent of other_concept else
                 ``False``
        :rtype: bool
        """
        temp = other_concept
        while temp is not None:
            if temp == self:
                return True
            try:
                temp = temp.parent
            except Exception:
                print(temp)
                assert False
        return False

    def num_concepts(self):
        """
        Return the number of concepts contained below the current node in the
        tree.

        When called on the :attr:`CobwebTree.root` this is the number of nodes
        in the whole tree.

        :return: the number of concepts below this concept.
        :rtype: int
        """
        children_count = 0
        for c in self.children:
            children_count += c.num_concepts()
        return 1 + children_count

    def iterative_output_json_helper(self):
        self.update_label_count_size()
        output = {}
        output['count'] = self.count.item()
        output['mean'] = self.mean.tolist()
        output['meanSq'] = self.meanSq.tolist()
        output['label_counts'] = self.label_counts.tolist()
        return json.dumps(output)

    def iterative_output_json(self):
        output = ""

        visited = set()
        curr = self

        while curr is not None:
            if curr.concept_id not in visited:
                node_output = curr.iterative_output_json_helper()
                if len(output) > 1 and output[-1] == "}":
                    output += ", "
                output += node_output[:-1]
                output += ', "children": ['
                visited.add(curr.concept_id)

            for child in curr.children:
                if child.concept_id not in visited:
                    curr = child
                    break
            else:
                curr = curr.parent
                output += "]}"

        return output

    # TODO remove and just use above output, left for legacy use with viz.
    def output_json(self):
        return json.dumps(self.output_dict())

    def visualize(self):
        from matplotlib import pyplot as plt
        plt.imshow(self.mean.numpy())
        plt.show()

    def output_dict(self):
        """
        Outputs the categorization tree in JSON form

        :return: an object that contains all of the structural information of
                 the node and its children
        :rtype: obj
        """
        self.update_label_count_size()

        output = {}
        output['name'] = "Concept" + str(self.concept_id)
        output['size'] = self.count.item()
        output['children'] = []

        temp = {}
        temp['_category_utility'] = {"#ContinuousValue#": {'mean': self.category_utility().item(), 'std': 1, 'n': 1}}

        if self.total_label_count > 0:
            temp['label'] = {label: self.label_counts[self.tree.labels[label]].item() for label in self.tree.labels}

        for child in self.children:
            output["children"].append(child.output_dict())

        output['counts'] = temp
        output['mean'] = self.mean.tolist()
        # output['meanSq'] = self.meanSq.tolist()

        return output

    def predict(self, most_likely=True):
        """
        Predict the value of an attribute, using the specified choice function
        (either the "most likely" value or a "sampled" value).

        :param attr: an attribute of an instance.
        :type attr: :ref:`Attribute<attributes>`
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled".
        :type choice_fn: a string
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: The most likely value for the given attribute in the node's
                 probability table.
        :rtype: :ref:`Value<values>`
        """
        self.update_label_count_size()

        if most_likely:
            label = None
            if self.total_label_count > 0:
                label = self.tree.reverse_labels[self.label_counts.argmax().item()]
            return self.mean.detach().clone(), label
        else:
            label = None
            if self.total_label_count > 0:
                p_labels = label_counts / label_counts.sum()
                label = self.tree.reverse_labels[torch.multinomial(p_labels, 1).item()]
            return torch.normal(self.mean, self.std), label
