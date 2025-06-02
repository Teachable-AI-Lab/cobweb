import copy

"""
A partition/clustering is derived by generating the 'best' partition 
based on the memberships of the instances among the children of the root (i.e. the concepts at the 2nd level).

We generate this partition by multiple simulations.
Given a number of splits, nth_split, in each recursion, 
we choose a concept at the 2nd level to split (i.e. remove it and elevate its children to the 2nd level)
that results in the greatest category utility (of the branch including the root and its children).

Thanks to Erik Harpstead for his previous work.

"""


def cluster(tree, instances, inst_size, min_split=1, max_split=1):
	for c in cluster_iter(tree, instances, inst_size, min_split=min_split, max_split=max_split, labels=True):
		yield c



def cluster_iter(tree, instances, inst_size, min_split=1, max_split=100000, labels=True):
	if min_split < 1:
		raise ValueError("min_split must be >= 1")
	if min_split > max_split:
		raise ValueError("min_split must be <= max_split")
	if len(instances) == 0:
		raise ValueError("cluster_iter called on an empty list.")

	tree = copy.deepcopy(tree)

	"""
	temp_clusters:
	the collection of clusters that each instance belongs to.
		doesn't necessary the leaves since the tree is adjusted incrementally
		they don't really matter - 
		they are just used to find the concept at the 2nd level that each instance belongs to
	"""
	temp_clusters = [tree.ifit(instance) for instance in instances]

	for nth_split in range(1, max_split + 1):

		cluster_assign = []
		child_cluster_assign = []

		if nth_split >= min_split:
			clusters = []
			for i, c in enumerate(temp_clusters):

				# Find c: the concept at the 2nd level that each instance belongs to
				# Find child: the concept at the 3rd level (if any; otherwise None) 
				# 			  that each instance belongs to
				"""
				Find c: 
					the concept at the 2nd level that each instance belongs to -> cluster_assign and clusters
				Find child: 
					the concept at the 3rd level (if any; otherwise None) that each instance belong to
					so if it exists, it is a child of c.
					-> child_cluster_assign
				"""
				child = None
				while c.parent and c.parent.parent:
					child = c
					c = c.parent
				if labels:
					clusters.append("Concept" + str(c.concept_id))
				else:
					clusters.append(c)
				cluster_assign.append(c)
				child_cluster_assign.append(child)

			yield clusters

		"""
		Then simulate the cases that each child at the 2nd level is split.
		Find the one resulting in the greatest CU and proceed.
		"""

		split_cus = []
		for i, target in enumerate(set(cluster_assign)):
			if len(target.children) == 0:
				continue
			c_labels = [label if label != target else child_cluster_assign[j] 
						for j, label in enumerate(cluster_assign)]
			split_cus.append((CU(c_labels, temp_clusters, inst_size), i, target))

		split_cus = sorted(split_cus)

		if not split_cus:
			break

		tree.root.split(split_cus[0][2])

		nth_split += 1


def CU(cluster, leaves, inst_size):
	"""
	Calculate the category utility of a tree state given clusters and leaves
	(the branch including the root and its children ONLY)
	"""
	temp_root = cluster[0].__class__(inst_size)
	temp_root.tree = cluster[0].tree
	for c in set(cluster):
		temp_child = cluster[0].__class__(inst_size)
		temp_child.tree = c.tree
		for leaf in leaves:
			if c.is_parent(leaf):
				temp_child.update_counts_from_node(leaf)
		temp_root.update_counts_from_node(temp_child)
		temp_root.children.append(temp_child)
	return - temp_root.category_utility()
