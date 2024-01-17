from sys import stdin

class BinaryTree(object):
    def __init__(self, n=0):
        self.tree = [[] for _ in range(n+2)]

    def __len__(self):
        # Return the total number of elements in the tree
        return len(self.data)

    def lelf(self, parent):
        return self.tree[parent][0]

    def right(self, parent):
        return self.tree[parent][1]

    def set_node(self, parent, child1, child2):
        self.tree[parent].append(child1)
        self.tree[parent].append(child2)

    def is_leaf(self, parent):
        print(f"root: {parent}")
        return self.tree[parent*2] == -1 and self.tree[parent*2 + 1] == -1


def width_of_binary_tree(args):
    '''
    Calculate the widest level and the width of that level
    =================================================================================================
    Arguments:
        + args: something containing information about the input binary tree
    Outputs:
        + widest_level: widest level of given binary tree
        + max_width: widht of the widest level of given binary tree
    '''

    ### TODO: fill in here ###
    def traversal(binTree, root, height, kwargs):
        if root == -1:
            return

        left = binTree.tree[root][0]
        right = binTree.tree[root][1]
        traversal(binTree, left, height+1, kwargs)
        # print(f"root: {root}")
        kwargs['w'] += 1
        if height in kwargs['width_level']:
            if kwargs['max_width'] < kwargs['w'] - kwargs['width_level'][height]: 
                kwargs['max_width'] = kwargs['w'] - kwargs['width_level'][height]
                kwargs['widest_level'] = height
            kwargs['width_level'][height] = kwargs['w']
        else:
            kwargs['width_level'][height] = kwargs['w']
        # print(kwargs['width_level'])
        # print(f"debug: {kwargs['widest_level']} {kwargs['max_width']}")

        traversal(binTree, right, height+1, kwargs)


    g = {
            "widest_level": 1,
            "max_width": 0,
            "width_level": {},
            "w": 0
    }
    traversal(args['tree'], args['root'], 1, g)
    widest_level = g['widest_level']
    max_width = g['max_width'] + 1

    ##########################

    return widest_level, max_width

def main():
    ### TODO: You are free to define the input value of the function as you wish. ###
    n = int(input())
    args = {}
    args['tree'] = BinaryTree(n)
    root = [-1] * (n+1)
    for _ in range(n):
        line = input()
        k, l , r = [int(word) for word in line.split()]
        if l != -1:
            root[l] = k
        if r != -1:
            root[r] = k

        args['tree'].set_node(k, l, r)
    for i in range(1, len(root)):
        if root[i] == -1:
            args['root'] = i

    output = width_of_binary_tree(args)

    with open("output.txt", "w") as f:
        f.write(f"{output[0]} {output[1]}\n")

if __name__ == "__main__":
    main()
