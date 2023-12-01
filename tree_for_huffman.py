class TreeNode:
    def __init__(self,data,probability):
        self.data = data
        self.probability = probability
        self.children = []
        self.parent = None
        self.bit = None
        self.bit_sequence = False

    def add_child(self,child):
        child.parent = self
        self.children.append(child)

    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.data , self.probability,"Bit - ",self.bit,"Bit sequence - ",self.bit_sequence) 
        
        if self.children:
            for i in self.children:
                i.print_tree()

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level
    def assign_bits(self):
        #assign bits to children
        # print(type(self.children))
        
        if self.children:
            self.children[0].bit = "1"
            self.children[1].bit = "0"
            for i in self.children:
                i.assign_bits()
                
    def assign_bit_sequence(self):
        # print(self.data)
        #handling the root node
        if not self.parent:
            for i in self.children:
                    i.assign_bit_sequence()
        if self.parent:
            self.bit_sequence = self.parent.bit_sequence + self.bit
            if self.children:
                for i in self.children:
                        i.assign_bit_sequence()
            else:
                self.bit_sequence = self.bit_sequence[1:]

    def get_leaf_values(self):

        if not self.children:
            #a leaf node
            return self.data,self.bit_sequence

        left_values = self.children[0].get_leaf_values()
        right_values = self.children[1].get_leaf_values()
        self.children[0].get_leaf_values()
        self.children[1].get_leaf_values()

        return left_values,right_values

def flatten_tuple(nested_tuple):
    flattened_list = []
    for element in nested_tuple:
        if isinstance(element, tuple):
            flattened_list.extend(flatten_tuple(element))
        else:
            flattened_list.append(element)
    tuples_list = []

    # for i in range(0, len(flattened_list), 2):
    #     tuples_list.append((flattened_list[i], flattened_list[i + 1]))
    # print(tuples_list)
    return flattened_list

def build_huffman_tree_from_leaves():
    #creating leaf nodes for each symbol
    #A = 0.13 ,B = 0.26 ,C = 0.5 ,D =0.11
    test = TreeNode("T",0.11)
    symbol_1 = TreeNode("87",0.02734375)
    symbol_2 = TreeNode("111",0.04296875)
    symbol_3 = TreeNode("135",0.03515625)
    symbol_4 = TreeNode("159",0.046875)
    symbol_5 = TreeNode("183",0.09375)
    symbol_6 = TreeNode("207",0.05078125)
    symbol_7 = TreeNode("231",0.05859375)
    symbol_8 = TreeNode("255",0.64453125)
    
    #[0.02734375 0.04296875 0.03515625 0.046875   0.09375    0.05078125 0.05859375 0.64453125]
    #[ 87 111 135 159 183 207 231 255]

    leaves = [symbol_1,symbol_2,symbol_3,symbol_4,symbol_5,symbol_6,symbol_7,symbol_8]
    sorted_leaves = sorted(leaves, key=lambda node: node.probability, reverse=True)
    # #now itteratively create the tree
    count = 1
    while len(sorted_leaves)>1:
        new_node_probability = sorted_leaves[-1].probability + sorted_leaves[-2].probability
        new_node = TreeNode(data= f"inter_{count}",probability=new_node_probability)
        new_node.add_child(sorted_leaves[-2])
        new_node.add_child(sorted_leaves[-1]) # does this order matter
        sorted_leaves = sorted_leaves[:-2]
        sorted_leaves.append(new_node)
        sorted_leaves = sorted(sorted_leaves, key=lambda node: node.probability, reverse=True)
        count = count + 1
    sorted_leaves[-1].bit_sequence = "X" 
    sorted_leaves[-1].assign_bits()
    sorted_leaves[-1].assign_bit_sequence()
    # sorted_leaves[-1].print_tree()
    flattened_list = flatten_tuple(sorted_leaves[-1].get_leaf_values())
    tuple_list = []
    for i in range(0, len(flattened_list), 2):
        tuple_list.append((flattened_list[i], flattened_list[i + 1]))
    flattened_list = tuple_list
    print(flattened_list)


if __name__ == '__main__':
    build_huffman_tree_from_leaves()