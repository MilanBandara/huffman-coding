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
        print(self.data)
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

def build_huffman_tree_from_leaves():
    #creating leaf nodes for each symbol
    #A = 0.13 ,B = 0.26 ,C = 0.5 ,D =0.11
    test = TreeNode("T",0.11)
    symbol_1 = TreeNode("A",0.05)
    symbol_2 = TreeNode("B",0.09)
    symbol_3 = TreeNode("C",0.12)
    symbol_4 = TreeNode("D",0.13)
    symbol_5 = TreeNode("E",0.16)
    symbol_6 = TreeNode("F",0.45)
    

    leaves = [symbol_1,symbol_2,symbol_3,symbol_4,symbol_5,symbol_6]
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
    sorted_leaves[-1].print_tree()

    # return sorted_leaves[-1]

# def get_codes(root):


# huffman_tree_root = build_huffman_tree_from_leaves()




if __name__ == '__main__':
    build_huffman_tree_from_leaves()