class TreeNode:
    def __init__(self,data,probability):
        self.data = data
        self.probability = probability
        self.children = []
        self.parent = None
        self.bit = None

    def add_child(self,child):
        child.parent = self
        self.children.append(child)

    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.data , self.probability,"Bit - ",self.bit) 
        
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
            self.children[0].bit = 1
            self.children[1].bit = 0
            for i in self.children:
                i.assign_bits()

def build_huffman_tree_from_leaves():
    #creating leaf nodes for each symbol
    #A = 0.13 ,B = 0.26 ,C = 0.5 ,D =0.11
    test = TreeNode("T",0.11)
    symbol_1 = TreeNode("A",0.13)
    symbol_2 = TreeNode("B",0.26)
    symbol_3 = TreeNode("C",0.5)
    symbol_4 = TreeNode("D",0.11)
    

    leaves = [symbol_1,symbol_2,symbol_3,symbol_4]
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
    sorted_leaves[-1].assign_bits()    
    sorted_leaves[-1].print_tree()



if __name__ == '__main__':
    build_huffman_tree_from_leaves()