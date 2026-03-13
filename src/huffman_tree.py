class HuffmanTree:
    def __init__(self, word_frequency):
        self.V_size = len(word_frequency)
        self.word_frequency = word_frequency

        self.count = []
        self.code = []
        self.path_to_root = []
        self.leaves_under = []

        self.sorted_word_ids_by_frequency = []
        self.word_codes = []
        self.word_paths = []
        for i in range(self.V_size):
            self.word_codes.append([])
            self.word_paths.append([])

        self.build_tree_start()
    
    def build_tree_start(self):
        if self.V_size == 0:
            return

        if self.V_size == 1:
            self.word_codes[0] = []
            self.word_paths[0] = []
            return
        
        self.sorted_word_ids_by_frequency = sorted(range(self.V_size), key=lambda word_id: self.word_frequency[word_id])

        for i in range(self.V_size):
            self.count.append(int(self.word_frequency[self.sorted_word_ids_by_frequency[i]]))
            self.code.append([])
            self.path_to_root.append([])
            self.leaves_under.append([i])
        
        smallest_leaf_node = 0
        smallest_internal_node = self.V_size

        self.ok = self.build_tree_loop(smallest_leaf_node, smallest_internal_node)
        if self.ok:
            self.build_word_data()

    def build_tree_loop(self, smallest_leaf_node, smallest_internal_node):
        # full tree has been built
        while len(self.count) < 2 * self.V_size - 1:
            # first smallest node
            if smallest_leaf_node < self.V_size and (smallest_internal_node >= len(self.count) or self.count[smallest_leaf_node] <= self.count[smallest_internal_node]):
                min1 = smallest_leaf_node
                smallest_leaf_node += 1
            else:
                min1 = smallest_internal_node
                smallest_internal_node += 1

            # second smallest node
            if smallest_leaf_node < self.V_size and (smallest_internal_node >= len(self.count) or self.count[smallest_leaf_node] <= self.count[smallest_internal_node]):
                min2 = smallest_leaf_node
                smallest_leaf_node += 1
            else:
                min2 = smallest_internal_node
                smallest_internal_node += 1
            
            # create new internal node
            new_node_index = len(self.count)
            new_count = self.count[min1] + self.count[min2]

            self.count.append(new_count)
            self.code.append([])
            self.path_to_root.append([])
            self.leaves_under.append(self.leaves_under[min1] + self.leaves_under[min2])

            # left child gets bit 0
            for leaf_index in self.leaves_under[min1]:
                self.code[leaf_index].append(0)
                self.path_to_root[leaf_index].append(new_node_index)

            # right child gets bit 1
            for leaf_index in self.leaves_under[min2]:
                self.code[leaf_index].append(1)
                self.path_to_root[leaf_index].append(new_node_index)
        return True

    def build_word_data(self):
        for leaf in range(self.V_size):
            word_id = self.sorted_word_ids_by_frequency[leaf]

            # reverse because these were built from leaf to root
            root_to_leaf_code = list(reversed(self.code[leaf]))
            root_to_leaf_path = list(reversed(self.path_to_root[leaf]))

            internal_path = []
            for node_index in root_to_leaf_path:
                internal_path.append(node_index - self.V_size) # map node_index to rows in the hidden_output_matrix

            self.word_codes[word_id] = root_to_leaf_code
            self.word_paths[word_id] = internal_path