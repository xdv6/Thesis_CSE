import heapq
from collections import defaultdict, deque

class AStarController:
    def __init__(self, options):
        """
        options: list of (rm_id, from_node, to_node)
        Used to generate heuristics automatically.
        """
        self.options = options
        self.graph = defaultdict(list)              # from_node -> [(to_node, cost, option_id)]
        self.heuristic = self._chosen_heuristic()
        self.start_node = 0                         # Always plan from node 0
        self.explored_edges = set()                 # Keep track of explored edges

    def _chosen_heuristic(self):
        return {
            -1: 0,

            # Root
            0: 4,

            # One step from root
            1: 3, 17: 3, 33: 3, 49: 3,

            # Two steps from root (intermediate)
            2: 2, 7: 2, 12: 2, 18: 2, 23: 2, 28: 2,
            34: 2, 39: 2, 44: 2, 50: 2, 55: 2, 60: 2,

            # Direct predecessors of terminals
            3: 1, 5: 1, 8: 1, 10: 1, 13: 1, 15: 1,
            19: 1, 21: 1, 24: 1, 26: 1, 29: 1, 31: 1,
            35: 1, 37: 1, 40: 1, 42: 1, 45: 1, 47: 1,
            51: 1, 53: 1, 56: 1, 58: 1, 61: 1, 63: 1
        }

    def convert_options_to_nodes(self, option_indices, options):
        """
        Converts a list of option indices to the corresponding sequence of nodes.

        Args:
            option_indices (list): List of indices into the options list.
            options (list): List of (rm_id, u1, u2) option tuples.

        Returns:
            list: List of traversed node IDs.
        """
        if not option_indices:
            return []

        # Start with the source node of the first option
        path_nodes = [options[option_indices[0]][1]]  # u1 of first option

        for idx in option_indices:
            _, _, u2 = options[idx]
            path_nodes.append(u2)

        return path_nodes

    def update_with_option_result(self, option_id, reward, from_node, to_node):
        """
        Called externally when a new edge is explored.
        """
        if (from_node, to_node) not in self.explored_edges:
            cost = -reward
            self.graph[from_node].append((to_node, cost, option_id))
            self.explored_edges.add((from_node, to_node))


    def get_cost_of_path(self, path):
        """
        Returns the cost of a path.
        Path is a list of option_ids.
        """
        total_cost = 0
        for option_id in path:
            from_node, to_node = self.options[option_id][1:3]
            # Find the matching (to_node, cost, option_id) in the adjacency list
            for neighbor, cost, opt_id in self.graph[from_node]:
                if neighbor == to_node and opt_id == option_id:
                    total_cost += cost
                    break
            else:
                raise ValueError(f"No edge from {from_node} to {to_node} with option_id {option_id}")
        return total_cost

    def get_plan(self):
        """
        Plans from node 0 to any known goal using current graph.
        Returns full list of option_ids from 0 to a goal node.
        A node is considered a goal if it has no outgoing edges in the current graph (lazy A*).
        """
        # min priority queue (f, g, node, path)
        open_set = [(self.heuristic.get(self.start_node, 0), 0, self.start_node, [])]
        visited = set()
        graph_keys = set(self.graph.keys())  # nodes with known outgoing edges



        while open_set:

            f, g, node, path = heapq.heappop(open_set)

            if node not in graph_keys:  # ✅ Node has no outgoing edges — it's a terminal node
                return path

            if node in visited:
                continue
            visited.add(node)

            for neighbor, cost, option_id in self.graph.get(node, []):
                if neighbor in visited:
                    continue
                new_g = g + cost
                new_f = new_g + self.heuristic.get(neighbor, float('inf'))
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [option_id]))



        return []  # No path found
