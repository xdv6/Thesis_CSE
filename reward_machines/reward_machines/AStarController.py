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
            0: 3,
            1: 2,
            6: 2,
            11: 2,
            2: 1,
            4: 1,
            7: 1,
            9: 1,
            12: 1,
            14: 1,
            3: 0,
            5: 0,
            8: 0,
            10: 0,
            13: 0,
            15: 0
        }

    def update_with_option_result(self, option_id, reward, from_node, to_node):
        """
        Called externally when a new edge is explored.
        """
        if (from_node, to_node) not in self.explored_edges:
            cost = -reward
            self.graph[from_node].append((to_node, cost, option_id))
            self.explored_edges.add((from_node, to_node))

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
