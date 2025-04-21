import heapq
import random

class AStarController:
    def __init__(self, env, cost_dict, heuristic, option_map):
        """
        env: the environment (for access to RM state)
        cost_dict: {(from_node, to_node): cost}
        heuristic: {node: h(node)}
        option_map: list of (rm_id, from_node, to_node)
        """
        self.env = env
        self.cost_dict = cost_dict
        self.heuristic = heuristic
        self.option_map = option_map  # list where index is option_id

    def get_action(self, state, valid_options):
        """
        Returns the next option_id according to A* plan from current RM state to goal
        """
        current_rm_state = self.env.get_rm_state()  # e.g., 0, 1, 2...

        # Build graph of valid options
        graph = {}
        for option_id in valid_options:
            _, from_node, to_node = self.option_map[option_id]
            if from_node not in graph:
                graph[from_node] = []
            graph[from_node].append((to_node, option_id))

        # A* search
        open_set = [(self.heuristic[current_rm_state], 0, current_rm_state, [])]  # (f, g, node, path)
        visited = set()

        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if self.heuristic[node] == 0:
                if path:
                    return path[0]  # return the first option_id in the path
                else:
                    return random.choice(valid_options)

            if node in visited:
                continue
            visited.add(node)

            for neighbor, option_id in graph.get(node, []):
                cost = self.cost_dict.get((node, neighbor), 1.0)
                new_g = g + cost
                new_f = new_g + self.heuristic[neighbor]
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [option_id]))

        # fallback
        return random.choice(valid_options)
