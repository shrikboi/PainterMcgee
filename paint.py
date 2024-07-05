from painting import Painting


class Paint:
    def __init__(self, original_image, rectangle_list, size):
        self.expanded = 0
        self.painting = Painting(original_image, rectangle_list, size)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.painting

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        self.expanded = self.expanded + 1
        return [(state.do_move(move), move, 1) for move in state.get_legal_moves()]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions. The sequence must
        be composed of legal moves
        """
        return len(actions)
