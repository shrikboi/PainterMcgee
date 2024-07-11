import numpy as np


def not_a_real_search(paint):
    curr_state = paint.get_start_state()
    for i in range(1600):
        print(f"iteration {i}")
        best_action = None
        best_score = np.inf
        for successor, action, cost in paint.get_successors(curr_state):
            if successor.score() < best_score:
                best_score = successor.score()
                best_action = action
        curr_state.add_move(best_action)

    return curr_state.current_painting
