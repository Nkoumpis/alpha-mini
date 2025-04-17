# Updated MCTS implementation (acts.py)
# Original source: https://github.com/CGLemon/pyDLGO/blob/master/mcts.py

from utils.board import Board, PASS, RESIGN, BLACK, WHITE
from network import Network         # import from root module
from utils.time_control import TimeControl
import math

class Node:
    CPUCT = 0.5  # PUCT hyperparameter

    def __init__(self, p: float):
        self.policy = p          # Prior probability from policy network
        self.nn_eval = 0.0       # Value network estimate (clamped)
        self.values = 0.0        # Accumulated value
        self.visits = 0          # Visit count
        self.children = {}       # Map vertex -> Node

    def clamp(self, v: float) -> float:
        """Map winrate in [-1,1] to [0,1]"""
        return (v + 1) / 2

    def inverse(self, v: float) -> float:
        """Swap perspective of winrate"""
        return 1 - v

    def expand_children(self, board: Board, network: Network) -> float:
        """
        Populate children using policy network and return clamped value.
        """
        policy, value = network.get_outputs(board.get_features())
        # Add legal moves
        for idx in range(board.num_intersections):
            vtx = board.index_to_vertex(idx)
            if board.legal(vtx):
                self.children[vtx] = Node(policy[idx])
        # Add pass move
        self.children[PASS] = Node(policy[board.num_intersections])
        # Clamp scalar value
        self.nn_eval = self.clamp(value)
        return self.nn_eval

    def remove_superko(self, board: Board):
        """Remove children that violate superko rule."""
        for vtx in list(self.children.keys()):
            if vtx != PASS:
                next_board = board.copy()
                next_board.play(vtx)
                if next_board.superko():
                    self.children.pop(vtx)

    def puct_select(self) -> int:
        """Select child with highest PUCT score."""
        total_visits = sum(child.visits for child in self.children.values()) or 1
        sqrt_total = math.sqrt(total_visits)
        best_move, best_score = None, -float('inf')
        for vtx, child in self.children.items():
            # Q value
            if child.visits != 0:
                q = self.inverse(child.values / child.visits)
            else:
                q = self.clamp(0)
            # PUCT score
            score = q + self.CPUCT * child.policy * (sqrt_total / (1 + child.visits))
            if score > best_score:
                best_score, best_move = score, vtx
        return best_move

    def update(self, v: float):
        """Backpropagate value."""
        self.values += v
        self.visits += 1

    def get_best_move(self, resign_threshold: float) -> int:
        """Choose move with most visits, resign if too low value."""
        best_vtx = max(self.children.items(), key=lambda item: item[1].visits)[0]
        child = self.children[best_vtx]
        if self.inverse(child.values / child.visits) < resign_threshold:
            return RESIGN
        return best_vtx

    def to_string(self, board: Board) -> str:
        """Debug string of statistics."""
        header = f"Root -> W: {self.values/self.visits:.2%}, P: {self.policy:.2%}, V: {self.visits}\n"
        lines = [header]
        # Sort children by visits descending
        for visits, vtx in sorted(((c.visits, v) for v, c in self.children.items()), reverse=True):
            child = self.children[vtx]
            if child.visits != 0:
                w = self.inverse(child.values / child.visits)
                lines.append(
                    f"  {board.vertex_to_text(vtx):4} -> W: {w:.2%}, P: {child.policy:.2%}, V: {child.visits}\n"
                )
        return ''.join(lines)

class Search:
    """Monte Carlo Tree Search controller."""
    def __init__(self, board: Board, network: Network, time_control: TimeControl):
        self.root_board = board
        self.root_node = None
        self.network = network
        self.time_control = time_control

    def _prepare_root_node(self):
        self.root_node = Node(1.0)
        val = self.root_node.expand_children(self.root_board, self.network)
        self.root_node.remove_superko(self.root_board)
        self.root_node.update(val)

    def _play_simulation(self, color: int, board: Board, node: Node) -> float:
        """Recursively simulate one playout."""
        # Terminal check: two passes
        if board.num_passes >= 2:
            score = board.final_score()
            if score > 1e-4:
                return 1.0 if color == BLACK else 0.0
            if score < -1e-4:
                return 1.0 if color == WHITE else 0.0
            return 0.5
        # Select or expand
        if node.children:
            vtx = node.puct_select()
            board.to_move = color
            board.play(vtx)
            next_color = WHITE if color == BLACK else BLACK
            next_node = node.children[vtx]
            value = self._play_simulation(next_color, board, next_node)
        else:
            value = node.expand_children(board, self.network)
        node.update(value)
        return node.inverse(value)

    def think(self, playouts: int, resign_threshold: float, verbose: bool) -> int:
        """Run multiple playouts and choose the best move."""
        if self.root_board.num_passes >= 2:
            return PASS
        self.time_control.clock()
        if verbose:
            print(self.time_control)
        self._prepare_root_node()
        from tqdm import tqdm
        for _ in tqdm(range(playouts)):
            max_time = self.time_control.get_thinking_time(
                self.root_board.to_move,
                self.root_board.board_size,
                self.root_board.move_num
            )
            if self.time_control.should_stop(max_time):
                break
            sim_board = self.root_board.copy()
            sim_color = sim_board.to_move
            self._play_simulation(sim_color, sim_board, self.root_node)
        self.time_control.took_time(self.root_board.to_move)
        if verbose:
            print(self.root_node.to_string(self.root_board))
            print(self.time_control)
        return self.root_node.get_best_move(resign_threshold)
