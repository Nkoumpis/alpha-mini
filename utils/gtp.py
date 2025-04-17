from sys import stderr, stdout, stdin
from utils.board import Board, PASS, RESIGN, BLACK, WHITE, INVLD
from network import Network  # import from root, not utils.network
from utils.mcts import Search
from config import BOARD_SIZE, KOMI, INPUT_CHANNELS, PAST_MOVES
from utils.time_control import TimeControl

class GTP_ENGINE:
    def __init__(self, args):
        self.args = args
        self.board = Board(BOARD_SIZE, KOMI)
        self.network = Network(BOARD_SIZE)
        self.time_control = TimeControl()
        self.network.trainable(False)
        self.board_history = [self.board.copy()]
        if self.args.weights is not None:
            # load_ckpt instead of load_pt
            self.network.load_ckpt(self.args.weights)

    # For GTP command "clear_board"
    def clear_board(self):
        self.board.reset(self.board.board_size, self.board.komi)
        self.board_history = [self.board.copy()]

    # For GTP command "genmove"
    def genmove(self, color):
        c = self.board.to_move
        if color.lower() in ("black", "b"):
            c = BLACK
        elif color.lower() in ("white", "w"):
            c = WHITE
        self.board.to_move = c
        search = Search(self.board, self.network, self.time_control)
        move = search.think(self.args.playouts, self.args.resign_threshold, self.args.verbose)
        if self.board.play(move):
            self.board_history.append(self.board.copy())
        return self.board.vertex_to_text(move)

    # For GTP command "play"
    def play(self, color, move):
        c = INVLD
        if color.lower() in ("black", "b"):
            c = BLACK
        elif color.lower() in ("white", "w"):
            c = WHITE
        if move == "pass":
            vtx = PASS
        elif move == "resign":
            vtx = RESIGN
        else:
            x = ord(move[0]) - (ord('A') if move[0].isupper() else ord('a'))
            y = int(move[1:]) - 1
            if x >= 8:
                x -= 1
            vtx = self.board.get_vertex(x, y)
        if c != INVLD:
            self.board.to_move = c
        if self.board.play(vtx):
            self.board_history.append(self.board.copy())
            return True
        return False

    # For GTP command "undo"
    def undo(self):
        if len(self.board_history) > 1:
            self.board_history.pop()
            self.board = self.board_history[-1].copy()

    # For GTP command "boardsize"
    def boardsize(self, bsize):
        self.board.reset(bsize, self.board.komi)
        self.board_history = [self.board.copy()]

    # For GTP command "komi"
    def komi(self, k):
        self.board.komi = k

    # For GTP command "time_settings"
    def time_settings(self, main_time, byo_time, byo_stones):
        if not main_time.isdigit() or not byo_time.isdigit() or not byo_stones.isdigit():
            return False
        self.time_control.time_settings(int(main_time), int(byo_time), int(byo_stones))
        return True

    # For GTP command "time_left"
    def time_left(self, color, time, stones):
        c = INVLD
        if color.lower() in ("black", "b"):
            c = BLACK
        elif color.lower() in ("white", "w"):
            c = WHITE
        if c == INVLD:
            return False
        self.time_control.time_left(c, int(time), int(stones))
        return True

    # For GTP command "showboard"
    def showboard(self):
        stderr.write(str(self.board))
        stderr.flush()

class GTP_LOOP:
    COMMANDS_LIST = [
        "quit", "name", "version", "protocol_version", "list_commands",
        "play", "genmove", "undo", "clear_board", "boardsize", "komi",
        "time_settings", "time_left", "showboard"
    ]

    def __init__(self, args):
        self.engine = GTP_ENGINE(args)
        self.args = args
        self.loop()

    def loop(self):
        while True:
            cmd = stdin.readline().split()
            if not cmd:
                continue
            main = cmd[0]
            if main == "quit":
                self.success_print("")
                break
            self.process(cmd)

    def process(self, cmd):
        main = cmd[0]
        if main == "name":
            self.success_print("dlgo")
        elif main == "version":
            version = "0.1"
            self.success_print(version)
        elif main == "protocol_version":
            self.success_print("2")
        elif main == "list_commands":
            self.success_print("\n".join(self.COMMANDS_LIST))
        elif main == "clear_board":
            self.engine.clear_board()
            self.success_print("")
        elif main == "play" and len(cmd) >= 3:
            if self.engine.play(cmd[1], cmd[2]):
                self.success_print("")
            else:
                self.fail_print("")
        elif main == "undo":
            self.engine.undo()
            self.success_print("")
        elif main == "genmove" and len(cmd) >= 2:
            move = self.engine.genmove(cmd[1])
            self.success_print(move)
        elif main == "boardsize" and len(cmd) >= 2:
            self.engine.boardsize(int(cmd[1]))
            self.success_print("")
        elif main == "komi" and len(cmd) >= 2:
            self.engine.komi(float(cmd[1]))
            self.success_print("")
        elif main == "showboard":
            self.engine.showboard()
            self.success_print("")
        elif main == "time_settings" and len(cmd) >= 4:
            if self.engine.time_settings(cmd[1], cmd[2], cmd[3]):
                self.success_print("")
            else:
                self.fail_print("")
        elif main == "time_left" and len(cmd) >= 4:
            if self.engine.time_left(cmd[1], cmd[2], cmd[3]):
                self.success_print("")
            else:
                self.fail_print("")
        else:
            self.fail_print("Unknown command")

    def success_print(self, res):
        stdout.write(f"= {res}\n\n")
        stdout.flush()

    def fail_print(self, res):
        stdout.write(f"? {res}\n\n")
        stdout.flush()
