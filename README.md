# alphago mini Setup and Usage Guide

This guide explains how to clone, patch, train, and run the Minigo AlphaZero‑style Go engine on macOS, and integrate it with the Sabaki GUI.

---

## Prerequisites

- **macOS** with Terminal and Git
- **Python 3.13.2** (verify with `python3 --version`)
- **Homebrew** (optional)

---

## 1. Clone the Repository

```bash
cd ~
git clone https://github.com/chenyuntc/minigo.git
cd minigo
```

---

## 2. Create & Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # prompt shows (.venv)
```

---

## 3. Install Dependencies

```bash
pip install "torch>=1.3.1" tqdm dill fire numpy
```

Verify installation:

```bash
pip show torch
```

---

## 4. Unzip SGF Data

```bash
unzip -q sgf.zip
```

---

## 5. Patch Source Files

### 5.1 `train_behaviour_cloning.py`

```diff
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-network = Network(board_size=9)
-network.trainable()
-network = network.to(device)
+network = Network(board_size=9).to(device)
+network.trainable()

 # In the training loop:
-    inputs   = inputs.cuda()
-    target_p = target_p.cuda()
-    target_v = target_v.cuda()
+    inputs   = inputs.to(device)
+    target_p = target_p.to(device)
+    target_v = target_v.to(device)
```

### 5.2 `network.py`

```diff
-def get_outputs(...):
-    return m(p).item(), v.item()
+def get_outputs(...):
+    probs = torch.softmax(p, dim=1)[0].detach().cpu().numpy()
+    return probs, v.item()
```

### 5.3 `utils/mcts.py`

```diff
-from utils.network import Network
+from network import Network

-        self.nn_eval = self.clamp(value[0])
+        self.nn_eval = self.clamp(value)
```

### 5.4 `utils/gtp.py`

```diff
-from utils.network import Network
+from network import Network

-        self.network.load_pt(args.weights)
+        self.network.load_ckpt(args.weights)
```

---

## 6. Quick Smoke Test

Run a short training session to verify the pipeline:

```bash
python3 train_behaviour_cloning.py --steps=100 --verbose_step=20
```

---

## 7. Full Training (CPU)

> **Warning:** ~27 days on CPU for 1.28M steps. Use a GPU (e.g., Colab) for faster results.

```bash
python3 train_behaviour_cloning.py --steps=1280000
```

---

## 8. Test CLI GTP Engine

Make the launcher executable and run:

```bash
chmod +x dlgo.py
python3 dlgo.py --playouts 10 --resign-threshold 0.01
```

Interact via terminal:

```bash
boardsize 9
clear_board
genmove black
showboard
quit
```

Or as a one‑liner:

```bash
echo -e "boardsize 9\nclear_board\ngenmove black\nshowboard\nquit" \
  | python3 dlgo.py --playouts 10 --resign-threshold 0.01
```

---

## 9. Install Sabaki GUI

1. Download the macOS archive from Sabaki releases:
   - `sabaki-v0.52.2-mac-arm64.7z` (Apple Silicon)
   - `sabaki-v0.52.2-mac-x64.7z` (Intel)
2. Extract with p7zip or The Unarchiver:

   ```bash
   brew install p7zip   # if needed
   7z x sabaki-v0.52.2-mac-arm64.7z
   ```
3. Move `Sabaki.app` to `/Applications`.

---

## 10. Configure Minigo in Sabaki

Open Sabaki’s Preferences → Engines → `+` and enter:

- **Name:** `Minigo`
- **Protocol:** GTP
- **Path:** `/Users/you/minigo/.venv/bin/python3`
- **Arguments:**
  ```
  -u /Users/you/minigo/dlgo.py --weights 128000 --playouts 800 --resign-threshold 0.01
  ```
- **Working Directory:** `/Users/you/minigo`

Click **Save**.

---

## 11. Play via Sabaki GUI

1. **File → New:** set Handicap to **No stones**, Board Size **9×9** → **OK.**
2. Under the board, set **Black: Minigo**, **White: Human**.
3. Press **F8** (or Window → Show Engine Panel) to open the right‑hand sidebar.
4. In the Engine panel, click the color toggle until it reads **○ Minigo** (engine will play Black).
5. Click **Play ▶️** to have Minigo generate Black’s move (a black stone appears).
6. Click on the board to place a white stone (your move).
7. Alternate **Play ▶️** and clicks to complete the game.
8. Use **Undo** or **Clear Board** in the Engine panel as needed.

---

## 12. Tips for Stronger Play

- Increase `--playouts` (e.g., to 1600) for deeper search.
- After training finish, use real checkpoint: `--weights 128000`.
- Use 19×19 boards via File → New for full‑size games.
- Save games as SGF in Sabaki for later analysis.

---

Enjoy your self‑trained Minigo Go engine!

