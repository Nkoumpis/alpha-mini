import os
os.environ["OMP_NUM_THREADS"] = "1"
import time

import torch
import torch.nn as nn
import torch.optim as optim
# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from network import Network
from dataset import DataSet


def main(
    data_dir="sgf", steps=400000, verbose_step=1000, batch_size=2048, learning_rate=1e-3
):
    # Initialize network and move to the correct device
    network = Network(board_size=9)
    network.trainable()
    network = network.to(device)

    # Prepare the data set from SGF files
    data_set = DataSet(data_dir, batch_size, steps)
    dataloader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],    # Unbatch each sample
        num_workers=0,                # In-process data loading
    )

    # Loss functions and optimizer
    cross_entry = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(
        network.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    policy_running_loss = 0.0
    value_running_loss = 0.0
    start_time = time.time()
    from tqdm import tqdm

    for step, data in tqdm(enumerate(dataloader), total=steps):
        # Unpack batch
        inputs, target_p, target_v = data
        # Move tensors to the correct device
        inputs   = inputs.to(device)
        target_p = target_p.to(device)
        target_v = target_v.to(device)

        # Forward pass
        policy_output, value_output = network(inputs)

        # Compute and backpropagate loss
        p_loss = cross_entry(policy_output, target_p)
        v_loss = mse_loss(value_output, target_v)
        loss = p_loss + v_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate running losses
        policy_running_loss += p_loss.item()
        value_running_loss += v_loss.item()

        # Learning rate decay and checkpointing
        if (step + 1) % 128000 == 0:
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
            network.save_ckpt(f"{step + 1}")

        # Verbose logging
        if (step + 1) % verbose_step == 0:
            elapsed = time.time() - start_time
            rate = verbose_step / elapsed
            remaining_step = steps - (step + 1)
            eta_seconds = int(remaining_step / rate)
            print(
                f"{time.strftime('%H:%M:%S')} | "
                f"Step {step+1}/{steps} "
                f"({100*(step+1)/steps:.2f}%): "
                f"policy_loss={policy_running_loss/verbose_step:.4f}, "
                f"value_loss={value_running_loss/verbose_step:.4f}, "
                f"{rate:.2f} steps/sec, ETA {eta_seconds}s"
            )
            policy_running_loss = 0.0
            value_running_loss = 0.0
            start_time = time.time()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
