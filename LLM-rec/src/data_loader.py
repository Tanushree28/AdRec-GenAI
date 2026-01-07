# LLM-rec/src/data_loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Dict


class KuaiRecInteractions(Dataset):
    """
    Simple dataset over a (user_id, item_id, watch_ratio) table.
    Assumes columns: user_id, item_id, watch_ratio.
    """

    def __init__(self, df: pd.DataFrame):
        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(df["video_id"].values, dtype=torch.long)
        self.watch_ratios = torch.tensor(df["watch_ratio"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "watch_ratio": self.watch_ratios[idx],
        }


def load_big_matrix(path: str, sep: str = ",") -> pd.DataFrame:
    # Use dtypes to keep memory lower on 16GB RAM
    df = pd.read_csv(
        path,
        sep=sep,
        dtype={"user_id": "int32", "video_id": "int32"},
        usecols=["user_id", "video_id", "watch_ratio"],
    )
    return df


def make_dataloaders(
    df: pd.DataFrame,
    batch_size: int,
    val_ratio: float,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    dataset = KuaiRecInteractions(df)
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # infer number of users/items from df
    n_users = int(df["user_id"].max()) + 1
    n_items = int(df["video_id"].max()) + 1

    stats = {"n_users": n_users, "n_items": n_items}
    return train_loader, val_loader, stats
