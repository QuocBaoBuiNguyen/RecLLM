import torch
from torch.utils.data import Dataset


class QFormerAlignmentDataset(Dataset):
    def __init__(self, filename: str):
        obj = torch.load(filename, map_location="cpu", weights_only=False)
        self.samples = obj["samples"]
        self.stats = obj.get("stats", {})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        out = {
            "sample_type": sample["sample_type"],
            "u": torch.tensor(sample["u"], dtype=torch.long),
            "i_left": torch.tensor(sample["i_left"], dtype=torch.long),
            "i_right": torch.tensor(sample["i_right"], dtype=torch.long),
            "weight": torch.tensor(sample["weight"], dtype=torch.float),
            "instruction": sample["instruction"],
            "text": sample["text"],
        }
        # Variable-length history id list (user_item samples from newer pkls;
        # [] for other sample types and for old pkls). Emitted for EVERY row —
        # the collate derives its key set from the first batch element, and
        # train batches mix sample types, so a conditional key would either be
        # dropped or KeyError depending on batch order. Kept as a plain list:
        # the collate keeps non-tensor fields as lists; the loss pads per
        # batch and falls back to the MF user vector when all rows are empty.
        out["his"] = list(sample.get("his", []))
        return out
