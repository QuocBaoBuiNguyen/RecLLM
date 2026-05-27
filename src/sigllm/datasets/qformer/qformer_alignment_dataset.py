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
        return {
            "sample_type": sample["sample_type"],
            "u": torch.tensor(sample["u"], dtype=torch.long),
            "i_left": torch.tensor(sample["i_left"], dtype=torch.long),
            "i_right": torch.tensor(sample["i_right"], dtype=torch.long),
            "weight": torch.tensor(sample["weight"], dtype=torch.float),
            "instruction": sample["instruction"],
            "text": sample["text"],
        }
