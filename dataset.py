import torch


class RetailDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None, transform=None, device=None):
        self.data = data
        self.labels = labels
        self.num_classes = len(set(labels))
        self.transform = transform
        self.device = device if device is not None else torch.device("cpu")

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx].detach().clone(), device=self.device)
            for key, val in self.data.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], device=self.device)
        return item

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return "RetailDataset"

    def __str__(self):
        return str(
            {
                "data": self.data["pixel_values"].shape,
                "labels": self.labels.shape,
                "num_classes": self.num_classes,
                "num_samples": len(self.labels),
            }
        )
