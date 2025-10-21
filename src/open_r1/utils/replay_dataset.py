"""Dataset helpers for mixing fresh and replayed experiences during training."""

import copy

from torch.utils.data import Dataset


class ReplayMixDataset(Dataset):
    """Dataset wrapper that annotates rows with the replay flag."""

    def __init__(self, base_ds, tok):
        """Store the underlying dataset and tokenizer used for assertions."""
        self.base_ds  = base_ds
        self.tok      = tok          # just for the debug assert

    def __len__(self):                       # unchanged
        """Return the number of records in the underlying dataset."""
        return len(self.base_ds)

    def __getitem__(self, idx):              # no replay logic here
        """Return the example at ``idx`` while clearing the replay flag."""
        item = copy.deepcopy(self.base_ds[idx])

        # → quick sanity–check only if already tokenised
        if "input_ids" in item:
            last_user = next(m for m in reversed(item["prompt"])
                             if m["role"] == "user")
            assert last_user["content"] in self.tok.decode(item["input_ids"]), \
                   "⛔ clue missing from encoded prompt!"

        item["is_replay"] = 0                # mark as fresh
        return item

def replay_collate(batch, *, replay_buffer, replay_prob):
    """Identity collate that simply clears replay flags and preserves accuracy."""
    for ex in batch:
        ex["is_replay"] = 0
        # do NOT pop accuracy here!
        # ex.pop("accuracy", None)    ← remove this line
    return batch
