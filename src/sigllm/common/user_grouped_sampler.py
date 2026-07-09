"""User-grouped sampler for per-user pairwise (uAUC) training.

The default random sampling almost never places two interactions of the same
user in one batch, so a per-user pairwise ranking loss (BPR, a uAUC surrogate)
would have no pairs to learn from. This sampler reorders sample indices so that
every contiguous ``batch_size`` window is composed of single-user "chunks" with
interleaved positive/negative labels — giving the loss many valid same-user
(pos, neg) pairs per batch.

It is a plain ``Sampler`` (not a ``BatchSampler``): it yields a permutation of
all indices, and the DataLoader chunks that stream into ``batch_size`` windows.
This keeps it compatible with the existing DataLoader wiring and the
``IterLoader.set_epoch`` shuffle hook. Every sample is emitted exactly once per
epoch (drop_last trims the final partial window).
"""

import random

import numpy as np
from torch.utils.data import Sampler


class UserGroupedSampler(Sampler):
    def __init__(
        self,
        user_ids,
        labels,
        batch_size,
        items_per_user=4,
        num_replicas=1,
        rank=0,
        seed=0,
    ):
        self.batch_size = int(batch_size)
        self.items_per_user = max(2, int(items_per_user))
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        # Incremented each __iter__ so we reshuffle every epoch even when the
        # runner never calls set_epoch (single-process: IterLoader only calls it
        # under distributed). Keeps batches fresh across epochs.
        self._iter_count = 0

        user_ids = np.asarray(user_ids).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        self._num_samples = int(user_ids.shape[0])

        # Group sample indices by user, split by label so we can interleave
        # pos/neg inside each user (maximises mixed-label chunks).
        self._user_pos = {}
        self._user_neg = {}
        for idx in range(self._num_samples):
            u = int(user_ids[idx])
            bucket = self._user_pos if int(labels[idx]) == 1 else self._user_neg
            bucket.setdefault(u, []).append(idx)
        self._users = sorted(set(self._user_pos) | set(self._user_neg))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _interleave(pos, neg, rng):
        rng.shuffle(pos)
        rng.shuffle(neg)
        out = []
        i = j = 0
        while i < len(pos) or j < len(neg):
            if i < len(pos):
                out.append(pos[i])
                i += 1
            if j < len(neg):
                out.append(neg[j])
                j += 1
        return out

    def __iter__(self):
        # Tuple seed -> fresh, deterministic shuffle each epoch whether the
        # variation comes from set_epoch (distributed) or _iter_count (single).
        rng = random.Random((self.seed, self.epoch, self._iter_count))
        self._iter_count += 1
        users = list(self._users)
        rng.shuffle(users)
        # Shard users across ranks (distributed); world_size==1 -> all users.
        users = users[self.rank :: self.num_replicas]

        chunks = []
        for u in users:
            seq = self._interleave(
                list(self._user_pos.get(u, [])),
                list(self._user_neg.get(u, [])),
                rng,
            )
            for s in range(0, len(seq), self.items_per_user):
                chunks.append(seq[s : s + self.items_per_user])

        rng.shuffle(chunks)
        order = [idx for chunk in chunks for idx in chunk]
        return iter(order)

    def __len__(self):
        if self.num_replicas == 1:
            return self._num_samples
        return self._num_samples // self.num_replicas
