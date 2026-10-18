"""
Prefix Caching and Cache-Aware Routing

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - PrefixCache
def block_hashes(tokens, block_size):
    hashes = []
    previous_hash = 0

    for i in range(0, len(tokens) - block_size + 1, block_size):
        block = tuple(tokens[i:i + block_size])
        current_hash = hash((previous_hash, block))
        hashes.append(current_hash)
        previous_hash = current_hash

    return hashes


class PrefixCache:
    def __init__(self, capacity_blocks, block_size):
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self.blocks = {}
        self.tick = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _touch(self, block_hash):
        self.tick += 1
        self.blocks[block_hash] = self.tick

    def lookup(self, tokens):
        hashes = block_hashes(tokens, self.block_size)
        n_present = 0

        for block_hash in hashes:
            if block_hash not in self.blocks:
                break

            self._touch(block_hash)
            n_present += 1

        self.hits += n_present
        self.misses += len(hashes) - n_present

        return n_present * self.block_size

    def insert(self, tokens):
        hashes = block_hashes(tokens, self.block_size)

        for block_hash in hashes:
            if block_hash not in self.blocks:
                if self.capacity_blocks <= 0:
                    continue

                if len(self.blocks) >= self.capacity_blocks:
                    lru_hash = min(
                        self.blocks,
                        key=self.blocks.get
                    )
                    del self.blocks[lru_hash]
                    self.evictions += 1

            self._touch(block_hash)

    def hit_rate(self):
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

