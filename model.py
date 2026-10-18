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

# Step 2 - ttft_with_cache
def ttft_with_cache(prompt_len, cached_tokens, prefill_tps, overhead_s):
    return round(overhead_s + (prompt_len - cached_tokens) / prefill_tps, 6)


def prefill_saved_s(prompt_len, cached_tokens, prefill_tps):
    if cached_tokens == 0:
        return 0
    return cached_tokens / prefill_tps


def kv_bytes(tokens, kv_bytes_per_token):
    return tokens * kv_bytes_per_token


def blocks_for(memory_bytes, block_size, kv_bytes_per_token):
    return int(memory_bytes // (block_size * kv_bytes_per_token))


def savings_report(prompt_lens, cached_lens, prefill_tps, overhead_s):
    total_prompts = sum(prompt_lens)
    total_cached = sum(cached_lens)

    cached_times = [
        ttft_with_cache(p, c, prefill_tps, overhead_s)
        for p, c in zip(prompt_lens, cached_lens)
    ]
    uncached_times = [
        ttft_with_cache(p, 0, prefill_tps, overhead_s)
        for p in prompt_lens
    ]

    mean_ttft = sum(cached_times) / len(cached_times) if cached_times else 0.0
    mean_ttft_nocache = (
        sum(uncached_times) / len(uncached_times) if uncached_times else 0.0
    )

    ttft_reduction = (
        1 - mean_ttft / mean_ttft_nocache if mean_ttft_nocache else 0.0
    )
    fraction_saved = total_cached / total_prompts if total_prompts else 0.0

    return {
        "mean_ttft": round(mean_ttft, 6),
        "mean_ttft_nocache": round(mean_ttft_nocache, 6),
        "ttft_reduction": round(ttft_reduction, 6),
        "tokens_saved": int(total_cached),
        "fraction_saved": round(fraction_saved, 6),
    }

