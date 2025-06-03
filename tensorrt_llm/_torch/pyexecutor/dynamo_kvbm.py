from typing import Any, List, Optional

import torch

from tensorrt_llm.bindings.internal.batch_manager import BaseKVCacheManager


class DynamoKvCacheManager(BaseKVCacheManager):

    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_max_num_blocks(*args: Any, **kwargs: Any):
        pass

    def allocate_pools(self, use_uvm: bool = False):
        pass

    def release_pools(self):
        pass

    # Ignore for now.
    def start_scheduling(self):
        pass

    def add_token(self, request_id: int):
        pass

    def add_sequence(
        self,
        request_id: int,
        input_length: int,
        beam_width: int,
        llm_request: Optional[Any] = None,
    ) -> None:
        pass

    def remove_sequence(self,
                        request_id: int,
                        llm_request: Optional[Any] = None):
        pass

    # Ignore for now.
    def scheduling_remove_sequence(self, request_id: int):
        pass

    def get_block_pool_pointers(self):
        pass

    def get_layer_to_pool_mapping(self):
        pass

    # Ignore for now.
    def get_block_offsets_of_batch(
        self,
        output: torch.Tensor,
        first_batch_slot_idx: int,
        batch_size: int,
        beam_width: int,
    ):
        pass

    # Ignore for now.
    def copy_block_offsets(
        self,
        output: torch.Tensor,
        output_slot_offset: int,
        request_id: int,
    ):
        pass

    def copy_batch_block_offsets(self, output: torch.Tensor,
                                 request_ids: List[int]):
        pass

    # Ignore for now.
    def get_latest_events(self, timeout_ms: Optional[float] = None):
        pass

    # Ignore for now.
    def rewind_kv_cache(self, request_id: int, rewind_lengths: int):
        pass

    # Ignore for now.
    def store_context_blocks(self, llm_request: Any):
        pass

    def get_cache_block_ids(self, request_id: int, window_size: int):
        pass

    def get_batch_cache_block_ids(self, request_ids: List[int],
                                  window_size: int):
        pass

    # Ignore for now.
    def get_newly_allocated_block_ids(self, request_id: int, window_size: int):
        pass

    # Only need num_free_blocks.
    def get_kv_cache_stats(self):
        pass

    # Ignore for now
    def get_needed_blocks_one_step(self, llm_request: Any, flag: bool,
                                   value: int):
        pass

    # Ignore for now.
    def get_remaining_blocks_to_completion(self, llm_request: Any, value: int):
        pass

    def get_primary_pool_data(self, layer_idx: int):
        pass

    # Ignore for now.
    def flush_iteration_events(self):
        pass

    @property
    def cross_kv(self):
        pass

    @property
    def enable_block_reuse(self):
        pass

    @property
    def max_blocks_per_seq(self):
        pass

    @property
    def max_num_blocks(self):
        pass

    @property
    def num_pools(self):
        pass

    @property
    def tokens_per_block(self):
        pass
