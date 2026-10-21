"""Order the waiting queue of vLLM's synchronous scheduler with a policy file.

A policy is one Python file::

    uses_prefix_hits = False   # optional; True adds the two prefix fields and the 128-request cap

    def score(request: dict, ctx: dict) -> float: ...

Each pass that would admit waiting requests scores every waiting request once, after the running
requests are scheduled and before the admission loop, and tries them in ascending score, ties by
``submit_seq``; NaN counts as +inf. Admission, token budgets, KV allocation, chunking and preemption
stay native: the first request that does not fit still stops the loop, nothing is preempted for a
candidate, and the order is not written back to the queue. The pass keeps the native order, without
calling the policy, when fewer than two requests wait and, for a prefix-aware policy, when more
than 128 wait.

``request`` fields: ``queue_index`` (native order, preempted requests first), ``submit_seq`` (arrival
order at this scheduler, the tie key), ``wait_ms`` (since the frontend received the request),
``age_tokens`` (tokens of the passes this scheduler started since the request arrived), ``prompt_tokens``,
``generated_tokens`` (kept across a preemption), ``num_preemptions``, and with ``uses_prefix_hits``
``cached_prefix_tokens`` (whole blocks, at most the sequence length minus one) and ``needed_blocks``
(blocks the admission gate requires to be free: the sequence's blocks minus the cached ones that a
running request holds). ``ctx`` fields: ``now_ms``, ``worker_id``, ``worker_role``, ``num_running``,
``max_num_seqs``, ``waiting_len``, ``block_size``, ``num_gpu_blocks`` (the pool minus vLLM's null
block) and ``free_blocks`` (blocks no running request holds; cached ones count as free).

Usage: put this directory on PYTHONPATH, set WAITING_POLICY to the policy file and start vLLM with
``--scheduler-cls waiting_order_scheduler.WaitingOrderScheduler --no-async-scheduling``. With
WAITING_POLICY_LOG set, every policy call appends one JSON line: the per-pass ``ctx`` fields, the
requests exactly as the policy saw them, and the chosen order as ``submit_seq`` values.

Written against vLLM v0.26.0's Scheduler; the class relies on its waiting loop asking
``_select_waiting_queue_for_scheduling`` for the queue to peek and pop from.
"""

import importlib.util
import json
import math
import numbers
import os
import sys
import time

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

PREFIX_HITS_WAITING_CAP = 128


def load_policy(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"waiting_policy_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PassOrder:
    """The queue one pass pops from: the policy's order over a snapshot of the native queue."""

    def __init__(self, native, ordered):
        self.native = native
        self.ordered = ordered
        self.position = 0

    def peek_request(self):
        return self.ordered[self.position]

    def pop_request(self):
        request = self.ordered[self.position]
        self.native.remove_request(request)
        self.position += 1
        return request


class WaitingOrderScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.policy == SchedulingPolicy.FCFS, "the policy orders the FCFS queue"
        assert not self.scheduler_config.async_scheduling, "written for the synchronous scheduler"
        self.policy_module = load_policy(os.environ["WAITING_POLICY"])
        self.uses_prefix_hits = bool(getattr(self.policy_module, "uses_prefix_hits", False))
        log_path = os.environ.get("WAITING_POLICY_LOG")
        self.decision_log = open(log_path, "a") if log_path else None
        self.worker_id = os.getpid()
        self.next_submit_seq = 1  # the simulator numbers requests from 1
        self.tokens_started = 0  # tokens of every pass this scheduler has started
        self.pass_order = None

    def add_request(self, request):
        assert not request.resumable, "streaming-input sessions are outside this adapter's scope"
        request.submit_seq = self.next_submit_seq
        self.next_submit_seq += 1
        request.enqueued_at_tokens = self.tokens_started
        super().add_request(request)

    def schedule(self, *args, **kwargs):
        self.pass_order = None
        try:
            output = super().schedule(*args, **kwargs)
        finally:
            self.pass_order = None
        self.tokens_started += output.total_num_scheduled_tokens
        return output

    def _select_waiting_queue_for_scheduling(self):
        assert not self.skipped_waiting, "blocked waiting states are outside this adapter's scope"
        if self.pass_order is None:  # decided once per pass, as the simulator does
            requests = list(self.waiting)
            native = len(requests) < 2 or (self.uses_prefix_hits and len(requests) > PREFIX_HITS_WAITING_CAP)
            self.pass_order = self.waiting if native else PassOrder(self.waiting, self.order(requests))
        return self.pass_order

    def order(self, requests):
        now = time.time()
        ctx = {
            "now_ms": now * 1000.0,
            "worker_id": self.worker_id,
            "worker_role": "aggregated",
            "num_running": len(self.running),
            "max_num_seqs": self.max_num_running_reqs,
            "waiting_len": len(requests),
            "block_size": self.block_size,
            "num_gpu_blocks": self.cache_config.num_gpu_blocks - 1,
            "free_blocks": self.kv_cache_manager.block_pool.get_num_free_blocks(),
        }
        views = [self.view(index, request, now) for index, request in enumerate(requests)]
        log_line = None
        if self.decision_log is not None:
            log_line = {key: ctx[key] for key in ("now_ms", "worker_id", "num_running", "waiting_len", "free_blocks")}
            log_line["requests"] = [dict(view) for view in views]
        keyed = []
        for index, view in enumerate(views):
            score = self.policy_module.score(view, ctx)
            if isinstance(score, bool) or not isinstance(score, numbers.Real):
                raise TypeError(f"score must return a number, got {type(score).__name__}")
            if score != score:  # NaN
                score = math.inf
            keyed.append((score, view["submit_seq"], index))
        keyed.sort()
        if log_line is not None:
            log_line["order"] = [submit_seq for _, submit_seq, _ in keyed]
            self.decision_log.write(json.dumps(log_line) + "\n")
            self.decision_log.flush()
        return [requests[index] for _, _, index in keyed]

    def view(self, index, request, now):
        view = {
            "queue_index": index,
            "submit_seq": request.submit_seq,
            "wait_ms": (now - request.arrival_time) * 1000.0,
            "age_tokens": self.tokens_started - request.enqueued_at_tokens,
            "prompt_tokens": request.num_prompt_tokens,
            "generated_tokens": request.num_output_tokens,
            "num_preemptions": request.num_preemptions,
        }
        if self.uses_prefix_hits:
            # The lookup admission runs, without touching blocks or recording a hit.
            hit_blocks, hit_tokens, _ = self.kv_cache_manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
            held = sum(1 for block in hit_blocks[0] if block.ref_cnt > 0)
            view["cached_prefix_tokens"] = hit_tokens
            view["needed_blocks"] = -(-request.num_tokens // self.block_size) - held
        return view
