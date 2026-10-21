"""Scheduler-only tests: no model, no GPU. Run from the repository root:

    python -m pytest -q tools/waiting_order_scheduler
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from tests.v1.kv_connector.unit.utils import create_model_runner_output  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402
from waiting_order_scheduler import WaitingOrderScheduler  # noqa: E402

FCFS = "def score(request, ctx):\n    return request['queue_index']\n"
LONGEST_FIRST = "def score(request, ctx):\n    return -request['prompt_tokens']\n"
RECORD_PREFIX = "uses_prefix_hits = True\n\ndef score(request, ctx):\n    return request['queue_index']\n"
NAN_FIRST = "def score(request, ctx):\n    return float('nan') if request['submit_seq'] == 1 else 0.0\n"
CHECK_CTX = (
    "def score(request, ctx):\n"
    "    assert ctx['num_gpu_blocks'] == 9999 and ctx['free_blocks'] == 9999  # 10000 blocks minus the null block\n"
    "    assert ctx['block_size'] == 16 and ctx['max_num_seqs'] == 1 and ctx['worker_role'] == 'aggregated'\n"
    "    return request['queue_index']\n"
)


def make_scheduler(tmp_path, monkeypatch, policy, log=False, **kwargs):
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(policy)
    monkeypatch.setenv("WAITING_POLICY", str(policy_path))
    if log:
        monkeypatch.setenv("WAITING_POLICY_LOG", str(tmp_path / "decisions.jsonl"))
    else:
        monkeypatch.delenv("WAITING_POLICY_LOG", raising=False)
    base = create_scheduler(**kwargs)  # the same configuration vLLM's own scheduler tests use
    return WaitingOrderScheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def new_ids(output):
    return [r.req_id for r in output.scheduled_new_reqs]


def decisions(tmp_path):
    return [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]


def test_queue_index_policy_reproduces_the_native_scheduler(tmp_path, monkeypatch):
    # Prefix caching on, with a prefix-aware policy, so the per-request cache lookups run as on the server.
    ours = make_scheduler(tmp_path, monkeypatch, RECORD_PREFIX, max_num_seqs=2, max_num_batched_tokens=64, enable_prefix_caching=True)
    native = create_scheduler(max_num_seqs=2, max_num_batched_tokens=64, enable_prefix_caching=True)
    for scheduler in (ours, native):
        requests = create_requests(num_requests=5, num_tokens=20, max_tokens=3)
        steps = []
        for step in range(8):
            if step < 3:
                scheduler.add_request(requests[step])
            if step == 4:
                scheduler.add_request(requests[3])
                scheduler.add_request(requests[4])
            output = scheduler.schedule()
            steps.append((new_ids(output), dict(output.num_scheduled_tokens)))
            scheduler.update_from_output(output, create_model_runner_output(scheduler.running))
        if scheduler is ours:
            ours_steps = steps
    assert ours_steps == steps


def test_policy_order_decides_which_requests_a_pass_admits(tmp_path, monkeypatch):
    scheduler = make_scheduler(tmp_path, monkeypatch, LONGEST_FIRST, max_num_batched_tokens=35)
    for req_id, length in (("0", 10), ("1", 20), ("2", 30)):
        scheduler.add_request(create_requests(num_requests=1, num_tokens=length, req_ids=[req_id])[0])
    output = scheduler.schedule()
    # Longest first: 30 tokens whole, then 5 of the 20-token prompt (chunked), then the budget is gone.
    assert new_ids(output) == ["2", "1"]
    assert output.num_scheduled_tokens == {"2": 30, "1": 5}
    assert [r.request_id for r in scheduler.waiting] == ["0"]


def test_views_carry_the_contract_fields_and_age_tokens_count_passes(tmp_path, monkeypatch):
    scheduler = make_scheduler(tmp_path, monkeypatch, CHECK_CTX, log=True, max_num_seqs=1, max_num_batched_tokens=64)
    r0, r1, r2 = create_requests(num_requests=3, num_tokens=16, max_tokens=4)
    scheduler.add_request(r0)
    output = scheduler.schedule()  # pass 1: r0's 16-token prefill; one waiting request, policy not called
    scheduler.update_from_output(output, create_model_runner_output(scheduler.running))
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    for _ in range(3):  # passes 2-4: r0 decodes one token each; the running set is full, no policy call
        output = scheduler.schedule()
        scheduler.update_from_output(output, create_model_runner_output(scheduler.running))
    assert decisions(tmp_path) == []  # the log file is opened at construction; nothing was scored yet
    output = scheduler.schedule()  # pass 5: r0 finished after 4 output tokens; r1 and r2 are scored
    log = decisions(tmp_path)
    assert len(log) == 1
    first = log[0]
    assert set(first) == {"now_ms", "worker_id", "num_running", "waiting_len", "free_blocks", "requests", "order"}
    assert first["waiting_len"] == 2 and first["num_running"] == 0
    views = first["requests"]
    assert [v["queue_index"] for v in views] == [0, 1]
    assert [v["submit_seq"] for v in views] == [2, 3]
    assert all(v["prompt_tokens"] == 16 and v["generated_tokens"] == 0 and v["num_preemptions"] == 0 for v in views)
    assert all(v["wait_ms"] >= 0 for v in views)
    assert "cached_prefix_tokens" not in views[0]
    # Passes 2-4 scheduled one token each after r1 and r2 arrived; pass 1 ran before they arrived.
    assert [v["age_tokens"] for v in views] == [3, 3]
    assert first["order"] == [2, 3]
    assert new_ids(output) == ["1"]  # one seat (max_num_seqs=1): the first in the order is admitted


def test_prefix_fields_match_the_admission_gate(tmp_path, monkeypatch):
    scheduler = make_scheduler(
        tmp_path, monkeypatch, RECORD_PREFIX, log=True, enable_prefix_caching=True, block_size=16, max_num_seqs=2
    )
    shared = create_requests(num_requests=2, num_tokens=48, same_prompt=True, max_tokens=8)
    other = create_requests(num_requests=2, num_tokens=48, req_ids=["unused", "other"])[1]  # a different prompt
    scheduler.add_request(shared[0])
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output(scheduler.running))  # its 3 blocks are cached
    scheduler.add_request(shared[1])
    scheduler.add_request(other)
    output = scheduler.schedule()  # shared[0] is running: one free seat, two waiting, the policy is called
    same, different = decisions(tmp_path)[0]["requests"]
    # The hit is capped at num_tokens - 1 = 47 and to whole blocks: 32 tokens. shared[0] holds those two
    # blocks, so the gate only needs the third one free; the other prompt needs all three.
    assert same["cached_prefix_tokens"] == 32 and same["needed_blocks"] == 1
    assert different["cached_prefix_tokens"] == 0 and different["needed_blocks"] == 3
    assert new_ids(output) == ["1"]  # queue_index order admits shared[1]; the seat count stops the pass


def test_evictable_hit_blocks_still_count_as_needed(tmp_path, monkeypatch):
    scheduler = make_scheduler(
        tmp_path, monkeypatch, RECORD_PREFIX, log=True, enable_prefix_caching=True, block_size=16, max_num_seqs=2
    )
    shared = create_requests(num_requests=2, num_tokens=48, same_prompt=True, max_tokens=8)
    other = create_requests(num_requests=2, num_tokens=48, req_ids=["unused", "other"])[1]
    scheduler.add_request(shared[0])
    output = scheduler.schedule()
    # shared[0] finishes on its first token: its blocks stay cached but nobody holds them
    scheduler.update_from_output(output, create_model_runner_output(scheduler.running, use_eos=True))
    scheduler.add_request(shared[1])
    scheduler.add_request(other)
    scheduler.schedule()
    same, different = decisions(tmp_path)[0]["requests"]
    # The hit is still 32 tokens, but the gate counts evictable blocks as blocks to allocate.
    assert same["cached_prefix_tokens"] == 32 and same["needed_blocks"] == 3
    assert different["cached_prefix_tokens"] == 0 and different["needed_blocks"] == 3


def test_prefix_cap_holds_for_the_whole_pass(tmp_path, monkeypatch):
    # 130 waiting with a prefix-aware policy: the pass starts above the cap and stays native even after
    # admissions bring the queue under 128; the next pass, starting at 126, is scored.
    scheduler = make_scheduler(tmp_path, monkeypatch, RECORD_PREFIX, log=True, enable_prefix_caching=True, max_num_seqs=4)
    for request in create_requests(num_requests=130, num_tokens=16, max_tokens=1):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert new_ids(output) == ["0", "1", "2", "3"] and decisions(tmp_path) == []
    scheduler.update_from_output(output, create_model_runner_output(scheduler.running, use_eos=True))
    scheduler.schedule()
    log = decisions(tmp_path)
    assert len(log) == 1 and log[0]["waiting_len"] == 126


def test_nan_scores_last_and_ties_follow_submit_seq(tmp_path, monkeypatch):
    scheduler = make_scheduler(tmp_path, monkeypatch, NAN_FIRST, log=True, max_num_batched_tokens=64)
    for request in create_requests(num_requests=3, num_tokens=16):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert new_ids(output) == ["1", "2", "0"]
    assert decisions(tmp_path)[0]["order"] == [2, 3, 1]


def test_a_non_numeric_score_is_an_error(tmp_path, monkeypatch):
    scheduler = make_scheduler(tmp_path, monkeypatch, "def score(request, ctx):\n    return 'low'\n")
    for request in create_requests(num_requests=2, num_tokens=16):
        scheduler.add_request(request)
    with pytest.raises(TypeError):
        scheduler.schedule()


def test_async_scheduling_is_refused(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        make_scheduler(tmp_path, monkeypatch, FCFS, async_scheduling=True)
