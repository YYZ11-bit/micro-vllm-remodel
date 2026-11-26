"""Scheduling policies and request admission logic."""

from collections import deque
from dataclasses import dataclass
from math import ceil

from micro_vllm.config import LLMConfig
from micro_vllm.core.kv_cache_manager import KVCacheHit, KVCacheManager
from micro_vllm.core.request import FinishReason, Request, RequestStatus
from micro_vllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from micro_vllm.executor.interface import ModelRunnerOutput, RequestVerification
from micro_vllm.speculative.acceptance import greedy_accept
from micro_vllm.speculative.drafter import Drafter

_SPECULATION_WINDOW = 8
_SPECULATION_MIN_ACCEPTANCE = 0.30
_SPECULATION_INITIAL_COOLDOWN = 8
_SPECULATION_MAX_COOLDOWN = 64

@dataclass
class _SpeculationState:
    accepted: int = 0
    verified: int = 0
    disabled: bool = False
    cooldown: int = 0
    probe: bool = False
    backoff: int = _SPECULATION_INITIAL_COOLDOWN

class Scheduler:
    """Own request state and build token-budgeted execution plans."""

    def __init__(
        self,
        config: LLMConfig,
        kv_cache_manager: KVCacheManager,
        drafter: Drafter | None = None,
    ) -> None:
        self._config = config
        self._kv_cache_manager = kv_cache_manager
        self._drafter = drafter
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._requests: dict[str, Request] = {}
        self._speculation_states: dict[str, _SpeculationState] = {}
        self._pending_plan: SchedulerOutput | None = None

    def add_request(self, request: Request) -> None:
        """Append a new waiting request to the admission queue."""

        scheduler_config = self._config.scheduler_config
        if request.status is not RequestStatus.WAITING:
            raise ValueError("request must be in WAITING status")
        if request.num_prompt_tokens >= scheduler_config.max_model_len:
            raise ValueError("prompt length must be less than max_model_len")
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")

        self._kv_cache_manager.register_request(request.request_id)
        self._requests[request.request_id] = request
        self._waiting.append(request)

    def abort_request(self, request_id: str) -> bool:
        """Abort an active request, returning whether its state changed."""

        request = self._requests.get(request_id)
        if request is None or request.is_finished:
            return False

        if request.status is RequestStatus.WAITING:
            self._waiting.remove(request)
        elif request.status is RequestStatus.RUNNING:
            self._running.remove(request)
        request.finish(FinishReason.ABORTED)
        if self._pending_plan is not None and any(
            scheduled.request_id == request_id
            for scheduled in self._pending_plan.scheduled_requests
        ):
            self._pending_plan = None
        self._kv_cache_manager.free_request(request_id)
        self._speculation_states.pop(request_id, None)
        return True

    def get_request(self, request_id: str) -> Request | None:
        """Return scheduler-owned request state for read-only inspection."""

        return self._requests.get(request_id)

    def has_unfinished_requests(self) -> bool:
        """Whether any scheduler-owned request has not reached a terminal state."""

        return any(not request.is_finished for request in self._requests.values())

    def schedule(self) -> SchedulerOutput:
        """Build an immutable plan for one token-budgeted scheduler step."""

        if self._pending_plan is not None:
            return self._pending_plan

        scheduler_config = self._config.scheduler_config
        remaining_token_budget = scheduler_config.max_num_batched_tokens
        scheduled_requests: list[ScheduledRequest] = []
        preempted_request_ids: list[str] = []
        cache_capacity_exhausted = False

        for request in self._running:
            if (
                remaining_token_budget == 0
                or len(scheduled_requests) == scheduler_config.max_num_seqs
            ):
                break
            self._maybe_propose_draft_tokens(
                request,
                remaining_token_budget=remaining_token_budget,
            )
            num_scheduled_tokens = min(
                request.num_pending_tokens,
                remaining_token_budget,
                scheduler_config.max_model_len - request.num_computed_tokens,
            )
            scheduled_request = self._allocate_scheduled_request(
                request,
                request.num_computed_tokens,
                num_scheduled_tokens,
            )
            while (
                scheduled_request is None
                and self._running[-1] is not request
            ):
                victim = self._running.pop()
                victim.preempt()
                self._kv_cache_manager.free_request_blocks(victim.request_id)
                self._waiting.append(victim)
                preempted_request_ids.append(victim.request_id)
                scheduled_request = self._allocate_scheduled_request(
                    request,
                    request.num_computed_tokens,
                    num_scheduled_tokens,
                )
            if scheduled_request is None:
                if not scheduled_requests:
                    raise RuntimeError(
                        f"request {request.request_id} cannot fit in KV cache"
                    )
                cache_capacity_exhausted = True
                break
            scheduled_requests.append(scheduled_request)
            remaining_token_budget -= num_scheduled_tokens
            
        while (
            self._waiting
            and not cache_capacity_exhausted
            and remaining_token_budget > 0
            and len(self._running) < scheduler_config.max_num_seqs
        ):
            request = self._waiting[0]
            cache_hit = self._kv_cache_manager.find_longest_cache_hit(
                request.request_id,
                request.all_token_ids,
            )
            num_hit_tokens = cache_hit.num_computed_tokens
            num_scheduled_tokens = min(
                request.num_known_tokens - num_hit_tokens,
                remaining_token_budget,
                scheduler_config.max_model_len - num_hit_tokens,
            )
            scheduled_request = self._allocate_scheduled_request(
                request,
                num_hit_tokens,
                num_scheduled_tokens,
                cache_hit=cache_hit,
            )
            if scheduled_request is None:
                if not scheduled_requests:
                    raise RuntimeError(
                        f"request {request.request_id} cannot fit in KV cache"
                    )
                break
            scheduled_requests.append(scheduled_request)
            remaining_token_budget -= scheduled_request.num_scheduled_tokens

            request.resume(num_hit_tokens)
            self._waiting.popleft()
            self._running.append(request)

        plan = SchedulerOutput(
            tuple(scheduled_requests),
            tuple(preempted_request_ids),
        )
        if plan.scheduled_requests:
            self._pending_plan = plan
        return plan

    def update(self, plan: SchedulerOutput, result: ModelRunnerOutput) -> None:
        """Atomically commit executor results for one scheduler plan."""

        if plan is not self._pending_plan:
            raise ValueError("scheduler plan is not pending or was already updated")

        scheduler_config = self._config.scheduler_config
        sampled_tokens = {
            sampled_token.request_id: sampled_token.token_id
            for sampled_token in result.sampled_tokens
        }

        finished_request_ids: list[str] = []
        verifications = {
            verification.request_id: verification
            for verification in result.verifications
        }
        
        for scheduled_request in plan.scheduled_requests:
            request = self._requests[scheduled_request.request_id]
            verification = verifications.get(request.request_id)
            if scheduled_request.num_draft_tokens > 0 and verification is None:
                raise ValueError(
                    f"request {scheduled_request.request_id} was scheduled for "
                    f"{scheduled_request.num_scheduled_tokens} tokens but has no "
                    "verification result"
                )
            if scheduled_request.num_draft_tokens == 0 and verification is not None:
                raise ValueError(
                    f"request {scheduled_request.request_id} was scheduled for "
                    "0 tokens but has a verification result"
                )
            if verification is not None:
                self._update_speculative_request(
                    request,
                    scheduled_request,
                    verification,
                    finished_request_ids,
                )
                continue
            request.advance_computed_tokens(
                scheduled_request.num_scheduled_tokens
            )
            token_id = sampled_tokens.get(request.request_id)
            if token_id is not None:
                request.append_sampled_token(token_id)
                params = request.sampling_params
                finish_reason = None
                if (
                    params.eos_token_id is not None
                    and token_id == params.eos_token_id
                    and not params.ignore_eos
                ):
                    finish_reason = FinishReason.EOS
                elif request.num_output_tokens >= params.max_tokens:
                    finish_reason = FinishReason.LENGTH
                elif (
                    request.num_known_tokens
                    == scheduler_config.max_model_len
                ):
                    finish_reason = FinishReason.MODEL_LENGTH
                if finish_reason is not None:
                    request.finish(finish_reason)
                    finished_request_ids.append(request.request_id)

            self._kv_cache_manager.commit_step(
                request.request_id,
                request.all_token_ids,
                request.num_computed_tokens,
            )

        if finished_request_ids:
            finished_request_id_set = set(finished_request_ids)
            self._running[:] = [
                request
                for request in self._running
                if request.request_id not in finished_request_id_set
            ]
            for request_id in finished_request_ids:
                self._speculation_states.pop(request_id, None)
                self._kv_cache_manager.free_request(request_id)
        self._pending_plan = None

    def _allocate_scheduled_request(
        self,
        request: Request,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        *,
        cache_hit: KVCacheHit | None = None,
    ) -> ScheduledRequest | None:
        allocation = self._kv_cache_manager.allocate_slots(
            request.request_id,
            num_computed_tokens,
            num_scheduled_tokens,
            cache_hit=cache_hit,
        )
        if allocation is None:
            return None

        end_computed = num_computed_tokens + num_scheduled_tokens
        requires_sampling = end_computed == request.num_known_tokens
        num_draft_tokens = request.num_draft_tokens
        if num_draft_tokens > 0 and (
            not requires_sampling
            or num_scheduled_tokens != num_draft_tokens + 1
        ):
            raise ValueError(
                "speculative request must schedule one pending token plus all drafts"
            )
        return ScheduledRequest(
            request_id=request.request_id,
            input_token_ids=request.all_token_ids[
                num_computed_tokens:end_computed
            ],
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            requires_sampling=requires_sampling,
            block_tables=allocation.block_tables,
            sampling_params=(
                request.sampling_params if requires_sampling else None
            ),
            sample_index=(
                request.num_output_tokens if requires_sampling else None
            ),
            num_draft_tokens=request.num_draft_tokens if requires_sampling else 0,
        )

    def _maybe_propose_draft_tokens(self, request: Request, remaining_token_budget: int) -> None:
        if self._drafter is None:
            return
        if request.sampling_params.is_greedy is False:
            return
        if request.num_pending_tokens != 1:
            return

        state = self._speculation_states.setdefault(request.request_id, _SpeculationState())
        if state.disabled:
            if state.cooldown > 0:
                state.cooldown -= 1
                return
            state.probe = True

        max_model_len = self._config.scheduler_config.max_model_len
        k_limit = min(
            self._config.speculative_config.max_draft_tokens,
            remaining_token_budget - 1,
            max_model_len - request.num_computed_tokens - 1
        )
        if k_limit <= 0:
            return
        drafterOutput = self._drafter.propose(
            request.all_token_ids,
            k_limit
        )
        token_ids = drafterOutput.tokens_ids
        if not token_ids:
            if state.disabled:
                state.probe = False
                state.backoff = min(state.backoff * 2, _SPECULATION_MAX_COOLDOWN)
                state.cooldown = state.backoff
                state.accepted = 0
                state.verified = 0
            return
        if len(token_ids) > 0:
            request.append_draft_tokens(list(token_ids))

    def _update_speculative_request(self, request: Request, 
                                    scheduled_request: ScheduledRequest,
                                    verification: RequestVerification,
                                    finished_request_ids: list[str])-> None:
        proposals = list(request.draft_token_ids)
        acceptance_result = greedy_accept(proposals, list(verification.predicted_token_ids))
        self._record_speculation_result(request.request_id, 
                                        accepted = len(acceptance_result.accepted), 
                                        verified = len(proposals),
                                        )
        request.clear_draft_tokens()
        committed_accepted_count = 0
        finish_reason: FinishReason | None = None
        for token_id in acceptance_result.accepted:
            request.append_sampled_token(token_id)
            committed_accepted_count += 1
            finish_reason = self._finish_reason_for_token(request, token_id)
            if finish_reason is not None:
                break

        if finish_reason is None:
            if acceptance_result.replacement:
                token_id = acceptance_result.replacement[0]
            else:
                token_id = acceptance_result.extra[0]
            request.append_sampled_token(token_id)
            finish_reason = self._finish_reason_for_token(request, token_id)
        request.advance_computed_tokens(1 + committed_accepted_count)
        block_size = self._kv_cache_manager._layout.block_size
        num_logical_blocks = ceil((request.num_computed_tokens + block_size - 1) // block_size)
        self._kv_cache_manager.truncate_request(request.request_id, num_logical_blocks)
        self._kv_cache_manager.commit_step(
                        request.request_id,
                        request.all_token_ids,
                        request.num_computed_tokens,
                    )
        if finish_reason is not None:
            request.finish(finish_reason)
            finished_request_ids.append(request.request_id)

    def _record_speculation_result(self, request_id: str, *, accepted: int, verified: int) -> None:
        state = self._speculation_states.setdefault(request_id, _SpeculationState())
        state.accepted += accepted
        state.verified += verified
        if state.probe:
            state.probe = False
            if(
                state.verified == 0
                or state.accepted / state.verified < _SPECULATION_MIN_ACCEPTANCE
            ):
                state.disabled = True
                state.backoff = min(state.backoff * 2, _SPECULATION_MAX_COOLDOWN)
                state.cooldown = state.backoff
            else:
                state.disabled = False
                state.cooldown = 0
                state.backoff = max(state.backoff // 2, _SPECULATION_INITIAL_COOLDOWN)
            state.accepted = 0
            state.verified = 0
            return

        if state.verified < _SPECULATION_WINDOW:
            return  

        if(
            state.accepted / state.verified < _SPECULATION_MIN_ACCEPTANCE
        ):
            state.disabled = True
            state.probe = False
            state.backoff = _SPECULATION_INITIAL_COOLDOWN
            state.cooldown = state.backoff
        state.accepted = 0
        state.verified = 0

    def _finish_reason_for_token(
        self,
        request: Request,
        token_id: int,
    ) -> FinishReason | None:
        params = request.sampling_params
        if (
            params.eos_token_id is not None
            and token_id == params.eos_token_id
            and not params.ignore_eos
        ):
            return FinishReason.EOS
        if request.num_output_tokens >= params.max_tokens:
            return FinishReason.LENGTH
        if request.num_known_tokens == self._config.scheduler_config.max_model_len:
            return FinishReason.MODEL_LENGTH
        return None
