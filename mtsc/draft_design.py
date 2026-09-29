from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Sequence


# =========================
# KVCachePool Data Contract
# =========================

# Hashes identify token prefixes; backend-specific keys and namespaces are
# derived by the pool. Layout, topology and hash block size are configured
# when constructing the pool, rather than repeated in every event.
BlockHash = bytes
BlockIds = tuple[tuple[int, ...], ...]  # One tuple per KV cache group.


class KVReadyEvent(Protocol):
    """Device completion signal; waited on by I/O workers, not the caller."""

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class LoadEvent:
    """Load [start_load, end_load) into already allocated local blocks.

    block_ids describes the full request prefix, grouped by KV cache group.
    block_hashes contains prefix hashes through end_load. Both bounds are
    token offsets. The pool derives each group's chunk/block mapping from
    its configured layout.
    """

    request_id: str
    block_ids: BlockIds
    block_hashes: tuple[BlockHash, ...]
    start_load: int
    end_load: int


@dataclass(frozen=True)
class SaveEvent:
    """Save complete cache chunks in [start_save, end_save).

    Both bounds are token offsets into the full request prefix described by
    block_hashes and block_ids. Repeated saves for the same request may arrive
    before earlier saves finish; the pool queues and coalesces pending work.

    The caller keeps source blocks valid and unchanged until completion or
    preempt() returns. ready_event=None means the data is already ready.
    """

    request_id: str
    block_ids: BlockIds
    block_hashes: tuple[BlockHash, ...]
    start_save: int
    end_save: int
    ready_event: KVReadyEvent | None = None


@dataclass(frozen=True)
class LoadResult:
    """Terminal result for the request's one logical load.

    A backend may split the load into group/chunk I/O subtasks. Publish this
    result only after every subtask is terminal and no destination writes
    remain, including on failure. Completion does not imply full success.

    loaded_tokens is the contiguous usable prefix length, INCLUDING the
    event's start_load, valid across all cache groups. A miss may yield
    only start_load with error=None; an I/O failure sets error. Bytes beyond
    loaded_tokens must not be used, even if some chunks were copied.
    Invalid destination block IDs are reported separately by take_errors().
    """

    request_id: str
    loaded_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class SaveResult:
    """Result after all currently submitted saves for a request are terminal.

    Covers every save accepted since the previous SaveResult was collected.
    No running or pending save remains at collection time, including events
    not yet converted into futures. Terminal work includes successful, failed
    and safely cancelled operations; source reads must have stopped in every
    case. preempt() instead clears the work without publishing a result.
    error=None means
    every selected chunk succeeded; errors from earlier work must not be
    hidden by a later successful save. Existing chunks count as success.
    On failure, valid chunks already saved may remain in storage; atomic
    multi-chunk commit is not required. The request may submit more saves
    later; this result does not signal request completion.
    """

    request_id: str
    error: str | None = None


@dataclass
class PoolPollResult:
    loads: list[LoadResult] = field(default_factory=list)
    saves: list[SaveResult] = field(default_factory=list)


# =========================
# KVCachePool Interface
# =========================


class KVCachePool(ABC):
    """Remote KV cache wrapper with pluggable storage backends.

    Backends may include Mooncake Store, MongoDB, FileSystem or memory.
    The pool owns storage lookup and I/O; the connector owns request
    scheduling, block allocation and P-D fallback. The pool tracks pending
    work by request_id, with one result per load and aggregated save results.

    load/save submit work without waiting for I/O. A queue or worker thread
    is an implementation choice, not part of this contract. Results,
    including immediate completions, are collected through poll().
    Normally a request has one logical load in an allocation lifecycle,
    rather than one load per decode step. It may have multiple incremental
    saves as prefill/decode produces more tokens. At most one uncollected
    load is allowed; duplicate loads raise ValueError. Repeated saves are
    accepted, ordered per request and coalesced where safe. A retry requires
    the previous load and block errors to be consumed or preempt() to fence
    and clear them before resubmission.

    Internal tracking is an implementation choice. A typical layout is
    request_id -> one load future (aggregating any load subtasks), and
    request_id -> save state containing running work, pending events and
    accumulated errors. A list of save futures is also valid, but its
    completion check must include pending events not yet submitted to I/O.
    In the common contiguous-save case, one running save and one merged
    pending event suffice; incompatible snapshots remain separate.
    Events are immutable metadata snapshots; no KV tensor copy is required.
    The caller keeps registered tensors and task blocks alive until terminal
    completion or preempt() returns. Public calls are serialized by the
    caller; implementations synchronize their own background workers.
    Request IDs must not be reused while work, results or block errors remain
    outstanding. Error block IDs must be consumed before their blocks are reused;
    preempt() fences old work and clears results before resubmission.
    """

    @abstractmethod
    def register(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        """Register local KV memory before load/save.

        Layout information comes from pool configuration. Registered tensors
        remain alive until close(); re-registration with pending tasks is
        unsupported.
        """
        pass

    @abstractmethod
    def lookup(self, token_count: int, hashes: Sequence[BlockHash]) -> int:
        """Return the remotely available contiguous prefix length in tokens.

        Synchronous metadata query, without copying KV or allocating blocks.
        Returns 0 on a miss, at most token_count, aligned to usable cache
        chunks across all groups. Backend failures raise an exception.
        The caller may schedule this query asynchronously if needed.
        A hit is advisory: data can be evicted before load(), whose result
        remains authoritative. Lookup does not require registered KV memory.
        """
        pass

    @abstractmethod
    def load(self, load_event: LoadEvent) -> None:
        """Submit KV loading into the event's destination blocks.

        Non-blocking; misses and I/O failures are reported through poll().
        If split into multiple I/O subtasks, collect and fence all of them
        before reporting the request's LoadResult, even if one fails early.
        The connector decides whether to recompute or use P-D fallback.
        """
        pass

    @abstractmethod
    def save(self, save_event: SaveEvent) -> None:
        """Submit KV saving; completion is reported through poll().

        Non-blocking. Wait for ready_event before reading source KV memory.
        Save only complete chunks; do not persist an unfinished token tail.
        Keep an executing save's snapshot unchanged. Pending adjacent or
        overlapping ranges may be merged using their union and the newest
        block/hash snapshot, only if it covers the union and preserves each
        token position's source mapping. Wait for readiness covering all merged
        data; replacing older events is safe only if the newer signal fences
        their writes too. Otherwise retain separate pending snapshots.
        For example, while [0, 64) runs, pending [64, 96) and [96, 128)
        can become [64, 128). Never merge across block reuse after preemption.
        A new submission before a save result is collected extends the work
        covered by that result; do not emit stale completion for earlier work.
        """
        pass

    @abstractmethod
    def poll(self) -> PoolPollResult:
        """Drain completed load/save operations without waiting for I/O.

        Each result is returned once. LoadResult fences that load's writes;
        SaveResult fences all accepted saves for that request and is returned
        only when no running or pending save remains. Save completion must be
        rechecked when polled, since new submissions can extend pending work.
        Checking all existing futures is insufficient if a pending event
        has not been submitted yet. Completion means all related work is
        terminal, not necessarily successful; collect errors without returning
        early while other I/O can still access local memory.
        Operation bookkeeping is released when results and block errors are
        consumed; the pool needs no request-end signal. If a block is shared
        by load and save work, both must finish before the caller reuses it.
        Publishing a LoadResult also publishes its invalid block IDs for
        take_errors(); the caller drains those in the same polling cycle.
        SaveResult while the request is still running only updates the
        connector's save status; it must not directly become finished_sending.
        A later save submission makes that request's save status pending again.
        After request_finished(), the connector aggregates block-release
        requirements: request ended, no outstanding saves, and any required
        P-D send terminal. This applies to Store-only Decode saves too;
        send requirements describe block-release dependencies, not only PD.
        """
        pass

    @abstractmethod
    def take_errors(self) -> set[int]:
        """Drain invalid local destination block IDs from polled loads.

        Non-blocking; returns an empty set when no errors are pending. Include
        blocks whose requested KV is unusable after a miss, partial load or
        I/O failure, even when LoadResult.error is None. Exclude the existing
        local prefix and all save failures: failed persistence does not
        invalidate valid local KV. IDs are physical local block IDs, not
        hashes or per-request block indices.

        The connector may recover these blocks through P-D fallback before
        reporting remaining failures via get_block_ids_with_load_errors().
        The worker output carries them as invalid_block_ids for scheduler
        recovery. This set carries no additional completion notification;
        local memory access is fenced by the corresponding LoadResult.
        """
        pass

    @abstractmethod
    def preempt(self, request_id: str) -> None:
        """Cancel and fence all load/save work for a request before block reuse.

        May block if backend I/O cannot safely be cancelled. On return the
        request's work no longer reads or writes local KV memory, uncollected
        results, pending block errors and request bookkeeping are cleared,
        and resubmission is safe.
        Calls with no tracked request are harmless. Already persisted valid
        KV need not be rolled back.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Stop accepting work, cancel queued tasks and fence running I/O.

        Release registered memory/resources only after workers stop accessing
        them. Idempotent; subsequent submissions raise RuntimeError.
        """
        pass

# =========================
# KVTransfer Data Contract
# =========================


@dataclass(frozen=True)
class SendEvent:
    """Publish the request's ready source KV for P-D transfer.

    block_ids is the full source block table, grouped and ordered by KV cache
    group. All listed KV must be safe to read when send() is called. D selects
    a suffix by providing its destination block table in RecvEvent.
    Source blocks remain valid and unchanged until the send result is polled
    or preempt() returns.
    """

    request_id: str
    transfer_id: str
    block_ids: BlockIds


@dataclass(frozen=True)
class RecvEvent:
    """Request a source KV suffix into D's allocated destination blocks.

    block_ids contains only blocks to receive, grouped and ordered by KV cache
    group; it excludes blocks already populated locally. For each group, P
    copies the last len(destination_group) source blocks in order. Empty groups
    require no transfer. Source and destination tables describe the same prefix
    endpoint with compatible group/block layouts; arbitrary ranges are unsupported.
    The backend resolves matching P workers through bootstrap_addr and publishes
    destination memory information derived from register().
    """

    request_id: str
    transfer_id: str
    block_ids: BlockIds
    bootstrap_addr: str
    remote_engine_id: str
    remote_dp_rank: int


@dataclass(frozen=True)
class TransferResult:
    """Terminal result for one local send or recv, identified by request_id.

    error=None indicates success. A send result means source KV is no longer
    being read; a recv result means destination KV is no longer being written.
    Only a successful recv guarantees all requested destination KV is usable.
    Failed recv block IDs are exposed separately through take_errors().
    """

    request_id: str
    error: str | None = None


@dataclass
class TransferPollResult:
    sends: list[TransferResult] = field(default_factory=list)
    recvs: list[TransferResult] = field(default_factory=list)


# =========================
# KVTransfer Interface
# =========================


class KVTransfer(ABC):
    """D-initiated P-D direct transfer, implemented by P writing into D.

    P: register -> prefill completes -> send(ready source KV) -> poll().
    D: register -> recv(destination + bootstrap) -> poll(recv completion).
    send() and the remote recv request may arrive in either order. Transfer
    starts only after send() publishes ready KV and a receiver request is present.
    P writes the requested KV, then notifies D of completion or failure.

    request_id identifies local work and results, without a task handle.
    transfer_id matches P and D requests, whose local request IDs may differ.
    It is a unique cross-worker attempt ID supplied by the connector; retries
    use a new transfer_id so delayed messages cannot match new block ownership.
    It is not a backend-generated task ID requiring a result-to-request map.

    At most one uncollected send and one uncollected recv per local request;
    duplicate submissions of the same kind raise ValueError. Events are
    immutable metadata snapshots. Layout, topology, endpoint registration,
    protocol timeouts and peer discovery caching belong to backend setup.
    The backend validates peer identity, layout and block-table compatibility and
    derives TP/PP pairing and complete destination coverage before success.
    Callers serialize public calls; the backend synchronizes background work.

    This interface owns transfer and memory-access fences. Request scheduling,
    block allocation, persistence and cache lookup belong to the connector
    and KVCachePool. There is no request-end notification or public state machine.
    """

    @abstractmethod
    def register(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        """Register local KV memory on both P and D before send/recv.

        On P, also expose the worker control endpoint through bootstrap.
        Memory descriptors and transport addresses are backend-owned; callers
        only provide block IDs. Tensors stay alive until close(). Re-registering
        while transfers are pending is unsupported.
        """
        pass

    @abstractmethod
    def send(self, send_event: SendEvent) -> None:
        """Publish ready source KV without waiting for a receiver or I/O.

        Calling send() marks this source ready; the caller ensures device writes
        are complete and visible to the transport before calling. P must
        retain source blocks while waiting and during writes. Fan-out to D
        workers is managed internally: publish one send result only after all
        expected targets are terminal and all source reads have stopped.
        A source with no receiver may expire according to backend policy;
        retire its transfer_id before reporting failure so late pulls cannot
        access released blocks.
        """
        pass

    @abstractmethod
    def recv(self, recv_event: RecvEvent) -> None:
        """Submit a receive request without waiting for discovery or I/O.

        Resolve matching P workers through bootstrap_addr, scoped by remote
        engine and DP identity. Send them the transfer_id, destination block
        table and local memory descriptors. P waits for send() and writes
        directly into D; D waits for every required worker's terminal response.
        Report success only after all destination KV is ready for local use,
        including any required device synchronization or layout conversion.
        """
        pass

    @abstractmethod
    def poll(self) -> TransferPollResult:
        """Drain terminal send/recv results without waiting for I/O.

        Return each result once, including failures. Source block release is
        authorized by send completion; destination reuse by recv completion.
        If blocks are used by other operations, those must also be fenced.
        Publish failed recv block IDs with their result and drain take_errors()
        in the same polling cycle before block/request reuse.
        """
        pass

    @abstractmethod
    def take_errors(self) -> set[int]:
        """Drain invalid physical local block IDs from polled recv failures.

        Non-blocking. Report unusable blocks from RecvEvent.block_ids; if partial
        coverage cannot be proved, invalidate all listed destination blocks.
        Preserve the existing local prefix, which is not in that table. Send failures
        do not invalidate local source KV. The connector reports these IDs
        through get_block_ids_with_load_errors() for scheduler recovery.
        """
        pass

    @abstractmethod
    def preempt(self, request_id: str) -> None:
        """Cancel and fence the request's local send/recv before block reuse.

        Retire its transfer_id(s), reject delayed requests, cancel queued work,
        and fence active transfers. On D, remote P writes must stop before
        returning; a local timeout or discarded response alone is insufficient.
        May block if safe cancellation is unavailable. On return no transfer
        accesses the request's local blocks; pending results, block errors and
        local bookkeeping are cleared. Calls with no tracked request are safe.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Stop accepting work and fence local reads and remote writes.

        Notify pending peers as needed and release registered memory/transport
        resources only after transfers stop accessing them. Idempotent;
        subsequent send/recv submissions raise RuntimeError.
        """
        pass
