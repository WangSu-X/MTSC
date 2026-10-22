# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Basic wire schema of the supplied workspace Mooncake Connector.

Wire-compatibility integration tests also import the real native classes.
"""

from enum import IntEnum

import msgspec

ReqId = str
TransferId = str


class MooncakeXferMetadata(msgspec.Struct, omit_defaults=True):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[str, tuple[str, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]


class MooncakeXferResponseStatus(IntEnum):
    FINISH = 0
    CONTINUE = 1
    ERROR = 2


class MooncakeXferResponse(msgspec.Struct, omit_defaults=True):
    status: MooncakeXferResponseStatus
    ok_reqs: list[str] | None = None
    err_reqs: list[str] | None = None
    err_msg: str | None = None
