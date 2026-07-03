import json

import httpx
import pytest

from agent.synapse_client import SynapseClient

BOOTSTRAP_RESPONSE = {"reloaded": False, "writeHead": 1, "activeSession": 1}
APPEND_RESPONSE = {"thoughtId": 7, "slotIndex": 1, "salienceScore": 0.5, "persisted": True}
STATS_RESPONSE = {
    "agentId": 0,
    "writeHead": 5,
    "usedSlots": 5,
    "capacity": 1048576,
    "fillPercent": 0.0,
    "wrapped": False,
}


def _make_client(*handlers) -> SynapseClient:
    recorded: list[httpx.Request] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        for handler in handlers:
            result = handler(request)
            if result is not None:
                return result
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(transport_handler)
    client = SynapseClient.__new__(SynapseClient)
    import asyncio
    client._client = httpx.AsyncClient(
        base_url="http://test",
        headers={"X-Api-Key": "test-key"},
        transport=transport,
    )
    client._sessions = {}
    client._bootstrap_lock = asyncio.Lock()
    client._recorded = recorded
    return client


def _bootstrap_handler(request: httpx.Request) -> httpx.Response | None:
    if request.method == "POST" and "/bootstrap" in request.url.path:
        return httpx.Response(200, json=BOOTSTRAP_RESPONSE)
    return None


def _append_handler(request: httpx.Request) -> httpx.Response | None:
    if request.method == "POST" and "/thoughts" in request.url.path and "/bootstrap" not in request.url.path:
        return httpx.Response(201, json=APPEND_RESPONSE)
    return None


def _best_next_found_handler(request: httpx.Request) -> httpx.Response | None:
    if request.method == "GET" and "best-next" in request.url.path:
        return httpx.Response(200, json={"found": True, "slot": 5, "score": 0.9})
    return None


def _best_next_not_found_handler(request: httpx.Request) -> httpx.Response | None:
    if request.method == "GET" and "best-next" in request.url.path:
        return httpx.Response(200, json={"found": False})
    return None


def _best_next_404_handler(request: httpx.Request) -> httpx.Response | None:
    if request.method == "GET" and "best-next" in request.url.path:
        return httpx.Response(404, json={"error": "not found"})
    return None


def _stats_handler(request: httpx.Request) -> httpx.Response | None:
    if request.method == "GET" and "memory/stats" in request.url.path:
        return httpx.Response(200, json=STATS_RESPONSE)
    return None


def _unauthorized_handler(request: httpx.Request) -> httpx.Response | None:
    return httpx.Response(401, json={"error": "unauthorized"})


async def test_append_thought_returns_thought_id():
    client = _make_client(_bootstrap_handler, _append_handler)
    result = await client.append_thought("0", None, 12345, 1.0)
    assert result == 7

    append_req = next(r for r in client._recorded if "thoughts" in r.url.path and "/bootstrap" not in r.url.path)
    body = json.loads(append_req.content)
    assert body["stateHash"] == 12345
    assert body["successScore"] == 1.0
    assert body["sessionId"] == 1
    assert "parentId" not in body


async def test_append_thought_includes_parent_id_when_provided():
    client = _make_client(_bootstrap_handler, _append_handler)
    await client.append_thought("0", 3, 12345, 1.0)

    append_req = next(r for r in client._recorded if "thoughts" in r.url.path and "/bootstrap" not in r.url.path)
    body = json.loads(append_req.content)
    assert body["parentId"] == 3


async def test_bootstrap_called_once_for_two_appends():
    client = _make_client(_bootstrap_handler, _append_handler)
    await client.append_thought("0", None, 11111, 1.0)
    await client.append_thought("0", None, 22222, 0.0)

    bootstrap_calls = [r for r in client._recorded if "/bootstrap" in r.url.path]
    assert len(bootstrap_calls) == 1


async def test_get_best_next_returns_dict_when_found():
    client = _make_client(_bootstrap_handler, _best_next_found_handler)
    result = await client.get_best_next("0", 99999)
    assert result is not None
    assert result["found"] is True
    assert result["slot"] == 5


async def test_get_best_next_returns_none_when_not_found():
    client = _make_client(_bootstrap_handler, _best_next_not_found_handler)
    result = await client.get_best_next("0", 99999)
    assert result is None


async def test_get_best_next_returns_none_on_404():
    client = _make_client(_bootstrap_handler, _best_next_404_handler)
    result = await client.get_best_next("0", 99999)
    assert result is None


async def test_get_stats_returns_parsed_dict():
    client = _make_client(_stats_handler)
    result = await client.get_stats("0")
    assert "fillPercent" in result
    assert result["fillPercent"] == 0.0


async def test_unauthorized_raises():
    client = _make_client(_unauthorized_handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_stats("0")
