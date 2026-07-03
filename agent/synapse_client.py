import asyncio

import httpx
from dotenv import load_dotenv

load_dotenv()


class SynapseClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Api-Key": api_key},
            timeout=timeout,
        )
        self._sessions: dict[str, int] = {}
        self._bootstrap_lock = asyncio.Lock()

    async def _ensure_session(self, agent_id: str) -> int:
        if agent_id in self._sessions:
            return self._sessions[agent_id]
        async with self._bootstrap_lock:
            if agent_id in self._sessions:
                return self._sessions[agent_id]
            response = await self._client.post(f"/api/v1/agents/{agent_id}/bootstrap")
            response.raise_for_status()
            self._sessions[agent_id] = response.json()["activeSession"]
        return self._sessions[agent_id]

    async def append_thought(
        self,
        agent_id: str,
        parent_slot: int | None,
        state_hash: int,
        success_score: float,
    ) -> int:
        """Append a thought node to the agent's memory and return its thoughtId."""
        session_id = await self._ensure_session(agent_id)
        body: dict = {
            "stateHash": state_hash,
            "successScore": success_score,
            "sessionId": session_id,
        }
        if parent_slot is not None:
            # parentId omitted for root thoughts: schema minimum is 0, no null slot
            body["parentId"] = parent_slot
        response = await self._client.post(
            f"/api/v1/agents/{agent_id}/thoughts", json=body
        )
        response.raise_for_status()
        return response.json()["thoughtId"]

    async def get_best_next(self, agent_id: str, state_hash: int) -> dict | None:
        """Return the highest-salience next thought for this state, or None if no history."""
        session_id = await self._ensure_session(agent_id)
        response = await self._client.get(
            f"/api/v1/agents/{agent_id}/thoughts/best-next",
            params={"stateHash": state_hash, "sessionId": session_id, "fromSlot": 0},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return None if not data.get("found") else data

    async def get_stats(self, agent_id: str) -> dict:
        """Return raw memory stats for the agent (fillPercent, writeHead, usedSlots, capacity)."""
        response = await self._client.get(f"/api/v1/agents/{agent_id}/memory/stats")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
