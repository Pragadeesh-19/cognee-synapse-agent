import os

import httpx
import pytest
from dotenv import load_dotenv

from agent.synapse_client import SynapseClient

load_dotenv()


@pytest.fixture(scope="function")
async def live_synapse():
    base_url = os.environ["SYNAPSE_URL"]
    api_key = os.environ["SYNAPSE_API_KEY"]
    try:
        async with httpx.AsyncClient(timeout=2.0) as probe:
            await probe.get(
                f"{base_url}/api/v1/agents/0/memory/stats",
                headers={"X-Api-Key": api_key},
            )
    except httpx.ConnectError:
        pytest.skip("Synapse unreachable")

    client = SynapseClient(base_url=base_url, api_key=api_key)
    yield client
    await client.aclose()
