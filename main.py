import asyncio
import cognee

async def test():
    await cognee.prune.prune_data()
    await cognee.remember("The SQL query SELECT * FROM customers returns all customer rows.")
    results = await cognee.recall("customers")
    print(results)

asyncio.run(test())