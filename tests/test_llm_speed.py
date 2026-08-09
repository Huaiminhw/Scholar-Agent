import asyncio
import time

from real_llm_client import build_client


async def main():
    client = build_client()

    print("\n========== BARE LLM SPEED TEST ==========")
    print("开始调用模型...")

    start = time.perf_counter()

    try:
        response = await client.complete(
            messages=[
                {
                    "role": "user",
                    "content": "只回复OK，不要解释。"
                }
            ],
            temperature=0.0,
            response_schema=None,
        )

        elapsed = time.perf_counter() - start

        print("\n========== RESULT ==========")
        print("response:", response.content)

        print("\n========== SPEED ==========")
        print(f"time: {elapsed:.2f} s")

        print("\n========== TOKEN ==========")
        print("prompt_tokens:", response.usage.prompt_tokens)
        print("completion_tokens:", response.usage.completion_tokens)
        print("total_tokens:", response.usage.total_tokens)

        print("\n========== MODEL ==========")
        print("model:", response.model)
        print("request_id:", response.request_id)

    except Exception as e:
        elapsed = time.perf_counter() - start

        print("\n========== FAILED ==========")
        print("error:", repr(e))
        print(f"time before failure: {elapsed:.2f} s")

        raise


if __name__ == "__main__":
    asyncio.run(main())
