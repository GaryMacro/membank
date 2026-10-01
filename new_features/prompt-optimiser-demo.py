import asyncio
import os
import sys

# Ensure project root is on sys.path so sibling modules (like mem_bank) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langmem import create_prompt_optimizer
# from mem_bank import _build_langchain_chat_model

def _build_langchain_chat_model():  # pragma: no cover - heavy external dependency
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("QWEN_API_KEY (or DASHSCOPE_API_KEY) must be set for langmem backend")

    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
    model_name = os.getenv("QWEN_MODEL", "qwen-plus")

    return ChatOpenAI(  # type: ignore[call-arg]
        api_key=SecretStr(api_key.strip()),
        base_url=base_url.strip() if base_url else None,
        model=model_name.strip(),
        temperature=0.0,
        streaming=False,
    )
optimizer = create_prompt_optimizer(
    _build_langchain_chat_model(), kind="prompt_memory"
)



# Conversation with feedback about what could be improved
conversation = [
    {"role": "user", "content": "How do I write a bash script?"},
    {"role": "assistant", "content": "Let me explain bash scripting..."},
]
feedback = "Response should include a code example"

# Use the conversation and feedback to improve the prompt
trajectories = [(conversation, {"feedback": feedback})]

async def main():
    try:
        # Some LLM backends validate requests when a response_format of type
        # 'json_object' is used; they require the prompt/messages to contain
        # the word 'json' in some form. Include it here so the request is
        # accepted by those backends.
        system_prompt = (
            "You are a coding assistant. When appropriate, format your output as JSON."
        )
        better_prompt = await optimizer(trajectories, system_prompt)
        print(better_prompt)
    except Exception as e:
        # Print the full traceback and provide a sensible fallback so the demo script
        # runs even if the model call fails due to environment/configuration.
        import traceback

        traceback.print_exc()
        print("\nOptimizer call failed — showing fallback prompt:\n")
        # A reasonable fallback prompt to demonstrate expected output
        print("You are a coding assistant that always includes a concise code example when asked.")


if __name__ == "__main__":
    asyncio.run(main())
    # Output: 'You are a coding assistant that always includes...'