import asyncio
import os
import sys
import time
import traceback

# Ensure project root is on sys.path so sibling modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langmem import create_prompt_optimizer

def _build_local_langchain_chat_model():  # pragma: no cover - heavy external dependency
    """Build a LangChain Chat model instance similar to mem_bank._build_langchain_chat_model

    We keep this local to the script so we don't have to modify `mem_bank.py`.
    """
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies: ensure langchain_openai and pydantic are installed"
        ) from e

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


# Example conversation + feedback used across optimizer runs
conversation = [
    {"role": "user", "content": "How do I write a bash script?"},
    {"role": "assistant", "content": "Let me explain bash scripting..."},
]
feedback = "Response should include a code example"
trajectories = [(conversation, {"feedback": feedback})]

# System prompt must include the word 'json' to satisfy some backends when using json_object
system_prompt = (
    "You are a coding assistant. When appropriate, format your output as JSON and return an object"
    " with a single 'prompt' field containing the improved prompt."  # keeps it explicit
)


async def run_optimizer(optimizer, name: str, prompt: str):
    """Run an optimizer callable, measure runtime, and return (success, output, elapsed, error)

    We treat optimizer as an async callable that accepts (trajectories, prompt).
    """
    start = time.perf_counter()
    try:
        # Try the common usage first: await optimizer(trajectories, prompt)
        try:
            result = await optimizer(trajectories, prompt)
        except TypeError:
            # Some Runnables expose .ainvoke and expect a single dict arg
            if hasattr(optimizer, "ainvoke"):
                result = await optimizer.ainvoke({"trajectories": trajectories, "prompt": prompt})
            else:
                raise

        elapsed = time.perf_counter() - start

        # Normalize result shapes into a string prompt when possible
        normalized = None
        if isinstance(result, str):
            normalized = result
        elif isinstance(result, dict):
            # common keys that might hold the new prompt
            for k in ("new_prompt", "prompt", "optimized_prompt", "result"):
                if k in result:
                    normalized = result[k]
                    break
            if normalized is None:
                # fallback to JSON-ish string
                try:
                    import json

                    normalized = json.dumps(result)
                except Exception:
                    normalized = str(result)
        else:
            normalized = str(result)

        return True, normalized, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        tb = traceback.format_exc()
        # Fallback: build a simple deterministic improved prompt so the demo can continue
        fallback = (
            "You are a coding assistant that always includes a concise bash code example when asked."
        )
        return False, fallback, elapsed, tb


async def main():
    try:
        model = _build_local_langchain_chat_model()
    except Exception:
        traceback.print_exc()
        print("Failed to build model. Aborting.")
        return

    # Create three optimizers with small reflection settings to keep runtime small
    optimizers = []

    # Prompt memory (fastest, single call)
    try:
        pm = create_prompt_optimizer(model, kind="prompt_memory")
        optimizers.append(("prompt_memory", pm))
    except Exception:
        traceback.print_exc()
        print("Failed to create prompt_memory optimizer — skipping it.")

    # Metaprompt (configurable reflection steps)
    try:
        meta_cfg = {"min_reflection_steps": 1, "max_reflection_steps": 2, "metaprompt": "Propose concise prompt updates."}
        mp = create_prompt_optimizer(model, kind="metaprompt", config=meta_cfg)
        optimizers.append(("metaprompt", mp))
    except Exception:
        traceback.print_exc()
        print("Failed to create metaprompt optimizer — skipping it.")

    # Gradient (uses think + critique cycles)
    try:
        grad_cfg = {"min_reflection_steps": 1, "max_reflection_steps": 2, "gradient_prompt": "What to improve?", "metaprompt": "Apply the improvement to the prompt."}
        gp = create_prompt_optimizer(model, kind="gradient", config=grad_cfg)
        optimizers.append(("gradient", gp))
    except Exception:
        traceback.print_exc()
        print("Failed to create gradient optimizer — skipping it.")

    results = []

    # Run each optimizer sequentially (not parallel) to measure isolated runtimes
    for name, opt in optimizers:
        print(f"\nRunning optimizer: {name}")
        # prompt_memory implementations sometimes expect a simpler prompt and
        # can return different shapes; avoid forcing a JSON response for it.
        if name == "prompt_memory":
            # Request a JSON object with a 'new_prompt' key so the optimizer's
            # internal post-processing (which expects 'new_prompt') can succeed,
            # and include the word 'json' to satisfy backend validation.
            prompt_to_use = (
                "You are a coding assistant. Return a JSON object with a single key 'new_prompt'"
                " whose value is the improved prompt. Include the word 'json' in your response so the backend accepts the request."
            )
        else:
            prompt_to_use = system_prompt

        success, output, elapsed, error = await run_optimizer(opt, name, prompt_to_use)
        if success:
            print(f"[{name}] success in {elapsed:.2f}s")
            print("Result:")
            print(output)
        else:
            print(f"[{name}] failed in {elapsed:.2f}s — falling back (traceback below):")
            print(error)
            print("Fallback result:")
            print(output)
        results.append((name, success, output, elapsed))

    # Summary
    print("\n=== SUMMARY ===")
    for name, success, output, elapsed in results:
        status = "OK" if success else "FALLBACK"
        print(f"- {name}: {status} — {elapsed:.2f}s — prompt: {str(output)[:120]}{'...' if len(str(output))>120 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
