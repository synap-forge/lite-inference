#!/usr/bin/env python3
"""Benchmark the lite-inference OpenAI-compatible server."""

import argparse
import base64
import json
import mimetypes
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.request
import urllib.error


def build_image_url(image):
    """Turn an --image value into an OpenAI image_url 'url' string.

    A local file path is read and base64-encoded into a data URL so the
    server doesn't need filesystem access. http(s)://, file://, and existing
    data: URLs are passed through unchanged.
    """
    if image.startswith(("http://", "https://", "file://", "data:")):
        return image
    if not os.path.isfile(image):
        raise FileNotFoundError(f"image not found: {image}")
    mime = mimetypes.guess_type(image)[0] or "image/jpeg"
    with open(image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def build_content(prompt, image_urls):
    """Return the OpenAI message content: a plain string, or a multimodal array."""
    if not image_urls:
        return prompt
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def resolve_model(base_url, model):
    """If model is 'auto', fetch /v1/models and return the first available model id."""
    if model != "auto":
        return model
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read())
            return data["data"][0]["id"]
    except Exception:
        return "default"


def chat_completion(base_url, content, stream, model="auto", api_key=None, timeout=120):
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "stream": stream,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - t0
                            # approximate: split on spaces as a word-token proxy;
                            # server doesn't send per-token usage in stream chunks
                            total_tokens += max(1, len(delta.split()))
                    except (json.JSONDecodeError, KeyError):
                        pass
            else:
                body = json.loads(resp.read())
                first_token_time = None
                total_tokens = body.get("usage", {}).get("completion_tokens", 0)
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        try:
            body = json.loads(e.read())
            msg = body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return {"error": f"HTTP {e.code}: {msg}", "latency": elapsed}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"error": str(e), "latency": elapsed}

    elapsed = time.perf_counter() - t0
    if stream and total_tokens == 0:
        return {"error": "empty stream (no content tokens received)", "latency": elapsed}
    return {
        "latency": elapsed,
        "ttft": first_token_time,
        "tokens": total_tokens,
        "tps": total_tokens / elapsed if elapsed > 0 and total_tokens > 0 else None,
    }


def run_benchmark(base_url, prompt, concurrency, total_requests, stream, api_key,
                  model="auto", images=None):
    model = resolve_model(base_url, model)
    image_urls = [build_image_url(i) for i in (images or [])]
    content = build_content(prompt, image_urls)
    print(f"\nTarget:       {base_url}")
    print(f"Model:        {model}")
    print(f"Concurrency:  {concurrency}")
    print(f"Requests:     {total_requests}")
    print(f"Stream:       {stream}")
    print(f"Prompt:       {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if images:
        print(f"Images:       {len(images)} ({', '.join(os.path.basename(i) for i in images)})")
    print("-" * 60)

    results = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(chat_completion, base_url, content, stream, model, api_key)
            for _ in range(total_requests)
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            status = f"error: {r['error']}" if "error" in r else f"{r['latency']:.2f}s"
            print(f"  [{i:3d}/{total_requests}] {status}")

    wall = time.perf_counter() - t_start

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    latencies = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    tps_list = [r["tps"] for r in ok if r.get("tps") is not None]

    print("\n--- Results ---")
    print(f"Total wall time:  {wall:.2f}s")
    print(f"Successful:       {len(ok)}/{total_requests}")
    print(f"Errors:           {len(errors)}")

    if latencies:
        print(f"\nLatency (s):")
        print(f"  min    {min(latencies):.3f}")
        print(f"  mean   {statistics.mean(latencies):.3f}")
        print(f"  median {statistics.median(latencies):.3f}")
        print(f"  p95    {sorted(latencies)[int(len(latencies)*0.95)]:.3f}")
        print(f"  max    {max(latencies):.3f}")

    if ttfts:
        print(f"\nTime to first token (s):")
        print(f"  min    {min(ttfts):.3f}")
        print(f"  mean   {statistics.mean(ttfts):.3f}")
        print(f"  max    {max(ttfts):.3f}")

    if tps_list:
        print(f"\nTokens/sec per request:")
        print(f"  min    {min(tps_list):.1f}")
        print(f"  mean   {statistics.mean(tps_list):.1f}")
        print(f"  max    {max(tps_list):.1f}")
        total_tokens = sum(r.get("tokens", 0) for r in ok)
        print(f"\nThroughput:       {total_tokens / wall:.1f} tokens/sec (aggregate)")

    if errors:
        print("\nError details:")
        for r in errors:
            print(f"  {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark lite-inference server")
    parser.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--prompt", default="Tell me a short joke.", help="Prompt to send")
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="Parallel requests")
    parser.add_argument("-n", "--requests", type=int, default=10, help="Total requests")
    parser.add_argument("--stream", action="store_true", help="Use SSE streaming")
    parser.add_argument("--api-key", default=None, help="Bearer API key if required")
    parser.add_argument("--model", default="auto", help="Model id (default: auto-detect from /v1/models)")
    parser.add_argument("--image", action="append", default=None,
                        help="Image to attach (local path, http(s)/file/data URL). "
                             "Repeat for multiple images. Requires a multimodal model.")
    args = parser.parse_args()

    run_benchmark(
        base_url=args.url,
        prompt=args.prompt,
        concurrency=args.concurrency,
        total_requests=args.requests,
        stream=args.stream,
        api_key=args.api_key,
        model=args.model,
        images=args.image,
    )


if __name__ == "__main__":
    main()
