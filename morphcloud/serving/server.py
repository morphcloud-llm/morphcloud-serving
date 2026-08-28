"""MorphCloud-LLM control-plane API scaffold.

The GPU-backed vLLM data plane used by the experiments is not included in the
supplied artifact. This module exposes configuration and health state, and it
fails explicitly for generation instead of pretending the missing adapter is
functional.
"""
from __future__ import annotations

import argparse
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="MorphCloud-LLM", version="1.0")
STATE = {
    "model": None,
    "backend": None,
    "control_plane_ready": False,
    "generation_ready": False,
}


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64


@app.get("/health")
def health():
    return {"status": "ok" if STATE["control_plane_ready"] else "starting", **STATE}


@app.post("/generate")
def generate(req: GenerateRequest):
    del req
    raise HTTPException(
        status_code=501,
        detail=(
            "The GPU/vLLM data-plane adapter used by the experiments is not "
            "included in this artifact. Attach that adapter before enabling generation."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="MorphCloud-LLM control-plane scaffold")
    parser.add_argument("--model", default=os.getenv("MORPHCLOUD_MODEL", "meta-llama/Llama-2-7b-hf"))
    parser.add_argument("--checkpoint-backend", default=os.getenv("MORPHCLOUD_CHECKPOINT_BACKEND", "filesystem"))
    parser.add_argument("--checkpoint-path", default=os.getenv("MORPHCLOUD_CHECKPOINT_PATH", "/tmp/morphcloud-kv"))
    parser.add_argument("--predictor-weights", default=os.getenv("MORPHCLOUD_PREDICTOR_WEIGHTS", ""))
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--fallback-nodes", type=int, default=1)
    parser.add_argument("--preemption-threshold", type=float, default=0.7)
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    STATE.update(
        model=args.model,
        backend=args.checkpoint_backend,
        control_plane_ready=True,
        generation_ready=False,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
