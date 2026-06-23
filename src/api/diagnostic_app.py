from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="AgentCore Diagnostic Runtime", version="1.0.0")


@app.middleware("http")
async def log_everything(request: Request, call_next):
    body = await request.body()
    body_text = body.decode("utf-8", errors="replace")

    print("===================================")
    print("DIAG REQUEST", request.method, request.url.path)
    print("DIAG BODY", body_text)

    response = await call_next(request)

    print("DIAG RESPONSE", response.status_code)
    print("===================================")
    return response


@app.get("/ping")
def ping() -> dict[str, int | str]:
    return {"status": "Healthy", "time_of_last_update": int(time.time())}


@app.post("/invocations")
async def invocations(request: Request) -> dict[str, int | str | bool]:
    body = await request.body()
    return {
        "ok": True,
        "path": "/invocations",
        "method": request.method,
        "received_bytes": len(body),
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def catch_all(path: str, request: Request):
    body = await request.body()
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "path": f"/{path}",
            "method": request.method,
            "received_bytes": len(body),
        },
    )
