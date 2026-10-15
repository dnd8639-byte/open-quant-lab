"""`oql mcp`: the lab as a Model Context Protocol server, so an AI assistant (Claude Desktop, or any MCP client)
can work inside it with exactly the tools in agent_tools.py and nothing else.

Transport: JSON-RPC 2.0 over stdin/stdout, one message per line (the MCP stdio transport). Standard library only.
Anything the lab prints is sent to stderr, because stdout belongs to the protocol.
"""
from __future__ import annotations

import contextlib
import json
import sys
import traceback

from . import __version__
from .agent_tools import TOOLS, AgentError, call

SUPPORTED = ["2025-06-18", "2025-03-26", "2024-11-05"]
INSTRUCTIONS = (
    "You are a research assistant in Open Quant Lab, a strategy-research pipeline built around tests that cannot be "
    "gamed. Start with lab_overview. You may propose ideas, write pre-registrations and experiments, and explore "
    "development data. Every exploration is counted and reported next to the final result, so explore with purpose: "
    "state a mechanism first, test few things, and prefer ideas whose mechanism predicts the result. You cannot "
    "freeze a card, run the gate or the holdout, or deploy paper trading: when a card is ready, call request_review "
    "and a person decides. Never present an exploratory number as a finding.")


def _result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle(msg: dict):
    """One JSON-RPC message in, one response out (None for notifications)."""
    method, req_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if req_id is None:                                     # notifications (e.g. notifications/initialized)
        return None
    if method == "initialize":
        v = params.get("protocolVersion")
        return _result(req_id, {"protocolVersion": v if v in SUPPORTED else SUPPORTED[0],
                                "capabilities": {"tools": {"listChanged": False}},
                                "serverInfo": {"name": "open-quant-lab", "version": __version__},
                                "instructions": INSTRUCTIONS})
    if method == "ping":
        return _result(req_id, {})
    if method == "tools/list":
        return _result(req_id, {"tools": [{"name": n, "description": d, "inputSchema": s} for n, (_, d, s) in TOOLS.items()]})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                out = call(name, args)
            return _result(req_id, {"content": [{"type": "text", "text": json.dumps(out, indent=1, default=str)}], "isError": False})
        except (AgentError, KeyError, ValueError, TypeError, RuntimeError) as e:
            return _result(req_id, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
    return _error(req_id, -32601, f"method not found: {method}")


def serve(stdin=None, stdout=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_error(None, -32700, "parse error")) + "\n"); stdout.flush()
            continue
        try:
            resp = handle(msg)
        except Exception:                                  # never die on one bad request
            traceback.print_exc(file=sys.stderr)
            resp = _error(msg.get("id"), -32603, "internal error")
        if resp is not None:
            stdout.write(json.dumps(resp, default=str) + "\n")
            stdout.flush()
