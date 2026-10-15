# Working with an AI assistant (phase 5)

`oql mcp` runs the lab as a [Model Context Protocol](https://modelcontextprotocol.io) server. Any MCP client can
connect: Claude Desktop, Claude Code, or an open-source client with a local model. The assistant gets a fixed set
of tools and nothing else.

| The assistant can | Only a person can |
|---|---|
| Read the board, cards, datasets and quality checks | Freeze a card (fingerprint the rules) |
| Propose an idea card (a mechanism is required) | Run the gate |
| Write the pre-registration (the pass rule is fixed text) | Run the one-shot holdout |
| Declare the experiment from structured fields (no code) | Deploy paper trading |
| Explore one strategy at a time on development data | Archive or revise cards |
| Run a quick exploratory permutation check | |
| Ask for review with a summary | |

**Why it is built this way.** An AI can try strategies far faster than a person, which makes overfitting faster
too. So every exploration is logged to the card and the ledger, the freeze records how many happened, and the gate
report prints "Exploratory evaluations before the freeze" next to the result. The AI never writes Python (an
experiment is rendered from validated fields), so it cannot run arbitrary code on your machine or loosen a test.
Exploration only ever sees development data: the cutoff is each dataset's 70% point and never later than the
split of the card's experiment.

## Connect Claude Desktop (macOS)

**1. Give the assistant's copy of the lab a home outside Desktop, Documents and Downloads.** macOS protects those
folders, and Claude Desktop starts MCP servers in a way macOS does not credit to Claude's own permissions, so a
server kept there stops at once with "Operation not permitted" (see `~/Library/Logs/Claude/mcp-server-open-quant-lab.log`).
For example:

```bash
mkdir -p ~/quant
python3 -m venv ~/quant/.venv
~/quant/.venv/bin/pip install /path/to/open-quant-lab    # a copy, not -e: the code must live outside too
# keep the catalog (OQL_CATALOG) and the lab (OQL_LAB) under ~/quant as well
```

After changing the lab's code, run the same `pip install` again to update the assistant's copy.

**2. Add the server to the config.** Open the Claude menu in the macOS menu bar, then Settings…, Developer, Edit Config
(the file is `~/Library/Application Support/Claude/claude_desktop_config.json`). If the file already has settings, put
`mcpServers` inside the same outer braces: two JSON objects one after the other make Claude skip the whole file.

```json
{
  "mcpServers": {
    "open-quant-lab": {
      "command": "/Users/YOU/quant/.venv/bin/oql",
      "args": ["mcp"],
      "env": {
        "OQL_LAB": "/Users/YOU/quant/lab",
        "OQL_CATALOG": "/Users/YOU/quant/catalog"
      }
    }
  }
}
```

**3. Quit Claude completely and reopen it.** In a new chat, + then Connectors then Manage connectors should list
`open-quant-lab` with 11 tools. Then try: *"Look at my lab. Propose one idea with a clear mechanism that the data could
test, set up the experiment, explore at most five strategies, and ask me for review."*

When it asks for review, the card shows "AI asks for review" on the dashboard. Read its pre-registration, then
decide: `oql card freeze N` and `oql card gate N`, or send it back.

**Give the assistant no other way in.** The limits above hold for an assistant that reaches the lab only through
`oql mcp`. A chat with the lab's folder attached can also edit files or run `oql` from a shell, and then only its
own judgment stands between it and a freeze. Keep your real lab in a folder that no chat has attached.

## Test the server by hand
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"me","version":"1"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | oql mcp
```
