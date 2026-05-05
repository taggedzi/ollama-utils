# Privacy Notes

This project is a local utility for managing Ollama model libraries. It does not include telemetry, analytics, or any built-in outbound reporting service. Its network and local-data behavior is limited to the Ollama endpoints and files described below.

## What Is Sent To Ollama

### Update workflows

`tz_ollama_utils_update.py` and the GUI update actions use the local `ollama` CLI:

- `ollama list` to enumerate installed models
- `ollama pull <model>` to update installed models in place

These commands talk to the user's configured Ollama installation. This project does not add its own extra payloads.

### Test and inventory workflows

`tz_ollama_utils_test.py` and the GUI test tab can send the following requests to an Ollama HTTP API:

- `GET /api/tags` to list available models
- `POST /api/show` with `{"model": "<model-name>"}` to fetch model metadata
- `POST /api/generate` with `{"model": "<model-name>", "prompt": "Respond with exactly OK.", "stream": false}` to smoke-test completion-capable models
- `POST /api/embed` with `{"model": "<model-name>", "input": "test"}` to smoke-test embedding-capable models

If the Ollama API is unavailable, some test behavior falls back to local CLI execution such as `ollama run <model> "Respond with exactly OK."`.

### Search and discover workflows

The Search & Discover GUI cache refresh uses the Ollama HTTP API:

- `GET /api/tags`
- `POST /api/show` with `{"model": "<model-name>"}`

### Remote Ollama hosts

Remote Ollama API hosts are blocked by default.

- CLI usage requires `--allow-remote`.
- GUI usage requires an explicit user confirmation for each remote API base URL.
- Non-loopback remote API URLs must use `https`.

If you point the tool at a remote Ollama server, that server operator can see the model names and test payloads listed above.

## What Is Cached Locally

The Search & Discover feature stores a JSON cache at `~/.tz_ollama_utils/model_cache.json`.

By default, the cache stores a minimized model summary:

- API base URL
- model name
- digest
- size
- modified timestamp
- family, families, format, quantization, parameter size
- parent model
- capabilities
- context window
- whether a system prompt exists
- a short license summary derived from the first line of the reported license text
- cache timestamp

By default, the cache does **not** store:

- system prompt text
- template text
- modelfile text
- parameters text
- full license text

If `TZ_OLLAMA_UTILS_PERSIST_MODEL_TEXT=1` is set, the cache may also store the Ollama-provided `license`, `modelfile`, `template`, `system`, and `parameters` fields verbatim. Only enable that setting if you intentionally want those long text fields persisted on disk.

## What Lands In Reports

The test workflow writes a YAML report to `model_report.yaml` by default, or to the path selected by the user.

Reports can include:

- run timestamp
- effective timeout and VRAM policy
- API base URL summary
- warnings, skips, and failures
- one record per discovered model
- model overview data such as name, family, capabilities, digest, installed size, and context window
- test-attempt records including command labels, prompt labels, prompt values, result status, error summaries, and output previews

By default, reports omit preview text for long metadata fields such as:

- license text
- system prompt
- template
- modelfile
- parameters

Those previews are only written when `--include-metadata-previews` is passed.

The report sanitizer removes absolute filesystem paths from exported strings and redacts remote API URLs. Loopback API URLs such as `http://127.0.0.1:11434/api` are preserved.

## What This Project Does Not Do

- No built-in telemetry
- No cloud sync
- No automatic upload of reports or caches
- No bundled crash reporting

You remain responsible for where you store generated reports and whether the discovered model names, metadata, prompts, or output previews are appropriate to retain or share.
