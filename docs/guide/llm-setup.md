# LLM setup

KernelFoundry calls an LLM to generate and improve kernels. Everything else — the evolutionary
loop, the program database, sampling of parents and inspirations, optimization-aware prompting —
works the same regardless of which of these you use.

## API key (default)

Set the provider and model under `inference` and export the matching key:

```yaml
inference:
  provider: openai   # or anthropic
  model: gpt-5        # any model your provider offers; passed straight through
```

```bash
export OPENAI_API_KEY=...      # for provider: openai
export ANTHROPIC_API_KEY=...   # for provider: anthropic
```

Billed per token. Any OpenAI-compatible endpoint can be targeted with `base_url`. See "LLM
inference" in [Config parameters](config-parameters.md) for the full schema, including how to
ensemble several models.

## Claude subscription (no API key)

`provider: claude_cli` runs generation through the Claude Code CLI instead of the Anthropic API,
authenticated against a Claude subscription rather than billed per token.

Install the [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) and sign in once:

```bash
npm install -g @anthropic-ai/claude-code
claude   # sign in, then exit
```

Then select it like any other provider:

```bash
python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom \
    job_name=my_experiment gpu_arch=bmg language=SYCL \
    inference.provider=claude_cli inference.model=opus
```

`model` takes an alias such as `opus` or `sonnet`, or a dated model id to pin one. `disallow_tools`,
`max_turns` and `effort` are settable the same way (other providers ignore them):

```bash
... inference.provider=claude_cli inference.model=opus inference.effort=high
```

`timeout`, `max_retry` and `claude_exe` have no generic field, since their sensible defaults
differ per provider — set those by copying the full config in
[`configs/inference/claude_cli.yaml`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/configs/inference/claude_cli.yaml)
under `inference:` in a task's `config.yaml`.

### Differences from the API path

- **No temperature.** The CLI exposes no sampling controls; it's accepted and ignored. Branch
  diversity instead comes from each branch being a separate CLI session with a differing prompt.
- **One completion per invocation.** `num_completions: 4` runs the CLI four times in sequence.
- **No output ceiling.** `max_tokens` is recorded for reporting only.
- **Tools are discouraged, not disabled.** The CLI is a coding agent; the server tells it to
  answer from the prompt alone and passes `--disallowedTools`, but that flag is not a hard
  guarantee (it's ignored under `permissions.defaultMode: bypassPermissions`). The containment
  that does hold is the working directory: the subprocess runs in a throwaway temp directory, so
  a stray write lands there and is discarded.

If `ANTHROPIC_API_KEY` is set, the CLI would otherwise prefer it over your subscription login and
bill the API. `ClaudeCLIServer` strips `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_BASE_URL` and `ANTHROPIC_API_URL` from the CLI subprocess so this can't happen by
accident — a key in your own shell still affects the CLI when you run it yourself, just not this
subprocess.
