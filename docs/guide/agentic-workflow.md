# Optimize a kernel with a coding agent

This is the shortest path from "this kernel is slow" to a faster one that still passes your
tests, and the one to start with. You describe the kernel in a sentence; a coding agent does
the packaging, validation and integration, and checks with you before the two steps that
matter.

The alternative, authoring the package yourself and running the search from the command line,
is covered in [Quickstart](quickstart.md) and gives you more direct control. Everything the
agent produces is an ordinary task package, so you can switch to the manual path at any
point.

## What you need

- **KernelFoundry with the MCP server**, which is how the agent builds and benchmarks:

  ```bash
  pip install 'kernelfoundry[mcp,algo]'
  ```

- **An MCP-capable coding agent**. GitHub Copilot, Claude Code and Cursor all work.
- **A GPU and its toolchain**. See the
  [README](https://github.com/isl-org/kernelfoundry#requirements).

Register the server with your editor. `python -m kernelfoundry.mcp_server create_config`
generates a client config interactively. Full setup, including the choice between running
locally and submitting to a server, is in the
[MCP server reference](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/mcp_server/README.md).

:::{tip}
Point your agent at
[`AGENTS.md`](https://github.com/isl-org/kernelfoundry/blob/main/AGENTS.md) as well. It is the
workflow contract the agent follows: the constraints, the checkpoints and the definition of
done. Many agents discover the file on their own.
:::

## Start a session

Paste this, filling in the two required parts. Everything commented out is optional; the agent
infers it or asks:

```
Optimize <function> in <file path or upstream URL> for GPU with KernelFoundry.
# Optional (the agent will infer or ask if omitted):
# Language: <SYCL | OpenCL | CUDA>
# Benchmark shapes: <e.g. from upstream tests, or specific sizes>
# Build: <command>   Test: <command>   Env: <e.g. source /opt/intel/oneapi/setvars.sh>
# Constraints: <e.g. preserve macro-driven runtime flexibility>
```

A plain sentence works too. The agent asks for anything critical it cannot find.

### A worked example

Optimizing a real kernel from an upstream project:

```
Optimize the reusable_vectorized layer-norm OpenCL kernel from oneDNN
(https://github.com/uxlfoundation/oneDNN/blob/main/src/gpu/intel/lnorm/reusable_vectorized.cl).
Preserve the macro-driven runtime flexibility. Pull benchmark shapes from oneDNN's lnorm
tests, plus a few representative coverage shapes.
```

From that the agent freezes a faithful reference from upstream, gathers benchmark shapes
(recording which came from real tests and which are for coverage), scaffolds the package,
writes its `USER_INSTRUCTIONS`, validates locally, and stops to ask before submitting.

## What to expect

**It works autonomously and tells you what it chose.** It discovers the language, the
reference behaviour, benchmark shapes from your tests and call sites, and your build and test
commands by reading the codebase. Where it has to pick, it picks a sensible default and states
it rather than asking you to fill in a form.

**It writes the per-task guidance for you.** The `USER_INSTRUCTIONS` block in the package is
authored by the agent from your conversation. You edit it only to correct or add something,
for instance to name an
[optimization strategy](optimization-strategies.md) you already know pays off.

**It stops at exactly two checkpoints:**

1. **Before submitting a job**, because that consumes compute and time. You get a summary of
   the package and its assumptions, then a go/no-go.
2. **Before integrating** the result, because that edits your source.

Everything between those two points runs on its own. After each phase it reports what it did,
the evidence, what it validated, and what it assumed.

**What it produces** is an ordinary task package (see
[Anatomy of a task package](task-package.md)), so nothing here is a black box. If a run
goes badly, [Understanding results](understanding-results.md) explains how to read the graph
and work out why.

## Integrate and verify the result

Getting a faster kernel out of the loop is not the end. The number that matters is the
end-to-end improvement in your application, not the isolated kernel.

Ask the agent to continue once you have picked a kernel from the results, and it will merge
the optimized code back into your original implementation, preserving its semantics and
generality, and surface any trade-off for your review rather than changing behaviour quietly.
Then rebuild the full codebase, run its tests, and re-run your own benchmarks.

:::{important}
Keep the reference implementation general. The optimized kernel is tuned for the benchmark
shapes it was given, so before you ship it, confirm the edge cases and input shapes your
software actually relies on still work.
:::

## Next steps

- **Tune a run that has plateaued**: [Optimization strategies](optimization-strategies.md)
  gives you a line to paste into the agent for each one.
- **Understand the package** the agent built: [Anatomy of a task package](task-package.md)
  and [Writing tests](writing-tests.md).
- **Take manual control**: [Quickstart](quickstart.md) runs the shipped example end to end
  from the command line, with no LLM API key.
