# Docker quick start

KernelFoundry containers require a Linux host with Docker and Intel GPU drivers.

## Dev Container

Copy `docker/devcontainer` into the workspace containing your task packages, naming
it `.devcontainer`. Open that workspace in VS Code and select **Dev Containers:
Reopen in Container**. Set `KF_RENDER_GID` to the host render-node group id when it
differs from the default:

```bash
export KF_RENDER_GID="$(stat -c %g /dev/dri/renderD128)"
```

## Using the Docker image

We provide a simple wrapper script `kf` for composing the lengthy `docker run` commands.
To obtain the script copy the wrapper out of a published image, then run a task. 
Results are stored in
`./runs` on the host.

```bash
id=$(docker create ghcr.io/isl-org/kernelfoundry:latest)
docker cp "$id:/kf" ./kf
docker rm "$id"
chmod +x ./kf

./kf run task=tasks/example_custom task_origin=custom job_name=demo gpu_arch=bmg language=SYCL
./kf gui
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` before running, or place them in a
`.env` file in the current directory.

Prefix any `kf` command with `--dry-run` to print the Docker command without running it.

## Build the image

To build the image yourself from a repository checkout, use the same wrapper:

```bash
./docker/kf build
```

Use `KF_IMAGE=my-registry/kernelfoundry:tag ./docker/kf build` to build under a
different tag. When the local image does not exist, `kf run`, `gui`, `shell`, and
`exec` pull `KF_REMOTE_IMAGE` instead of building it.