#!/usr/bin/env bash
# Runs once when a devcontainer built from this image is created, as the unprivileged
# container user. It ships inside the image so that a project only needs to copy
# devcontainer.json, not this file. The oneAPI and virtualenv shell setup lives in the
# image itself, since writing to /etc/profile.d is not possible from here.
set -euo pipefail

# Set by the image; defaulted here only so that running this script by hand still works.
: "${KERNELFOUNDRY_HOME:=/opt/kernelfoundry}"

# Reaching the GPU takes more than --device=/dev/dri: the render node is owned by a group
# whose id is assigned per machine, and the container process has to carry that numeric id.
# A group name cannot stand in for it, because docker resolves --group-add against the
# container's own group file, where the host's id means nothing.
diagnose_gpu() {
    local gid held missing=()
    held=" $(id -G) "

    while read -r gid; do
        [[ -n "$gid" && "$held" != *" $gid "* ]] && missing+=("$gid")
    done < <(find /dev/dri -maxdepth 1 -name 'renderD*' -printf '%G\n' 2>/dev/null | sort -u)

    if [[ ! -e /dev/dri ]]; then
        echo "There is no /dev/dri here. Check that the host has an Intel GPU with a driver"
        echo "loaded, and that devcontainer.json passes --device=/dev/dri."
        return
    fi

    if ((${#missing[@]} == 0)); then
        echo "The render nodes are already reachable from here, so this looks like a driver"
        echo "problem rather than a permission one. Compare 'clinfo -l' against the host."
        return
    fi

    echo "This container does not belong to the group that owns the render node. Fix it in"
    echo "either of these ways, then run \"Dev Containers: Rebuild Container\":"
    echo
    echo "  - Edit .devcontainer/devcontainer.json and change the --group-add value to"
    echo "    ${missing[0]}. Simplest, and it stays with the project."
    echo
    echo "  - Or export KF_RENDER_GID=${missing[0]} on the host somewhere VS Code itself"
    echo "    inherits it, then restart VS Code. Setting it in the integrated terminal has"
    echo "    no effect, because the value is read by VS Code, not by the shell."
    if ((${#missing[@]} > 1)); then
        echo
        echo "This host uses more than one render group (${missing[*]}). KF_RENDER_GID covers"
        echo "the first; add the others as extra --group-add entries in devcontainer.json."
    fi
}

echo
if /opt/venv/bin/python - <<'PY'
import sys

import torch

print(f"torch {torch.__version__}")
names = [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]
if names:
    print("GPU: " + ", ".join(names))
sys.exit(0 if names else 1)
PY
then
    :
else
    echo "GPU: none visible."
    diagnose_gpu
fi

# Profiling reads GPU performance counters, which the host kernel withholds from non-root
# callers until these are relaxed. They reset on every host reboot and cannot be set from
# inside the container.
for knob in /proc/sys/dev/xe/observation_paranoid /proc/sys/dev/i915/perf_stream_paranoid; do
    if [[ -r "$knob" && "$(cat "$knob")" != "0" ]]; then
        echo
        echo "Profiling will fail until you run this on the HOST:"
        echo "    sudo sh -c 'echo 0 > $knob'"
    fi
done

cat <<EOF

KernelFoundry lives in $KERNELFOUNDRY_HOME. Your workspace is the working directory, so
results are written to ./runs next to your tasks and shared by every task in this folder.

If you do not have a task in your workspace to try, you can copy and run the example matmul
task in the container:
  cp -r $KERNELFOUNDRY_HOME/tasks/example_custom my_task
  python -m kernelfoundry.algorithm run task=my_task task_origin=custom \\
      job_name=first_try gpu_arch=<your_gpu_arch>

Set gpu_arch to match your Intel GPU, e.g., one of: lnl, bmg, ptl.

Start the UI:
  python -m kernelfoundry.gui                       # UI on forwarded port 8885
  # Open it via cmd/ctrl-click on localhost:8885 to open it in the integrated browser or
  # go to VS Code's PORTS tab and click on the browser icon to forward the UI to your local
  # browser

The virtualenv is root-owned, so adding a package needs sudo:
  sudo /opt/venv/bin/pip install <package>
EOF
