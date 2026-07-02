"""Integration of the Perfetto trace viewer into the NiceGUI frontend for visualizing profiling traces."""

from nicegui import ui
from pathlib import Path


def patch_html():
    """This needs to be called for each page to enable the trace viewer button"""
    ui.add_body_html('<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>')
    ui.add_body_html("""
<script>
const TRACEVIEWER_URL = 'https://ui.perfetto.dev';

async function fetchAndOpenTrace(traceUrl, title) {
  const resp = await fetch(traceUrl);
  console.log(resp)
  const blob = await resp.blob();
  console.log(blob)
  const arrayBuffer = await blob.arrayBuffer();
  openTrace(arrayBuffer, traceUrl, title);
}

function openTrace(arrayBuffer, traceUrl, trace_title) {
  const win = window.open(TRACEVIEWER_URL);
  if (!win) {
    return;
  }
  const timer = setInterval(() => win.postMessage('PING', TRACEVIEWER_URL), 50);

  const onMessageHandler = (evt) => {
    console.log(evt.data)
    if (evt.data !== 'PONG') return;

    // We got a PONG, the UI is ready.
    window.clearInterval(timer);
    window.removeEventListener('message', onMessageHandler);

    win.postMessage({
      perfetto: {
        buffer: arrayBuffer,
        title: trace_title,
    }}, TRACEVIEWER_URL);
  };

  window.addEventListener('message', onMessageHandler);
}
</script>
""")


def traceviewer(text: str, trace_path: Path, trace_title: str = "Untitled trace"):
    """Create a button for opening the trace file in the web viewer (new tab)
    Args:
        text: Button text
        trace_path: Path to the trace json file
        trace_title: Title to show in the trace viewer
    """

    async def open_trace():
        await ui.run_javascript(f'fetchAndOpenTrace("{str(trace_path)}", "{trace_title}")', timeout=15.0)

    return ui.button(text, on_click=open_trace)


def patch(profiler_data_dir):
    """Call this function once for the process to register the trace viewer button"""
    if not hasattr(ui, "traceviewer"):
        ui.traceviewer = traceviewer
        ui.traceviewer.patch_html = patch_html
        ui.traceviewer.profiler_data_dir = profiler_data_dir
