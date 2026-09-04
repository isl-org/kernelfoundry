"""
Turn captured output into text that is safe to store and to render as HTML.
"""

from __future__ import annotations

import re

# String terminator: OSC and friends end with BEL or ESC \.
_ST = r"(?:\x1b\\|\x07)"

# OSC 8 hyperlink: ESC ] 8 ; params ; URI ST   label   ESC ] 8 ; params ; ST
# The label is non-greedy so adjacent links do not merge into one.
_OSC8_LINK = re.compile(
    r"\x1b\]8;[^;\x1b\x07]*;(?P<uri>[^\x1b\x07]*)" + _ST + r"(?P<label>.*?)\x1b\]8;[^;\x1b\x07]*;" + _ST,
    re.DOTALL,
)

# Any other OSC sequence (window title, clipboard, ...).
_OSC = re.compile(r"\x1b\][^\x1b\x07]*" + _ST)

# DCS, SOS, PM, APC: ESC P / X / ^ / _ ... ST
_STRING_CMD = re.compile(r"\x1b[PX^_][^\n]*?" + _ST)

# CSI: ESC [ params intermediates final. Covers colour, cursor movement, erase, and the rest.
_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Escapes with no string payload: an optional intermediate then a final byte. ESC ( B selects a
# character set, ESC = switches keypad mode, ESC 7 saves the cursor. The final-byte range
# deliberately omits `[` (CSI), which is handled above; anything reaching here still carrying one
# is malformed, and consuming it is better than emitting it.
_SHORT_ESC = re.compile(r"\x1b(?:[ -/][0-~]|[0-Z\\-~])")


def _render_link(match: re.Match[str]) -> str:
    uri = match.group("uri").strip()
    label = _CSI.sub("", match.group("label"))
    if not label.strip():
        return uri
    if not uri or uri == label.strip():
        return label
    return f"{label} ({uri})"


def strip_terminal_escapes(text: str) -> str:
    if not text or "\x1b" not in text:
        return text
    text = _OSC8_LINK.sub(_render_link, text)
    text = _STRING_CMD.sub("", text)
    text = _OSC.sub("", text)
    text = _CSI.sub("", text)
    text = _SHORT_ESC.sub("", text)
    return text.replace("\x1b", "")
