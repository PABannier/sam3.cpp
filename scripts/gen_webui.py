#!/usr/bin/env python3
"""Regenerate examples/sam3-server/webui_html.h from index.html."""
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
html = (root / "examples/sam3-server/index.html").read_text()
esc = html.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
out = (
    "// Auto-generated from index.html by scripts/gen_webui.py — do not edit.\n"
    "static const char * sam3_server_index_html =\n"
    f'    "{esc}";\n'
)
(root / "examples/sam3-server/webui_html.h").write_text(out)
print(f"webui_html.h: {len(out)} bytes from {len(html)} chars")
