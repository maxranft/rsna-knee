import json, sys, pathlib
for p in map(pathlib.Path, sys.argv[1:]):
    try: nb = json.loads(p.read_text())
    except Exception as e: print("skip", p, e); continue
    out = []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "code": out.append(src)
        else: out.append("\n".join("# " + l for l in src.splitlines()))
    p.with_suffix(".nb.py").write_text("\n\n# %%\n".join(out))
