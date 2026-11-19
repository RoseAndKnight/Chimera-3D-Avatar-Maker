"""python3 -I tools/games [list|table|write|check]: see compatibility.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compatibility as K  # noqa: E402

a = sys.argv[1:] or ["list"]
if a[0] == "list":
    K.items()
elif a[0] == "table":
    print(K.array(), end="")
elif a[0] == "write":
    K.write()
elif a[0] == "check":
    e = K.check()
    print("\n".join(e) if e else "COMPATIBILITY.md is up to date, declarations complete.")
    sys.exit(1 if e else 0)
else:
    sys.exit(__doc__)
