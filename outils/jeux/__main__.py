"""python3 -I outils/jeux [liste|tableau|ecrire|verifier]: see compatibilite.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compatibilite as K  # noqa: E402

a = sys.argv[1:] or ["liste"]
if a[0] == "liste":
    K.liste()
elif a[0] == "tableau":
    print(K.tableau(), end="")
elif a[0] == "ecrire":
    K.ecrire()
elif a[0] == "verifier":
    e = K.verifier()
    print("\n".join(e) if e else "COMPATIBILITE.md is up to date, declarations complete.")
    sys.exit(1 if e else 0)
else:
    sys.exit(__doc__)
