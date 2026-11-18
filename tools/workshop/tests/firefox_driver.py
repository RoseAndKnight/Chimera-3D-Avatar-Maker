"""Small Firefox driver over Marionette, without dependencies.

Used to play a real session on the page from a script: open, click,
read the state, take a screenshot. Firefox must run with
`--headless --marionette --profile <folder>`. In a sandbox without
namespaces, set `security.sandbox.content.level` to 0 in this throwaway profile,
otherwise the tabs die at startup.
"""
import base64
import json
import os
import socket
import time


class Pilote:
    def __init__(self, port=2828, attente=120):
        fin = time.time() + attente
        while True:
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=300)
                break
            except OSError:
                if time.time() > fin:
                    raise
                time.sleep(1)
        self.n = 0
        self._lire()
        self.commande("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {"pageLoadStrategy": "none"}}})

    def _lire(self):
        tampon = b""
        while b":" not in tampon:
            tampon += self.s.recv(1)
        t, reste = tampon.split(b":", 1)
        t = int(t)
        while len(reste) < t:
            morceau = self.s.recv(min(1 << 20, t - len(reste)))
            if not morceau:
                raise ConnectionError("Firefox closed the connection")
            reste += morceau
        return json.loads(reste)

    def envoyer(self, nom, params=None):
        self.n += 1
        corps = json.dumps([0, self.n, nom, params or {}]).encode()
        self.s.sendall(str(len(corps)).encode() + b":" + corps)
        return self.n

    def commande(self, nom, params=None):
        n = self.envoyer(nom, params)
        while True:
            _, recu, erreur, valeur = self._lire()
            if recu == n:
                break
        if erreur:
            raise RuntimeError(f"{nom} : {erreur.get('error')} : {erreur.get('message')}")
        return valeur

    def aller(self, url):
        self.commande("WebDriver:Navigate", {"url": url})

    def taille(self, l, h):
        self.commande("WebDriver:SetWindowRect", {"width": l, "height": h})

    def js(self, code, *args, delai=120000):
        """Runs `code` (a function body) in the page; `return` gives back the value."""
        self.commande("WebDriver:SetTimeouts", {"script": delai})
        v = self.commande("WebDriver:ExecuteScript", {"script": code, "args": list(args), "sandbox": None})
        return v.get("value") if isinstance(v, dict) else v

    def attendre(self, condition, delai=120, pas=0.5):
        fin = time.time() + delai
        while time.time() < fin:
            try:
                if self.js(f"return !!({condition});"):
                    return True
            except RuntimeError:  # page or session still opening
                pass
            time.sleep(pas)
        raise TimeoutError(condition)

    def capture(self, chemin):
        v = self.commande("WebDriver:TakeScreenshot", {"full": False})
        v = v.get("value") if isinstance(v, dict) else v
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        tmp = chemin + ".tmp"
        with open(tmp, "wb") as f:
            f.write(base64.b64decode(v))
        os.replace(tmp, chemin)
        return chemin

    def fermer(self):
        try:
            self.commande("Marionette:Quit")
        except Exception:
            pass
