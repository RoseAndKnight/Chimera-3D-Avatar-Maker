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


class Driver:
    def __init__(self, port=2828, waiting=120):
        end = time.time() + waiting
        while True:
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=300)
                break
            except OSError:
                if time.time() > end:
                    raise
                time.sleep(1)
        self.n = 0
        self._read()
        self.command("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {"pageLoadStrategy": "none"}}})

    def _read(self):
        buffer = b""
        while b":" not in buffer:
            buffer += self.s.recv(1)
        t, rest = buffer.split(b":", 1)
        t = int(t)
        while len(rest) < t:
            chunk = self.s.recv(min(1 << 20, t - len(rest)))
            if not chunk:
                raise ConnectionError("Firefox closed the connection")
            rest += chunk
        return json.loads(rest)

    def send(self, name, params=None):
        self.n += 1
        body = json.dumps([0, self.n, name, params or {}]).encode()
        self.s.sendall(str(len(body)).encode() + b":" + body)
        return self.n

    def command(self, name, params=None):
        n = self.send(name, params)
        while True:
            _, received, error, value = self._read()
            if received == n:
                break
        if error:
            raise RuntimeError(f"{name} : {error.get('error')} : {error.get('message')}")
        return value

    def go(self, url):
        self.command("WebDriver:Navigate", {"url": url})

    def size(self, l, h):
        self.command("WebDriver:SetWindowRect", {"width": l, "height": h})

    def js(self, code, *args, delay=120000):
        """Runs `code` (a function body) in the page; `return` gives back the value."""
        self.command("WebDriver:SetTimeouts", {"script": delay})
        v = self.command("WebDriver:ExecuteScript", {"script": code, "args": list(args), "sandbox": None})
        return v.get("value") if isinstance(v, dict) else v

    def wait(self, condition, delay=120, stage=0.5):
        end = time.time() + delay
        while time.time() < end:
            try:
                if self.js(f"return !!({condition});"):
                    return True
            except RuntimeError:  # page or session still opening
                pass
            time.sleep(stage)
        raise TimeoutError(condition)

    def capture(self, path):
        v = self.command("WebDriver:TakeScreenshot", {"full": False})
        v = v.get("value") if isinstance(v, dict) else v
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(base64.b64decode(v))
        os.replace(tmp, path)
        return path

    def close(self):
        try:
            self.command("Marionette:Quit")
        except Exception:
            pass
