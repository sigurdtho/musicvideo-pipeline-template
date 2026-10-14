"""Background batches as transient systemd user units, so long renders survive the agent session.

  python -m pipeline.jobs launch <name> -- <module> [args...]   start `python -m <module> args` as unit __SLUG__-<name>
  python -m pipeline.jobs wait <name>                           block until the unit ends; print result + log tail
  python -m pipeline.jobs status                                list __SLUG__-* units
Logs: qc/logs/<name>.log
"""
import subprocess
import sys
import time

from pipeline.manifest import ROOT

PY = str(ROOT / "envs" / "pipeline" / "bin" / "python")


def unit(name: str) -> str:
    return f"__SLUG__-{name}"


def launch(name: str, module: str, args: list[str]) -> str:
    log = ROOT / "qc" / "logs" / f"{name}.log"
    cmd = " ".join([PY, "-u", "-m", module, *[f"'{a}'" for a in args]])
    subprocess.run(["systemd-run", "--user", f"--unit={unit(name)}", "--collect", f"--working-directory={ROOT}",
                    "-E", "PYTHONUNBUFFERED=1", "-E", "HF_HUB_DISABLE_TELEMETRY=1",
                    "/bin/sh", "-c", f"exec {cmd} >> {log} 2>&1"], check=True)
    return unit(name)


def active(name: str) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", unit(name)], capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating")


def wait(name: str, poll: float = 5.0) -> int:
    while active(name):
        time.sleep(poll)
    log = ROOT / "qc" / "logs" / f"{name}.log"
    tail = log.read_text().splitlines()[-15:] if log.exists() else []
    print(f"unit {unit(name)} finished", *tail, sep="\n")
    return 0


def main(argv):
    if argv[:1] == ["launch"]:
        name, sep = argv[1], argv.index("--")
        print(launch(name, argv[sep + 1], argv[sep + 2:]))
    elif argv[:1] == ["wait"]:
        return wait(argv[1])
    elif argv[:1] == ["status"]:
        subprocess.run(["systemctl", "--user", "list-units", "--all", "__SLUG__-*", "--no-pager"])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
