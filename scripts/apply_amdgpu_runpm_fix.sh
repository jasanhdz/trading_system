#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root: sudo $0" >&2
  exit 1
fi

GRUB_FILE="/etc/default/grub"
BACKUP_FILE="/etc/default/grub.codex.bak.$(date +%Y%m%d%H%M%S)"
GPU_DEVS=(
  "/sys/bus/pci/devices/0000:03:00.0"
  "/sys/bus/pci/devices/0000:07:00.0"
)

cp "$GRUB_FILE" "$BACKUP_FILE"

python3 - <<'PY'
from pathlib import Path

path = Path("/etc/default/grub")
text = path.read_text()
old = 'GRUB_CMDLINE_LINUX_DEFAULT="'
if old not in text:
    raise SystemExit("GRUB_CMDLINE_LINUX_DEFAULT not found")

lines = text.splitlines()
out = []
changed = False
for line in lines:
    if line.startswith('GRUB_CMDLINE_LINUX_DEFAULT='):
        prefix = 'GRUB_CMDLINE_LINUX_DEFAULT="'
        suffix = '"'
        value = line[len(prefix):-1] if line.endswith('"') else line[len(prefix):]
        parts = [p for p in value.split() if p != "amdgpu.runpm=0"]
        parts.append("amdgpu.runpm=0")
        new_value = " ".join(dict.fromkeys(parts))
        line = f'{prefix}{new_value}{suffix}'
        changed = True
    out.append(line)

if not changed:
    raise SystemExit("Failed to update GRUB_CMDLINE_LINUX_DEFAULT")

path.write_text("\n".join(out) + "\n")
PY

update-grub

for dev in "${GPU_DEVS[@]}"; do
  if [[ -w "${dev}/power/control" ]]; then
    echo on > "${dev}/power/control"
  fi
done

echo "Applied."
echo "GRUB backup: $BACKUP_FILE"
for dev in "${GPU_DEVS[@]}"; do
  if [[ -e "${dev}/power/control" ]]; then
    printf '%s control=' "$dev"
    cat "${dev}/power/control"
  fi
done

