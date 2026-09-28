#!/usr/bin/env bash
# JALEBI installer wrapper: finds a Python >= 3.10 and runs the interactive installer.
#   bash install.sh              interactive
#   bash install.sh --yes        accept all defaults
#   bash install.sh --help       all options (they are passed to install.py)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for py in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    if "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' 2>/dev/null; then
      exec "$py" install.py "$@"
    fi
  fi
done

cat <<'EOF'
No Python 3 found.  Install one of:
  * Miniforge (recommended; brings conda + Python):  https://github.com/conda-forge/miniforge
  * Python 3.12 from https://www.python.org/downloads/ or your package manager
then run:  bash install.sh
EOF
exit 1
