#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

##################################################################
# PyMCel release script
#
# - Updates the version in:
#   - setup.py
#   - src/pymcel/version.py
# - Builds (python -m build), validates (twine check), and uploads (twine upload)
# - If anything fails, automatically rolls back to the previous version
##################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SETUP_PY="$ROOT_DIR/setup.py"
VERSION_PY="$ROOT_DIR/src/pymcel/version.py"

BACKUP_DIR=""
ROLLED_BACK=0

log() { printf '%s\n' "$*"; }
err() { printf '%s\n' "$*" >&2; }
die() { err "ERROR: $*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

python_bin() {
  if have python; then echo "python"; return 0; fi
  if have python3; then echo "python3"; return 0; fi
  return 1
}

PY="$(python_bin)" || die "Could not find 'python' nor 'python3' in PATH."
have git || die "Could not find 'git' in PATH."

# Auto-activate local venv if present
if [[ -f ".pymcel/bin/activate" ]]; then
  log "Activating local environment (.pymcel)..."
  # shellcheck disable=SC1091
  source .pymcel/bin/activate
  PY="python3"
fi

rollback() {
  local exit_code=$?

  if [[ $ROLLED_BACK -eq 1 ]]; then
    exit "$exit_code"
  fi
  ROLLED_BACK=1

  err
  err "Release failed (exit code $exit_code). Rolling back version changes..."

  if [[ -n "${BACKUP_DIR}" && -d "${BACKUP_DIR}" ]]; then
    if [[ -f "${BACKUP_DIR}/setup.py" ]]; then cp -f "${BACKUP_DIR}/setup.py" "$SETUP_PY" || true; fi
    if [[ -f "${BACKUP_DIR}/version.py" ]]; then cp -f "${BACKUP_DIR}/version.py" "$VERSION_PY" || true; fi
  fi

  rm -rf "$ROOT_DIR/dist" "$ROOT_DIR/build" "$ROOT_DIR"/*.egg-info "$ROOT_DIR/src"/*.egg-info 2>/dev/null || true

  err "Rollback complete. Repository restored to the previous version (locally)."
  exit "$exit_code"
}

trap rollback ERR INT TERM

usage() {
  cat <<'EOF'
Usage:
  bash bin/release.sh <test|release> <version> [--dry-run|--no-upload]

Examples:
  bash bin/release.sh test 0.6.32
  bash bin/release.sh release 0.7.0
  bash bin/release.sh test 0.7.0 --dry-run

Notes:
  - Requires a clean working tree (no uncommitted changes).
  - If anything fails in build/check/upload, it restores the previous version.
  - With --dry-run/--no-upload, it runs build + twine check, but does NOT upload.
EOF
}

TYPE="${1:-}"
VERSION_NEW="${2:-}"
NO_UPLOAD=0

for arg in "${@:3}"; do
  case "$arg" in
    --dry-run|--no-upload) NO_UPLOAD=1 ;;
    *) die "Unknown argument: $arg (use --help)" ;;
  esac
done

if [[ -z "$TYPE" || "$TYPE" == "-h" || "$TYPE" == "--help" ]]; then
  usage
  exit 0
fi

case "$TYPE" in
  test|release) ;;
  *) die "Unknown release type '$TYPE'. Must be 'test' or 'release'." ;;
esac

if [[ ! -f "$SETUP_PY" ]]; then die "Missing file: $SETUP_PY"; fi
if [[ ! -f "$VERSION_PY" ]]; then die "Missing file: $VERSION_PY"; fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  die "Working tree is not clean. Commit or stash your changes before releasing."
fi

CURRENT_VERSIONS="$("$PY" - <<'PY'
import pathlib, re

root = pathlib.Path(".").resolve()
files = {
    "setup.py": root / "setup.py",
    "version.py": root / "src/pymcel/version.py",
}

def extract(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.name == "setup.py":
        m = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]\s*,?\s*$", text, flags=re.M)
        if not m:
            raise SystemExit(f"Could not find 'version=' in {path}")
        return m.group(1)
    m = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]\s*$", text, flags=re.M)
    if not m:
        raise SystemExit(f"Could not find 'version=' in {path}")
    return m.group(1)

versions = {k: extract(p) for k, p in files.items()}
print(versions["setup.py"])
print(versions["version.py"])
PY
)"

CURRENT_SETUP="$(printf '%s\n' "$CURRENT_VERSIONS" | sed -n '1p')"
CURRENT_VERPY="$(printf '%s\n' "$CURRENT_VERSIONS" | sed -n '2p')"

if [[ "$CURRENT_SETUP" != "$CURRENT_VERPY" ]]; then
  die "Current versions do not match: setup.py=$CURRENT_SETUP, version.py=$CURRENT_VERPY. Fix this before releasing."
fi

if [[ -z "$VERSION_NEW" ]]; then
  err "Current version: $CURRENT_SETUP"
  die "You must specify a new version."
fi

if ! printf '%s' "$VERSION_NEW" | grep -Eq '^[0-9]+(\.[0-9]+)+([a-zA-Z0-9\.\-]+)?$'; then
  die "Invalid version '$VERSION_NEW'. Use something like 0.7.0 or 0.7.0.1."
fi

if [[ "$VERSION_NEW" == "$CURRENT_SETUP" ]]; then
  die "New version ($VERSION_NEW) must be different from current ($CURRENT_SETUP)."
fi

log "Releasing PyMCel $VERSION_NEW (current: $CURRENT_SETUP) in '$TYPE' mode..."

BACKUP_DIR="$(mktemp -d -t pymcel-release.XXXXXX)"
cp -f "$SETUP_PY" "${BACKUP_DIR}/setup.py"
cp -f "$VERSION_PY" "${BACKUP_DIR}/version.py"

update_file() {
  local path="$1"
  local pattern="$2"
  local replacement="$3"

  UPDATE_PATH="$path" PATTERN="$pattern" REPLACEMENT="$replacement" "$PY" - <<'PY'
import os
import pathlib
import re

p = pathlib.Path(os.environ["UPDATE_PATH"])
pattern = os.environ["PATTERN"]
replacement = os.environ["REPLACEMENT"]

text = p.read_text(encoding="utf-8")
new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.M)
if n != 1:
    raise SystemExit(f"Could not update {p} (replacements: {n})")
p.write_text(new_text, encoding="utf-8")
PY
}

log "Updating version in setup.py and package files..."
update_file "$SETUP_PY" '^(\s*version\s*=\s*)["'"'"']([^"'"'"']+)["'"'"'](\s*,?\s*)$' "\\1'${VERSION_NEW}'\\3"
update_file "$VERSION_PY" '^(\s*version\s*=\s*)["'"'"']([^"'"'"']+)["'"'"'](\s*)$' "\\1'${VERSION_NEW}'\\3"

log "Cleaning previous build artifacts..."
rm -rf "$ROOT_DIR/dist" "$ROOT_DIR/build" "$ROOT_DIR"/*.egg-info "$ROOT_DIR/src"/*.egg-info 2>/dev/null || true

log "Building distributions (sdist/wheel)..."
"$PY" -m build

log "Ensuring twine check dependencies..."
"$PY" -m pip install -U pip
"$PY" -m pip install -U twine readme_renderer
"$PY" -m pip install -U --force-reinstall --only-binary=:all: nh3 || "$PY" -m pip install -U nh3

log "Validating distributions (twine check)..."
"$PY" -m twine check dist/*

if [[ $NO_UPLOAD -eq 1 ]]; then
  log "Dry-run: skipping upload. (build + twine check OK)"
  trap - ERR INT TERM
  rm -rf "$BACKUP_DIR" 2>/dev/null || true
  log "Done. Release (no upload) completed: $VERSION_NEW"
  exit 0
fi

if [[ "$TYPE" == "test" ]]; then
  log "Uploading to TestPyPI..."
  "$PY" -m twine upload --repository testpypi dist/* --verbose
else
  log "Uploading to PyPI..."
  "$PY" -m twine upload dist/* --verbose
fi

trap - ERR INT TERM
rm -rf "$BACKUP_DIR" 2>/dev/null || true
log "Done. Release completed: $VERSION_NEW"