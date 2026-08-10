#!/bin/bash

# Finder-friendly macOS launcher for the Catalysis Data Toolkit.
# Double-click this file, or run it from Terminal with: ./run.command

set -u
set -o pipefail

TOOLKIT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="$TOOLKIT_DIR/.conda_env"
VENV_DIR="$TOOLKIT_DIR/.venv"
REQUIREMENTS_FILE="$TOOLKIT_DIR/requirements.txt"

MODE="app"
OPEN_BROWSER=1
START_PATH="${CATALYSIS_TOOLKIT_START_PATH:-/}"
REQUESTED_PORT="${CATALYSIS_TOOLKIT_PORT:-5001}"

banner() {
    printf '\n'
    printf ' ============================================\n'
    printf '  Catalysis Data Toolkit - macOS launcher\n'
    printf ' ============================================\n'
    printf '\n'
}

pause_on_error() {
    if [ -t 0 ]; then
        printf '\nPress Return to close this window...'
        read -r _unused
    fi
}

die() {
    printf '\nERROR: %s\n' "$1" >&2
    pause_on_error
    exit 1
}

usage() {
    cat <<'EOF'
Usage: ./run.command [options]

Options:
  --xrd                Open directly to the XRD toolkit.
  --port PORT          Start at PORT and use the next free port if needed.
  --no-browser         Do not open the browser automatically.
  --batch ARGS...      Run scripts/xrd_batch.py with the remaining arguments.
  --fetch-cifs ARGS... Run scripts/fetch_cifs.py with the remaining arguments.
  -h, --help           Show this help.

Environment variables:
  CATALYSIS_TOOLKIT_PORT       Preferred server port (default: 5001).
  CATALYSIS_TOOLKIT_START_PATH Browser path (default: /).
  CATALYSIS_TOOLKIT_SKIP_GSAS  Set to 1 to skip optional GSAS-II installation.
EOF
}

find_conda() {
    local candidate
    local brew_prefix

    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    for candidate in \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "/opt/homebrew/bin/conda" \
        "/usr/local/bin/conda" \
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
        "/usr/local/Caskroom/miniforge/base/bin/conda"
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if command -v brew >/dev/null 2>&1; then
        brew_prefix="$(brew --prefix miniforge 2>/dev/null || true)"
        if [ -n "$brew_prefix" ] && [ -x "$brew_prefix/bin/conda" ]; then
            printf '%s\n' "$brew_prefix/bin/conda"
            return 0
        fi
    fi

    return 1
}

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    return 1
}

core_dependencies_ready() {
    "$PYTHON_BIN" -c \
        "import flask, yaml, numpy, pandas, openpyxl, matplotlib, PIL, requests, pymatgen, scipy" \
        >/dev/null 2>&1
}

gsas_ready() {
    "$PYTHON_BIN" -c \
        "from modules.xrd.gsasii_backend import is_available; raise SystemExit(0 if is_available() else 1)" \
        >/dev/null 2>&1
}

choose_free_port() {
    "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, min(65536, start + 100)):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    raise SystemExit(0)

raise SystemExit("No free local port found in the requested range.")
PY
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --xrd)
            START_PATH="/xrd"
            shift
            ;;
        --port)
            [ -n "${2:-}" ] || die "--port requires a port number."
            REQUESTED_PORT="$2"
            shift 2
            ;;
        --no-browser)
            OPEN_BROWSER=0
            shift
            ;;
        --batch)
            MODE="batch"
            shift
            break
            ;;
        --fetch-cifs)
            MODE="fetch-cifs"
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1 (use --help for usage)."
            ;;
    esac
done

if [ "$(uname -s)" != "Darwin" ]; then
    die "run.command is for macOS. On Windows, use run.bat."
fi

case "$REQUESTED_PORT" in
    ''|*[!0-9]*) die "Port must be a number between 1024 and 65535." ;;
esac
if [ "$REQUESTED_PORT" -lt 1024 ] || [ "$REQUESTED_PORT" -gt 65535 ]; then
    die "Port must be a number between 1024 and 65535."
fi

cd "$TOOLKIT_DIR" || die "Could not open the toolkit directory."
banner

USE_CONDA=0
PYTHON_BIN=""
CONDA_EXE="$(find_conda || true)"

if [ -n "$CONDA_EXE" ]; then
    printf 'Conda detected. Using a local toolkit environment.\n\n'
    CONDA_BASE="$("$CONDA_EXE" info --base 2>/dev/null || true)"

    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        eval "$("$CONDA_EXE" shell.bash hook 2>/dev/null)"
    fi

    if [ ! -x "$CONDA_ENV/bin/python" ]; then
        printf 'Creating the macOS conda environment (first run only)...\n'
        if conda create --prefix "$CONDA_ENV" python=3.11 --yes; then
            printf '\nConda environment created.\n\n'
        else
            printf '\nWARNING: Conda environment creation failed; trying Python venv instead.\n\n'
        fi
    fi

    if [ -x "$CONDA_ENV/bin/python" ] && conda activate "$CONDA_ENV"; then
        USE_CONDA=1
        PYTHON_BIN="$CONDA_ENV/bin/python"
    fi
fi

if [ "$USE_CONDA" -eq 0 ]; then
    SYSTEM_PYTHON="$(find_python || true)"
    [ -n "$SYSTEM_PYTHON" ] || die \
        "Python 3 was not found. Install Miniforge from https://github.com/conda-forge/miniforge and try again."

    printf 'Using a Python virtual environment.\n'
    printf 'Install Miniforge later if you need GSAS-II refinement.\n\n'

    if [ ! -x "$VENV_DIR/bin/python" ]; then
        printf 'Creating the Python environment (first run only)...\n'
        "$SYSTEM_PYTHON" -m venv "$VENV_DIR" || die "Could not create $VENV_DIR."
        printf 'Python environment created.\n\n'
    fi

    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate" || die "Could not activate $VENV_DIR."
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

if ! core_dependencies_ready; then
    printf 'Installing toolkit dependencies (first run may take several minutes)...\n\n'

    if [ "$USE_CONDA" -eq 1 ]; then
        conda install --prefix "$CONDA_ENV" --channel conda-forge --yes \
            numpy scipy matplotlib pandas pymatgen pillow openpyxl \
            || die "Conda could not install the scientific dependencies."
    fi

    "$PYTHON_BIN" -m pip install --requirement "$REQUIREMENTS_FILE" pycifrw xmltodict \
        || die "Python dependency installation failed. Check the internet connection and try again."

    core_dependencies_ready || die "One or more required Python packages still cannot be imported."
    printf '\nDependencies installed successfully.\n\n'
fi

if [ "$USE_CONDA" -eq 1 ] && ! gsas_ready; then
    if [ "${CATALYSIS_TOOLKIT_SKIP_GSAS:-0}" = "1" ]; then
        printf 'Skipping optional GSAS-II installation.\n\n'
    else
        printf 'Installing optional GSAS-II support (this may take several minutes)...\n\n'
        if conda install --prefix "$CONDA_ENV" --channel briantoby --yes gsas2pkg; then
            "$PYTHON_BIN" -m pip install pycifrw xmltodict >/dev/null 2>&1 || true
            if gsas_ready; then
                printf '\nGSAS-II is ready.\n\n'
            else
                printf '\nWARNING: GSAS-II installed but could not be loaded.\n'
                printf 'The Le Bail and in-house Rietveld workflows remain available.\n\n'
            fi
        else
            printf '\nWARNING: GSAS-II installation was not available for this Mac.\n'
            printf 'The Le Bail and in-house Rietveld workflows remain available.\n\n'
        fi
    fi
fi

if [ ! -f "$TOOLKIT_DIR/config.yaml" ] && [ -f "$TOOLKIT_DIR/config.yaml.example" ]; then
    cp "$TOOLKIT_DIR/config.yaml.example" "$TOOLKIT_DIR/config.yaml" \
        || die "Could not create config.yaml."
    printf 'Created config.yaml from the example file.\n'
    printf 'Add a Materials Project API key there to enable Materials Project searches.\n\n'
fi

case "$MODE" in
    batch)
        exec "$PYTHON_BIN" "$TOOLKIT_DIR/scripts/xrd_batch.py" "$@"
        ;;
    fetch-cifs)
        exec "$PYTHON_BIN" "$TOOLKIT_DIR/scripts/fetch_cifs.py" "$@"
        ;;
esac

PORT="$(choose_free_port "$REQUESTED_PORT")" \
    || die "Could not select an available local server port."

case "$START_PATH" in
    http://*|https://*)
        TOOLKIT_URL="$START_PATH"
        ;;
    /*)
        TOOLKIT_URL="http://127.0.0.1:$PORT$START_PATH"
        ;;
    *)
        TOOLKIT_URL="http://127.0.0.1:$PORT/$START_PATH"
        ;;
esac

printf 'Starting the Catalysis Data Toolkit at:\n'
printf '  %s\n\n' "$TOOLKIT_URL"
printf 'Close this Terminal window or press Control-C to stop the server.\n\n'

if [ "$OPEN_BROWSER" -eq 1 ]; then
    (
        sleep 1.5
        /usr/bin/open "$TOOLKIT_URL" >/dev/null 2>&1
    ) &
fi

"$PYTHON_BIN" -m flask --app app run --host 127.0.0.1 --port "$PORT"
SERVER_STATUS=$?

if [ "$SERVER_STATUS" -ne 0 ] && [ "$SERVER_STATUS" -ne 130 ]; then
    printf '\nThe toolkit server stopped unexpectedly (exit code %s).\n' "$SERVER_STATUS" >&2
    pause_on_error
fi

exit "$SERVER_STATUS"
