#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"

echo "Preparando ejemplos para la documentacion..."
python3 "$DIR/examples_doc.py"

echo "Documentacion preparada."
