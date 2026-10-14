#!/bin/sh
# Copy this template into a new project and link the shared tool environments.
# Usage: new_project.sh <slug> [parent]
# The slug is a short lowercase name. The default parent is $PROJECTS_DIR (see .env.example).
# Environments are the existing tree, linked, never copied.
set -eu

slug="${1:-}"
parent="${2:-${PROJECTS_DIR:?set PROJECTS_DIR or pass [parent]}}"

if ! printf '%s' "$slug" | grep -Eq '^[a-z][a-z0-9-]{0,40}$'; then
  echo "usage: new_project.sh <slug> [parent]" >&2
  echo "slug must match ^[a-z][a-z0-9-]{0,40}$" >&2
  exit 2
fi

case "$slug" in
  _template)
    echo "refusing slug $slug" >&2
    exit 2
    ;;
esac

dest="$parent/$slug"
if [ -e "$dest" ]; then
  echo "already exists: $dest" >&2
  exit 2
fi

src=$(CDPATH= cd "$(dirname "$0")" && pwd)
mkdir -p "$dest"
tar -C "$src" \
  --exclude=envs \
  --exclude=__pycache__ \
  --exclude='*.bak*' \
  --exclude=new_project.sh \
  --exclude=archive \
  --exclude=overlays/render \
  --exclude=overlays/boxes.json \
  --exclude=overlays/text.json \
  -cf - . | tar -C "$dest" -xf -

find "$dest" -type f \
  \( -name '*.py' -o -name '*.md' -o -name '*.html' -o -name '*.json' -o -name '*.sh' -o -name '*.txt' -o -name '*.css' \) \
  -exec sed -i "s/__SLUG__/$slug/g" {} +

# Shared tool environments (built once with pipeline.setup_tools), linked, never copied.
ln -s "${ENVS_DIR:?set ENVS_DIR to the shared envs/ tree}" "$dest/envs"
echo "$dest"
