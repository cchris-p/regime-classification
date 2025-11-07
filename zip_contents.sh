#!/usr/bin/env bash

set -e
set -u
set -o pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_NAME="${1:-regime_classification_artifacts_$(date +%Y%m%d_%H%M%S).zip}"
BASE="${ZIP_NAME%.zip}"
ARCHIVE_PATH="$ZIP_NAME"

# remove any legacy staging dir from old versions
rm -rf ./.zip_staging || true

# ---- Files to include (project-relative) ----
PATHS=(
  regime_partitioning/datasets/cpi_diff_core.py
  regime_partitioning/datasets/rate_diff_2y.py
  regime_partitioning/datasets/rv_20d.py
  regime_partitioning/api.py
  regime_partitioning/processing.py
  regime_partitioning/constants.py
  data-collected.txt
  fx-regime-partitioning.py
)
# --------------------------------------------

# Resolve existing entries and de-duplicate
added=()
missing=()
for rel in "${PATHS[@]}"; do
  if compgen -G "$rel" > /dev/null; then
    # expand globs to individual paths
    for m in $rel; do
      case " ${added[*]-} " in *" $m "*) : ;; *) added+=("$m");; esac
    done
  else
    # record literal paths that are missing (ignore unmatched globs)
    if [[ "$rel" != *"*"* && "$rel" != *?* && "$rel" != *"["* ]]; then
      missing+=("$rel")
    fi
  fi
done

# Create temp work area with single top-level folder
WORKDIR="$(mktemp -d -t regime_classification_zip_XXXXXX)"
DEST="$WORKDIR/$BASE"
mkdir -p "$DEST"

# Copy files/dirs into DEST preserving relative layout
for rel in "${added[@]}"; do
  if [[ -d "$rel" ]]; then
    mkdir -p "$DEST/$rel"
    cp -R "$rel/." "$DEST/$rel/"
  else
    if cp --parents -r "$rel" "$DEST/" 2>/dev/null; then
      :
    else
      mkdir -p "$(dirname "$DEST/$rel")"
      cp -r "$rel" "$DEST/$rel"
    fi
  fi
done


# Write a file list (all files that will be in the archive), relative to project root
LIST_FILE="${BASE}_filelist.txt"
(
  cd "$SCRIPT_DIR"
  tmp_list="$(mktemp)"
  for p in "${added[@]}"; do
    if [[ -d "$p" ]]; then
      find "$p" -type f -o -type l
    else
      printf '%s\n' "$p"
    fi
  done | sed 's#^\./##' | sort -u > "$tmp_list"
  mkdir -p "$DEST"
  mv "$tmp_list" "$DEST/$LIST_FILE"
)

# Write a full project tree snapshot at the archive root for reference
PROJECT_TREE_FILE="${BASE}_project_tree.txt"
(
  cd "$SCRIPT_DIR"
  if command -v tree >/dev/null 2>&1; then
    # exclude .git to keep size reasonable; remove -I .git to include it
    tree -a -I '.git' > "$DEST/$PROJECT_TREE_FILE"
  else
    printf '%s\n' "[tree not found; using find as fallback]" > "$DEST/$PROJECT_TREE_FILE"
    find . -print | sed '1d' >> "$DEST/$PROJECT_TREE_FILE"
  fi
)

# Create archive that contains a single top-level directory $BASE
# Always write the archive into $SCRIPT_DIR so scp can find it.
if command -v zip >/dev/null 2>&1; then
  # ZIP case
  [[ -f "$ZIP_NAME" ]] && rm -f "$ZIP_NAME"
  (
    cd "$WORKDIR"
    zip -r "$SCRIPT_DIR/$ZIP_NAME" "$BASE" >/dev/null
  )
  ARCHIVE_PATH="$ZIP_NAME"
else
  # TAR.GZ fallback
  ARCHIVE_PATH="${BASE}.tar.gz"
  [[ -f "$ARCHIVE_PATH" ]] && rm -f "$ARCHIVE_PATH"
  tar -czf "$SCRIPT_DIR/$ARCHIVE_PATH" -C "$WORKDIR" "$BASE"
fi

# Cleanup temp
rm -rf "$WORKDIR"
shopt -u nullglob || true

# Summary

echo
echo "Created: $ARCHIVE_PATH"
echo

echo "Included paths:"
for f in "${added[@]}"; do echo "- $f"; done

echo

if (( ${#missing[@]} )); then
  echo "Missing (skipped):"
  for f in "${missing[@]}"; do echo "- $f"; done
else
  echo "Missing (skipped): None"
fi

echo

echo "Uploading..."
scp "$ARCHIVE_PATH" lapis:~

rm regime_classification_artifacts_*.tar.gz || true