#!/usr/bin/env bash
set -euo pipefail

bucket="${MATSEPARATE_CONTEXT_BUCKET:-s3://matseparate-context-dataset-251995574236-us-west-2}"
workdir="${MATSEPARATE_MIRROR_WORKDIR:-/data/matseparate-context/raw}"

mkdir -p "$workdir"

download_and_upload() {
  local name="$1"
  local url="$2"
  local expected_size="$3"
  local type_tag="$4"
  local local_path="$workdir/$name"
  local s3_uri="$bucket/raw/$name"

  if aws s3 ls "$s3_uri" >/dev/null 2>&1; then
    echo "exists: $s3_uri"
    return 0
  fi

  if [ ! -f "$local_path" ] || [ "$(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path")" != "$expected_size" ]; then
    curl -L --fail --continue-at - --retry 8 --retry-delay 10 \
      --output "$local_path" "$url"
  fi

  actual_size="$(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path")"
  if [ "$actual_size" != "$expected_size" ]; then
    echo "size mismatch for $name: got $actual_size expected $expected_size" >&2
    exit 1
  fi

  aws s3 cp "$local_path" "$s3_uri" --only-show-errors
  aws s3api put-object-tagging \
    --bucket "${bucket#s3://}" \
    --key "raw/$name" \
    --tagging "TagSet=[{Key=source,Value=matador},{Key=type,Value=$type_tag}]"
}

download_and_upload \
  "matador.label.tar" \
  "https://fpcv.cs.columbia.edu/static/cvclass/downloads/matador.label.tar" \
  "7413760" \
  "label"

download_and_upload \
  "matador.appearance.tar" \
  "https://fpcv.cs.columbia.edu/static/cvclass/downloads/matador.appearance.tar" \
  "11391805440" \
  "appearance"

download_and_upload \
  "matador.context.tar" \
  "https://fpcv.cs.columbia.edu/static/cvclass/downloads/matador.context.tar" \
  "529514946560" \
  "context"
