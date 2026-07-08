#!/usr/bin/env bash
# Alias for manual push — pass your GitHub repo URL as the first argument.
exec "$(dirname "$0")/push_to_github.sh" "$@"