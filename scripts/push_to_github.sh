#!/usr/bin/env bash
# Manual GitHub push helper — run after you have git write access configured.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_URL="${1:-}"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: $0 <git-remote-url>"
  echo "Example: $0 git@github.com:YOUR_USER/research_tool.git"
  echo "Example: $0 https://github.com/YOUR_USER/research_tool.git"
  exit 1
fi

if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git push -u origin main
echo "Pushed to $REPO_URL"