#!/usr/bin/env python3
"""
phase_commit_fixed.py — Commit, push, and tag a repo for a given Phase.

Examples
  python phase_commit_fixed.py --phase 6
  python phase_commit_fixed.py --phase 11 --message "Phase 11 — GPU mask fixes"
  python phase_commit_fixed.py --phase 11 --branch main --yes
  python phase_commit_fixed.py --phase 11 --dry-run
  python phase_commit_fixed.py --phase 11 --force-tag  # recreate tag if it exists

Notes
- Pushes the current (or specified) branch.
- Creates an annotated tag like v11.0.0 or v11.5.0 and pushes it explicitly.
- Verifies the tag exists on the remote after push.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import re
from typing import List


def run(cmd: List[str], *, dry: bool = False, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd))
    if dry:
        # Simulate a successful run
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=check, capture_output=True, text=text)


def repo_root(*, dry: bool = False) -> str:
    cp = run(["git", "rev-parse", "--show-toplevel"], dry=dry)
    return cp.stdout.strip()


def current_branch(*, dry: bool = False) -> str:
    cp = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], dry=dry)
    return cp.stdout.strip()


def has_changes(*, dry: bool = False) -> bool:
    cp = run(["git", "status", "--porcelain"], dry=dry)
    return bool(cp.stdout.strip())


def local_tag_exists(tag: str, *, dry: bool = False) -> bool:
    try:
        run(["git", "rev-parse", "--verify", "--quiet", tag], dry=dry)
        return True
    except subprocess.CalledProcessError:
        return False


def remote_tag_exists(tag: str, *, dry: bool = False) -> bool:
    try:
        cp = run(["git", "ls-remote", "--tags", "origin", tag], dry=dry)
        return bool(cp.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def ensure_remote():
    cp = run(["git", "remote", "-v"])
    if "origin" not in cp.stdout:
        print("Error: no 'origin' remote configured.")
        sys.exit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, help="Phase number (e.g., 6, 10.5, 20.1)")
    p.add_argument("--message", type=str, help='Commit message (default: "End of PhaseX")')
    p.add_argument("--branch", type=str, help="Branch to push (default: current branch)")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="Show commands only; do not execute")
    p.add_argument("--push-tags", action="store_true", help="Also push all local tags after branch push")
    p.add_argument("--force-tag", action="store_true", help="If tag exists, recreate and overwrite it on remote")
    return p.parse_args()


def main():
    args = parse_args()

    # Phase
    phase_str = args.phase or input("Phase number (e.g., 6, 10.5): ").strip()
    if not re.fullmatch(r"\d+(\.\d+)?", phase_str):
        print("Error: phase must be like 6, 10.5, 20.1")
        sys.exit(1)

    phase = float(phase_str)
    # Tag scheme: whole number -> vX.0.0, decimal -> vX.Y.0 (one decimal place preserved)
    tag = f"v{phase:.1f}.0" if "." in phase_str else f"v{int(phase)}.0.0"
    tag = tag.replace(".0.0", ".0.0")  # no-op but explicit
    commit_msg = args.message or f"End of Phase {phase_str}"

    # Repo checks
    try:
        root = repo_root(dry=args.dry_run)
    except subprocess.CalledProcessError:
        print("Error: not inside a Git repository (or Git not installed).")
        sys.exit(1)

    ensure_remote()

    # Branch
    branch = args.branch or current_branch(dry=args.dry_run)

    print("\n---- Plan ----")
    print(f"Repo    : {root}")
    print(f"Branch  : {branch}")
    print(f"Commit  : {commit_msg}")
    print(f"Tag     : {tag}")
    print(f"Options : dry_run={args.dry_run}, push_tags={args.push_tags}, force_tag={args.force_tag}")
    if not args.yes:
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    try:
        # 1) Commit (only if there are changes)
        if has_changes(dry=args.dry_run):
            run(["git", "add", "."], dry=args.dry_run)
            run(["git", "commit", "-m", commit_msg], dry=args.dry_run)
        else:
            print("No changes to commit (working tree clean). Skipping commit.")

        # 2) Push branch
        run(["git", "push", "-u", "origin", branch], dry=args.dry_run)

        # 3) Create/Push tag
        need_create = True
        if local_tag_exists(tag, dry=args.dry_run):
            if args.force_tag:
                print(f"Tag {tag} exists locally; recreating due to --force-tag.")
                try:
                    run(["git", "tag", "-d", tag], dry=args.dry_run)
                except subprocess.CalledProcessError:
                    pass
            else:
                need_create = False
                print(f"Tag {tag} already exists locally.")
        if need_create:
            run(["git", "tag", "-a", tag, "-m", f"The end of Phase {phase_str}"], dry=args.dry_run)

        # Always push this tag explicitly
        push_cmd = ["git", "push", "origin", f"refs/tags/{tag}"]
        if args.force_tag:
            push_cmd.append("--force-with-lease")
        run(push_cmd, dry=args.dry_run)

        # Optional: push all tags
        if args.push_tags:
            run(["git", "push", "--tags"], dry=args.dry_run)

        # 4) Verify tag on remote
        # fetch tags first to make sure local knows about remote state
        run(["git", "fetch", "--tags"], dry=args.dry_run)
        if remote_tag_exists(tag, dry=args.dry_run):
            print(f"\n✓ Tag verified on remote: {tag}")
        else:
            print(f"\n⚠ Tag {tag} not visible via ls-remote. This can happen if the remote blocks tag pushes or due to permissions.")
            print("  Double-check on the Git hosting site, or push again with --force-tag.")

        print("\nDone.")
    except subprocess.CalledProcessError as e:
        print("\nGit command failed.")
        if e.stdout:
            print("STDOUT:\n", e.stdout)
        if e.stderr:
            print("STDERR:\n", e.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
