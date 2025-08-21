#!/usr/bin/env python3
"""
phase_commit.py — Commit, push, and tag a repo for a given Phase.

Usage:
  python phase_commit.py --phase 6
  python phase_commit.py            # will prompt for the phase number
Options:
  --message "custom msg"            # overrides default commit message
  --branch main                     # overrides current branch
  --yes                             # skip confirmation prompt
  --dry-run                         # show commands only; do not execute
"""

import argparse
import re
import subprocess
import sys

def run(cmd, dry=False, cwd=".", text=True):
    print("$ " + " ".join(cmd))
    if dry:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=text)

def get_current_branch(dry=False):
    cp = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], dry=dry)
    return cp.stdout.strip()

def repo_root(dry=False):
    cp = run(["git", "rev-parse", "--show-toplevel"], dry=dry)
    return cp.stdout.strip()

def tag_exists(tag, dry=False):
    # returns True if tag exists locally or remotely
    try:
        run(["git", "rev-parse", "--verify", "--quiet", tag], dry=dry)
        return True
    except subprocess.CalledProcessError:
        pass
    try:
        run(["git", "ls-remote", "--tags", "origin", tag], dry=dry)
        return True
    except subprocess.CalledProcessError:
        return False

def has_changes(dry=False):
    # returns True if there are unstaged or staged changes
    cp = run(["git", "status", "--porcelain"], dry=dry)
    return bool(cp.stdout.strip())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, help="Phase number (e.g., 6, 10.5, 20.1)")
    p.add_argument("--message", type=str, help='Commit message (default: "End of PhaseX")')
    p.add_argument("--branch", type=str, help="Branch to push (default: current branch)")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute")
    args = p.parse_args()

    # Phase input
    phase_str = args.phase
    if not phase_str:
        phase_str = input("Enter phase number (e.g., 6, 10.5, 20.1): ").strip()

    # Validate phase
    if not re.fullmatch(r"\d+(\.\d+)?", phase_str):
        print("Error: phase must be a positive number like 6, 10.5, or 20.1.")
        sys.exit(1)

    phase = float(phase_str)
    tag = f"v{phase:.1f}.0" if "." in phase_str else f"v{int(phase)}.0.0"
    commit_msg = args.message or f"End of Phase{phase}"

    try:
        root = repo_root(dry=args.dry_run)
    except subprocess.CalledProcessError:
        print("Error: not inside a Git repository (or Git not installed).")
        sys.exit(1)

    # Determine branch
    branch = args.branch
    if not branch:
        try:
            branch = get_current_branch(dry=args.dry_run)
        except subprocess.CalledProcessError:
            print("Error: cannot determine current branch.")
            sys.exit(1)

    # Summary
    print("\n=== Plan ===")
    print(f"Repo    : {root}")
    print(f"Branch  : {branch}")
    print(f"Commit  : {commit_msg}")
    print(f"Tag     : {tag}")
    print(f"Dry-run : {args.dry_run}")
    print("============\n")

    if not args.yes:
        ans = input("Proceed? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    try:
        # Stage changes only if there are any
        if has_changes(dry=args.dry_run):
            run(["git", "add", "."], dry=args.dry_run)
            run(["git", "commit", "-m", commit_msg], dry=args.dry_run)
        else:
            print("No changes to commit (working tree clean). Skipping commit.")

        # Push branch
        run(["git", "push", "-u", "origin", branch], dry=args.dry_run)

        # Check tag existence
        if tag_exists(tag, dry=args.dry_run):
            print(f"Tag {tag} already exists. Skipping tag creation.")
        else:
            run(["git", "tag", "-a", tag, "-m", f"The end of Phase {phase}"], dry=args.dry_run)
            run(["git", "push", "origin", tag], dry=args.dry_run)

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
