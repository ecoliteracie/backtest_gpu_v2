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

        # Check tag existence and create/push if needed
        if tag_exists(tag, dry=args.dry_run):
            print(f"Tag {tag} already exists. Skipping tag creation.")
        else:
            print(f"Creating and pushing tag {tag}...")
            try:
                # Create the tag with more detailed output
                print(f"  - Creating annotated tag: {tag}")
                try:
                    # First try to delete the tag if it exists locally
                    run(["git", "tag", "-d", tag], dry=args.dry_run)
                    print(f"  - Removed existing local tag {tag}")
                except subprocess.CalledProcessError:
                    pass  # Tag didn't exist locally, which is fine
                
                # Create the new tag
                run(["git", "tag", "-a", tag, "-m", f"The end of Phase {phase}"], dry=args.dry_run)
                print(f"  - Created local tag {tag}")
                
                # Verify the tag was created locally
                if not args.dry_run:
                    try:
                        run(["git", "show", tag], dry=args.dry_run)
                        print(f"  - Verified tag {tag} exists locally")
                    except subprocess.CalledProcessError as e:
                        print(f"  - ERROR: Failed to verify local tag: {e}")
                        raise
                
                # Push the tag with retries and force push if needed
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"\n  --- Attempt {attempt}/{max_retries} ---")
                        print(f"  - Pushing tag to remote: {tag}")
                        
                        # First, check remote connection
                        print("  - Checking remote connection...")
                        run(["git", "remote", "-v"], dry=args.dry_run)
                        
                        # Push with explicit refspec and force
                        print("  - Pushing with force...")
                        run(["git", "push", "origin", f"refs/tags/{tag}:refs/tags/{tag}", "--force"], dry=args.dry_run)
                        print(f"  ✓ Successfully pushed tag {tag} to remote")
                        break
                    except subprocess.CalledProcessError as e:
                        if attempt == max_retries:
                            print(f"  - Error pushing tag: {e}")
                            # Try one last time with --force-with-lease for safety
                            try:
                                run(["git", "push", "origin", f"refs/tags/{tag}", "--force-with-lease"], dry=args.dry_run)
                                print(f"  - Successfully force-pushed tag {tag} with --force-with-lease")
                                break
                            except subprocess.CalledProcessError as e2:
                                print(f"  - Final attempt failed: {e2}")
                                raise
                        print(f"  - Push failed, retrying ({attempt}/{max_retries})...")
                        import time
                        time.sleep(1)  # Wait a bit before retrying
                
                # Verify the tag exists on remote
                if not args.dry_run:
                    print("  - Verifying tag on remote...")
                    try:
                        # Fetch all tags from remote first
                        run(["git", "fetch", "--tags", "--force"], dry=args.dry_run)
                        
                        # Check if tag exists on remote
                        remote_tags = run(["git", "ls-remote", "--tags", "origin"], dry=args.dry_run).stdout
                        if f"refs/tags/{tag}" not in remote_tags:
                            # Try one more time with a fresh fetch
                            run(["git", "fetch", "--tags", "--force"], dry=args.dry_run)
                            remote_tags = run(["git", "ls-remote", "--tags", "origin"], dry=args.dry_run).stdout
                            if f"refs/tags/{tag}" not in remote_tags:
                                raise RuntimeError(f"Tag {tag} was not found on remote after push")
                        
                        print(f"\n✓ Successfully created and pushed tag {tag}")
                        print(f"   Remote URL: https://github.com/ecoliteracie/backtest_gpu_v2/releases/tag/{tag}")
                    except Exception as e:
                        print(f"\n⚠ Warning: Could not verify tag on remote: {e}")
                        print("  The tag might still have been pushed. Please check GitHub manually.")
                        print(f"  You can also try running: git push origin {tag} --force")
                
            except Exception as e:
                print(f"\n✗ Failed to create/push tag {tag}")
                print(f"Error: {str(e)}")
                print("\nYou may need to manually create and push the tag with:")
                print(f"  git tag -a {tag} -m \"The end of Phase {phase}\"")
                print(f"  git push origin {tag}")
                raise

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
