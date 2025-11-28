#!/usr/bin/env python3
"""
Run examples for all three privacy modes

Usage:
  python run_all_examples.py           # Run all modes
  python run_all_examples.py FL        # Run FL only
  python run_all_examples.py FL SS     # Run FL and SS
  python run_all_examples.py --force   # Force rerun all
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_mode_examples(mode, force=False):
    """Run examples for specified mode"""
    script_name = f"run_all_{mode.lower()}_examples.py"
    script_path = Path(__file__).parent / script_name  # scripts/run_all_*_examples.py
    
    if not script_path.exists():
        print(f" Script not found: {script_name}")
        return False
    
    print("\n" + "="*80)
    print(f"🚀 Running {mode} Mode Examples")
    print("="*80)
    
    cmd = [sys.executable, str(script_path)]
    if force:
        cmd.append('--force')
    
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except Exception as e:
        print(f" Execution failed: {str(e)}")
        return False

def main():
    print("="*80)
    print("Secret-Learn Examples Batch Runner")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse arguments
    args = sys.argv[1:]
    force = '--force' in args or '-f' in args
    
    # Remove force flag, remaining are modes
    modes_to_run = [arg for arg in args if arg not in ['--force', '-f']]
    
    # If no modes specified, run all
    if not modes_to_run:
        modes_to_run = ['FL', 'SS', 'SL']
    else:
        # Convert to uppercase
        modes_to_run = [m.upper() for m in modes_to_run]
    
    print(f"Modes to run: {', '.join(modes_to_run)}")
    print(f"Force rerun: {'Yes' if force else 'No'}")
    print()
    
    # Run each mode
    results = {}
    for mode in modes_to_run:
        if mode not in ['FL', 'SS', 'SL']:
            print(f"⚠️  Unknown mode: {mode}")
            continue
        
        success = run_mode_examples(mode, force)
        results[mode] = success
    
    # Summary
    print("\n" + "="*80)
    print("Overall Results")
    print("="*80)
    print()
    
    for mode in modes_to_run:
        if mode in results:
            status = "Success" if results[mode] else " Failed"
            print(f"  {mode}: {status}")
    
    print()
    print("="*80)
    print("View detailed logs:")
    print("  - logs/examples/FL/_SUMMARY.txt")
    print("  - logs/examples/SS/_SUMMARY.txt")
    print("  - logs/examples/SL/_SUMMARY.txt")
    print("="*80)

if __name__ == "__main__":
    main()

