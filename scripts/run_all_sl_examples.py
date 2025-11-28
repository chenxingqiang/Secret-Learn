#!/usr/bin/env python3
"""Run all SL examples and log results"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def check_log_success(log_file):
    """Check if a log file exists and shows success"""
    if not log_file.exists():
        return False
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if log shows success
            if 'SUCCESS' in content and 'Exit code: 0' in content:
                return True
    except:
        pass
    
    return False

def run_example(example_file, log_dir, force_run=False):
    """Run a single example and save its log"""
    example_name = example_file.stem
    log_file = log_dir / f"{example_name}.log"
    
    # Check if we can skip this example
    if not force_run and check_log_success(log_file):
        print(f"\n{'='*70}")
        print(f"Skipping: {example_name} (already successful)")
        print(f"{'='*70}")
        
        # Read elapsed time from existing log
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('# Execution time:'):
                        elapsed = float(line.split(':')[1].strip().rstrip('s'))
                        print(f"Status: SKIPPED (previous run: {elapsed:.2f}s)")
                        return True, elapsed
        except:
            pass
        print(f"Status: SKIPPED")
        return True, 0.0
    
    print(f"\n{'='*70}")
    print(f"Running: {example_name}")
    print(f"{'='*70}")
    
    start_time = time.time()

    try:
        # Run the example and capture output
        result = subprocess.run(
            [sys.executable, str(example_file)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            cwd=example_file.parent.parent.parent
        )

        elapsed_time = time.time() - start_time

        # Prepare log content
        log_content = f"""# Log for {example_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Execution time: {elapsed_time:.2f}s
# Exit code: {result.returncode}

{'='*70}
STDOUT:
{'='*70}
{result.stdout}

{'='*70}
STDERR:
{'='*70}
{result.stderr}

{'='*70}
SUMMARY:
{'='*70}
Exit code: {result.returncode}
Status: {'SUCCESS' if result.returncode == 0 else ' FAILED'}
Duration: {elapsed_time:.2f}s
"""

        # Write log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)

        # Print summary
        status = 'SUCCESS' if result.returncode == 0 else ' FAILED'
        print(f"Status: {status} ({elapsed_time:.2f}s)")
        print(f"Log: {log_file.relative_to(Path.cwd())}")

        return result.returncode == 0, elapsed_time

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        log_content = f"""# Log for {example_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Status: TIMEOUT after {elapsed_time:.2f}s

 Example timed out after 5 minutes
"""
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)

        print(f"Status:  TIMEOUT ({elapsed_time:.2f}s)")
        print(f"Log: {log_file.relative_to(Path.cwd())}")
        return False, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        log_content = f"""# Log for {example_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Status: ERROR

 Exception occurred: {str(e)}
"""
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)

        print(f"Status:  ERROR ({elapsed_time:.2f}s)")
        print(f"Log: {log_file.relative_to(Path.cwd())}")
        return False, elapsed_time

def main():
    base_dir = Path(__file__).parent.parent  # Go to project root
    examples_dir = base_dir / "examples" / "SL"
    log_dir = base_dir / "logs" / "examples" / "SL"
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all example files
    example_files = sorted([f for f in examples_dir.glob("*.py") if f.name != '__init__.py'])
    
    # Check for force run flag
    force_run = '--force' in sys.argv or '-f' in sys.argv
    
    print("="*70)
    print(f" Running All SL Examples: {len(example_files)} files")
    print("="*70)
    print(f"Log directory: {log_dir.relative_to(Path.cwd())}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not force_run:
        # Count existing successful logs
        existing_success = sum(1 for f in example_files 
                              if check_log_success(log_dir / f"{f.stem}.log"))
        print(f"Mode: Incremental (skip {existing_success} successful, use --force to rerun all)")
    else:
        print(f"Mode: Force rerun all examples")
    
    # Run all examples
    results = []
    total_start = time.time()
    skipped_count = 0
    run_count = 0
    
    for i, example_file in enumerate(example_files, 1):
        print(f"\n[{i}/{len(example_files)}] ", end='')
        success, elapsed = run_example(example_file, log_dir, force_run)
        results.append((example_file.name, success, elapsed))
        
        if success and elapsed == 0.0:  # Was skipped
            skipped_count += 1
        else:
            run_count += 1

    total_time = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    success_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - success_count
    
    print(f"\nTotal examples: {len(results)}")
    print(f"Successful: {success_count}")
    print(f" Failed: {failed_count}")
    print(f"⏭️  Skipped: {skipped_count} (already successful)")
    print(f"▶️  Executed: {run_count}")
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    if failed_count > 0:
        print(f"\n Failed examples:")
        for name, success, elapsed in results:
            if not success:
                print(f"   • {name}")

    print(f"\n📁 All logs saved to: {log_dir.relative_to(Path.cwd())}/")
    print("="*70)

    # Create summary file
    summary_file = log_dir / "_SUMMARY.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"SL Examples Execution Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Total: {len(results)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Skipped: {skipped_count}\n")
        f.write(f"Executed: {run_count}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"{'-'*70}\n")
        for name, success, elapsed in results:
            if elapsed == 0.0:
                status = '⏭️ '
                time_str = 'SKIPPED'
            else:
                status = '✅' if success else ''
                time_str = f'{elapsed:8.2f}s'
            f.write(f"{status} {name:50s} {time_str}\n")
    
    print(f"📄 Summary: {summary_file.relative_to(Path.cwd())}")
    print(f"\n💡 Tip: Use --force to rerun all examples (skip incremental check)")

if __name__ == "__main__":
    main()



