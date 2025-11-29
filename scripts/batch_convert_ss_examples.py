#!/usr/bin/env python3
"""
批量转换所有 SS 示例以支持多方运行

按照 adaboost_classifier.py 的完整模式转换
"""

import re
from pathlib import Path
from typing import Tuple, Optional


def extract_class_name(content: str) -> Optional[Tuple[str, str, str]]:
    """Extract class name and module info"""
    match = re.search(
        r'from secretlearn\.SS\.([^.]+)\.([^.]+) import (\w+)',
        content
    )
    if match:
        return match.groups()  # (module_path, module_name, class_name)
    return None


def convert_file(file_path: Path) -> bool:
    """Convert a single file"""
    try:
        with open(file_path, 'r') as f:
            original = f.read()
        
        # Skip if already converted
        if '--party' in original or 'parse_args' in original:
            return False
        
        info = extract_class_name(original)
        if not info:
            return False
        
        module_path, module_name, class_name = info
        
        content = original
        
        # 1. Add import os
        content = re.sub(
            r'(import numpy as np)',
            r'\1\nimport sys\nimport argparse\nimport time\nimport os',
            content
        )
        
        # 2. Add parse_args function
        parse_args_func = f'''

def parse_args():
    parser = argparse.ArgumentParser(description='SS {class_name}')
    parser.add_argument('--party', required=True, choices=['alice', 'bob'])
    parser.add_argument('--alice-addr', default='localhost:9494')
    parser.add_argument('--bob-addr', default='localhost:9495')
    return parser.parse_args()
'''
        
        content = re.sub(
            r'(\nfrom secretlearn\.SS\.[^\n]+\n)',
            r'\1' + parse_args_func,
            content
        )
        
        # 3. Modify main function start
        main_start = f'''def main():
    """Main function"""
    args = parse_args()
    party_name = args.party
    
    print("="*70)
    print(f" {class_name} - Party: {{party_name.upper()}}")
    print("="*70)
    print(f"\\n[{{party_name}}] PID: {{os.getpid()}}")
    
    try:
        print(f"\\n[{{party_name}}] [1/5] Initializing...")
        
        # Cluster config
        alice_host, alice_port = args.alice_addr.split(':')
        bob_host, bob_port = args.bob_addr.split(':')
        
        cluster_config = {{
            'parties': {{
                'alice': {{'address': f'{{alice_host}}:{{alice_port}}', 'listen_addr': f'0.0.0.0:{{alice_port}}'}},
                'bob': {{'address': f'{{bob_host}}:{{bob_port}}', 'listen_addr': f'0.0.0.0:{{bob_port}}'}},
            }},
            'self_party': party_name
        }}
        
        sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
        
        alice = sf.PYU('alice')
        bob = sf.PYU('bob')
        
        # Sync mechanism
        ready_file = f'/tmp/sf_{{party_name}}.lock'
        other_ready = f'/tmp/sf_{{"bob" if party_name == "alice" else "alice"}}.lock'
        open(ready_file, 'w').close()
        
        print(f"[{{party_name}}] Waiting for peer...")
        for _ in range(30):
            if os.path.exists(other_ready):
                break
            time.sleep(1)
        else:
            print(f"[{{party_name}}] ✗ Timeout")
            sys.exit(1)
        
        time.sleep(1)
        
        # SPU
        spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob'], runtime_config={{'protocol': 'SEMI2K'}}))
        print(f"[{{party_name}}] ✓ Initialized")
        try:
            os.remove(ready_file)
        except:
            pass
        
        if party_name == 'alice':
            # Alice runs the training
            print(f"\\n[{{party_name}}] [2/5] Preparing data...")'''
        
        content = re.sub(
            r'def main\(\):.*?print\("="\*70\)\s*\n\s*# Step 1:[^\n]*\n\s*print\([^\n]+\n',
            main_start + '\n',
            content,
            flags=re.DOTALL
        )
        
        # 4. Remove carol (3-party -> 2-party)
        content = re.sub(r"'carol': \{[^}]+\},?\s*", '', content)
        content = re.sub(r'\s*carol = sf\.PYU\([\'"]carol[\'"]\)', '', content)
        content = re.sub(r',\s*carol', '', content)
        content = re.sub(r'carol,\s*', '', content)
        content = re.sub(r"parties=\['alice', 'bob', 'carol'\]", "parties=['alice', 'bob']", content)
        content = re.sub(r"'protocol': 'ABY3'", "'protocol': 'SEMI2K'", content)
        
        # 5. Remove old initialization code
        content = re.sub(
            r'\s*# For single-node testing.*?print\("  ✓ SecretFlow initialized[^\n]+\n',
            '',
            content,
            flags=re.DOTALL
        )
        
        # 6. Add bob's waiting logic before cleanup
        content = re.sub(
            r'(\n\s+# Cleanup)',
            r'''
    else:
        # Bob waits and participates
        print(f"\n[{{party_name}}] Waiting for alice...")
        time.sleep(300)
        print(f"[{{party_name}}] ✓ Done")
\1''',
            content
        )
        
        # 7. Add error handling
        content = re.sub(
            r'(\s+# Cleanup\s+sf\.shutdown\(\).*?\n\n)(if __name__)',
            r'''\1
    except KeyboardInterrupt:
        print(f"\n[{{party_name}}] Interrupted")
        try:
            sf.shutdown()
        except:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"\n[{{party_name}}] Error: {{e}}")
        import traceback
        traceback.print_exc()
        try:
            sf.shutdown()
        except:
            pass
        sys.exit(1)


\2''',
            content
        )
        
        # 8. Update docstring
        content = re.sub(
            r'("""[^"]*?)\n\nThis example demonstrates',
            r'\1\n\nUsage:\n    Terminal 1: python ' + file_path.name + r' --party bob\n    Terminal 2: python ' + file_path.name + r' --party alice\n\nThis example demonstrates',
            content
        )
        
        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error: {file_path.name}: {e}")
        return False


def main():
    examples_dir = Path(__file__).parent.parent / 'examples' / 'SS'
    
    print("Batch converting SS examples...")
    print()
    
    skip = {'__init__.py', 'adaboost_classifier.py'}
    
    converted = 0
    total = 0
    
    for py_file in sorted(examples_dir.glob('*.py')):
        if py_file.name in skip:
            continue
        
        total += 1
        if convert_file(py_file):
            converted += 1
            print(f"✓ {py_file.name}")
    
    print()
    print("="*70)
    print(f"Converted: {converted}/{total}")
    print("="*70)


if __name__ == '__main__':
    main()

