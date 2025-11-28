#!/usr/bin/env python3
"""
Fix type annotations to use string annotations when SecretFlow not installed
"""

import re
from pathlib import Path

def fix_type_annotations(filepath):
    """Fix type annotations in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix: Use string annotations for SecretFlow types
        # This allows the code to be imported even when SecretFlow is not installed
        content = re.sub(r'devices:\s*Dict\[str,\s*PYU\]', r"devices: 'Dict[str, PYU]'", content)
        content = re.sub(r'heu:\s*Optional\[HEU\]', r"heu: 'Optional[HEU]'", content)
        content = re.sub(r'spu:\s*SPU([,\)])', r"spu: 'SPU'\1", content)
        content = re.sub(r'spu:\s*Optional\[SPU\]', r"spu: 'Optional[SPU]'", content)
        
        # Fix Union types with SecretFlow classes
        content = re.sub(r'Union\[FedNdarray,\s*VDataFrame\]', r"'Union[FedNdarray, VDataFrame]'", content)
        content = re.sub(r':\s*Union\[FedNdarray,\s*VDataFrame\]', r": 'Union[FedNdarray, VDataFrame]'", content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    print("="*90)
    print("Fixing Type Annotations for Optional SecretFlow")
    print("="*90)
    print()
    
    base = Path('/Users/xingqiangchen/jax-sklearn/secretlearn')
    fixed = 0
    
    for mode in ['FL', 'SS', 'SL']:
        mode_path = base / mode
        mode_fixed = 0
        
        for py_file in mode_path.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
            
            if fix_type_annotations(py_file):
                mode_fixed += 1
                fixed += 1
        
        if mode_fixed > 0:
            print(f"{mode}: Fixed {mode_fixed} files")
    
    print(f"\nTotal fixed: {fixed} files")
    print("✅ Type annotations are now string-based (works without SecretFlow)")
    print()

if __name__ == '__main__':
    main()
