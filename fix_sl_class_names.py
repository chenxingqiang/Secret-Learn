#!/usr/bin/env python3
"""
Fix class names in SL mode files (FL* -> SL*)
"""

import re
from pathlib import Path

def fix_sl_file(filepath):
    """Fix class name in a single SL file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix class name (FL -> SL)
        content = re.sub(r'^class FL(\w+):', r'class SL\1:', content, flags=re.MULTILINE)
        
        # Fix docstring references
        content = re.sub(r'Federated Learning (\w+)', r'Split Learning \1', content)
        content = content.replace('[FL]', '[SL]')
        content = content.replace('FL mode', 'SL mode')
        
        # Fix logger references
        content = re.sub(r'FL(\w+) with JAX', r'SL\1 with JAX', content)
        content = re.sub(r'FL(\w+) initialized', r'SL\1 initialized', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed"
        
        return False, "No changes"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("="*90)
    print("Fixing SL Mode Class Names (FL* -> SL*)")
    print("="*90)
    print()
    
    base_path = Path('/Users/xingqiangchen/jax-sklearn/secretlearn/SL')
    
    fixed_count = 0
    
    for py_file in base_path.rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
        
        success, message = fix_sl_file(py_file)
        
        if success:
            fixed_count += 1
            if fixed_count <= 20:
                print(f"✅ {py_file.parent.name}/{py_file.name}")
    
    if fixed_count > 20:
        print(f"... and {fixed_count - 20} more files")
    
    print()
    print(f"Fixed: {fixed_count} files")
    
    # Verify imports
    print("\nVerifying imports...")
    import_errors = 0
    for py_file in base_path.rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
        
        try:
            # Check if class name matches expected
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Extract algorithm name from filename
            filename = py_file.stem
            expected_class = 'SL' + ''.join(x.title() for x in filename.split('_'))
            
            # Check if class exists
            if f'class {expected_class}:' not in content:
                import_errors += 1
                if import_errors <= 5:
                    print(f"⚠️  {py_file.name}: Expected class {expected_class}")
        except:
            pass
    
    if import_errors == 0:
        print(f"✅ All SL class names are correct!")
    else:
        print(f"⚠️  {import_errors} files may have incorrect class names")
    
    print()

if __name__ == '__main__':
    main()

