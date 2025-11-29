#!/usr/bin/env python3
"""
Remove all Chinese characters from code files (comments and strings)
Keep only English in code
"""

import re
from pathlib import Path


# Translation mapping for common Chinese phrases
TRANSLATIONS = {
    # General
    'Batch': 'Batch',
    'convert': 'convert',
    'all': 'all',
    'examples': 'examples',
    'support': 'support',
    'multi-party': 'multi-party',
    'execution': 'execution',
    'following': 'following',
    'complete': 'complete',
    'pattern': 'pattern',
    
    # Functions
    'Extract': 'Extract',
    'class name': 'class name',
    'module': 'module',
    'info': 'info',
    'single': 'single',
    'file': 'file',
    
    # Code comments
    'Add': 'Add',
    'Modify': 'Modify',
    'Remove': 'Remove',
    'function': 'function',
    'start': 'start',
    'old': 'old',
    'initialization': 'initialization',
    'code block': 'code block',
    'training': 'training',
    'before': 'before',
    'position': 'position',
    'Find': 'Find',
    'error': 'error',
    'handling': 'handling',
    'Update': 'Update',
    'docstring': 'docstring',
    'waiting': 'waiting',
    'logic': 'logic',
    
    # Messages
    'cleanup': 'cleanup',
    'success': 'success',
    'failed': 'failed',
    'completed': 'completed',
    'correct': 'correct',
    'error': 'error',
}


def remove_chinese(text: str) -> str:
    """Remove or translate Chinese characters"""
    # Replace common Chinese phrases
    for chinese, english in TRANSLATIONS.items():
        text = text.replace(chinese, english)
    
    # Remove any remaining Chinese characters
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    
    return text


def clean_file(file_path: Path) -> bool:
    """Remove Chinese from a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original = f.read()
        
        # Check if has Chinese
        if not re.search(r'[\u4e00-\u9fff]', original):
            return False
        
        cleaned = remove_chinese(original)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        return True
        
    except Exception as e:
        print(f"Error: {file_path}: {e}")
        return False


def main():
    """Clean all Python files"""
    root = Path(__file__).parent.parent
    
    print("Removing Chinese characters from code files...")
    print()
    
    cleaned_count = 0
    
    # Clean scripts
    for py_file in (root / 'scripts').glob('*.py'):
        if clean_file(py_file):
            cleaned_count += 1
            print(f"✓ {py_file.relative_to(root)}")
    
    # Clean examples/SS
    for py_file in (root / 'examples' / 'SS').glob('*.py'):
        if clean_file(py_file):
            cleaned_count += 1
            print(f"✓ {py_file.relative_to(root)}")
    
    # Clean shell scripts
    for sh_file in (root / 'examples' / 'SS').glob('*.sh'):
        if clean_file(sh_file):
            cleaned_count += 1
            print(f"✓ {sh_file.relative_to(root)}")
    
    print()
    print("="*70)
    print(f"Cleaned {cleaned_count} files")
    print("="*70)


if __name__ == '__main__':
    main()

