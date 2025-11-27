"""Fix Unicode characters in example scripts for Windows compatibility."""

import sys
from pathlib import Path


def fix_unicode_in_file(filepath):
    """Replace Unicode characters with ASCII equivalents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Unicode characters
    content = content.replace('✓', '[OK]')
    content = content.replace('⚠️', '[WARNING]')
    content = content.replace('⚠', '[WARNING]')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {filepath}")


def main():
    """Fix all example scripts."""
    files = [
        'examples/jarvis_data_loading_example.py',
        'examples/multi_source_data_combination_example.py',
        'examples/application_ranking_example.py',
        'examples/experimental_planning_example.py',
        'examples/create_workflow_notebook.py',
    ]
    
    for filepath in files:
        path = Path(filepath)
        if path.exists():
            fix_unicode_in_file(path)
        else:
            print(f"Not found: {filepath}")
    
    print("\nAll files fixed!")


if __name__ == "__main__":
    main()
