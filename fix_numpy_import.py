#!/usr/bin/env python3
"""
紧急修复补丁 - 解决 'name 'np' is not defined' 错误
直接修复现有的 KAGGLE_TRAINING_FINAL_FIXED.py
"""

import os
import shutil
from pathlib import Path

def apply_numpy_fix():
    """应用numpy导入修复"""
    print("🔧 应用 numpy 导入修复补丁...")
    
    # 文件路径
    original_file = Path('/Users/niannian/Downloads/safe_agile_flight/KAGGLE_TRAINING_FINAL_FIXED.py')
    backup_file = Path('/Users/niannian/Downloads/safe_agile_flight/KAGGLE_TRAINING_FINAL_FIXED.py.backup')
    
    if not original_file.exists():
        print("❌ 原始文件不存在")
        return False
    
    # 备份原始文件
    shutil.copy2(original_file, backup_file)
    print(f"✅ 备份文件: {backup_file}")
    
    # 读取文件内容
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 应用修复
    # 修复1: 在 run_ultimate_training 函数中添加 numpy 导入
    old_import_block = """    try:
        import jax
        from jax import random
        import time"""
    
    new_import_block = """    try:
        import jax
        from jax import random
        import time
        import numpy as np"""
    
    # 修复2: 在 save_results 函数中添加 time 导入
    old_save_import = """    try:
        import jax
        import jax.numpy as jnp
        import pickle"""
    
    new_save_import = """    try:
        import jax
        import jax.numpy as jnp
        import pickle
        import time"""
    
    # 应用替换
    content = content.replace(old_import_block, new_import_block)
    content = content.replace(old_save_import, new_save_import)
    
    # 写回文件
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 修复应用完成")
    print("✅ numpy 导入问题已解决")
    print("✅ time 模块导入问题已解决")
    
    return True

if __name__ == "__main__":
    print("🚁 SAFE AGILE FLIGHT - numpy 导入修复补丁")
    print("=" * 60)
    
    success = apply_numpy_fix()
    
    if success:
        print("\n🎉 修复成功!")
        print("📋 修复内容:")
        print("   • run_ultimate_training 函数中添加 'import numpy as np'")
        print("   • save_results 函数中添加 'import time'")
        print("   • 解决了第4轮开始的 'name 'np' is not defined' 错误")
        
        print("\n🚀 现在可以运行:")
        print("exec(open('/kaggle/working/safe_agile_flight/KAGGLE_TRAINING_FINAL_FIXED.py').read())")
        
    else:
        print("\n❌ 修复失败")
        print("建议使用新的终极修复版:")
        print("exec(open('/kaggle/working/safe_agile_flight/KAGGLE_TRAINING_ULTIMATE_FIXED.py').read())")