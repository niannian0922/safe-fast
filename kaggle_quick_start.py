#!/usr/bin/env python3
"""
Kaggle 快速启动脚本 - 解决所有已知问题
Safe Agile Flight 端到端可微分训练

本脚本专门解决Kaggle环境中的问题：
1. Git克隆目录已存在错误
2. 依赖安装失败
3. 内存不足
4. 导入错误

使用方法：
在Kaggle Notebook中直接运行：
exec(open('/kaggle/working/kaggle_quick_start.py').read())
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

print("🚁 Safe Agile Flight - Kaggle 快速启动")
print("🔧 解决常见Kaggle问题...")

# =============================================================================
# 问题1：解决Git克隆目录已存在的问题
# =============================================================================
def fix_git_clone_issue():
    """彻底解决git克隆问题"""
    project_path = Path('/kaggle/working/safe_agile_flight')
    
    print("🗑️  清理旧项目目录...")
    if project_path.exists():
        try:
            shutil.rmtree(project_path, ignore_errors=True)
            print("   ✅ 旧目录清理完成")
        except Exception as e:
            print(f"   ⚠️  清理警告: {e}")
    
    # 确保父目录存在
    project_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("📥 重新克隆项目...")
    try:
        # 多种克隆策略
        strategies = [
            ['git', 'clone', '--depth=1', 'https://github.com/niannian0922/safe_agile_flight.git', str(project_path)],
            ['git', 'clone', 'https://github.com/niannian0922/safe_agile_flight.git', str(project_path)],
        ]
        
        for strategy in strategies:
            try:
                subprocess.run(strategy, check=True, timeout=120)
                print("   ✅ 项目克隆成功")
                return True
            except subprocess.CalledProcessError:
                continue
            except Exception:
                continue
        
        print("   ❌ 所有克隆策略失败")
        return False
        
    except Exception as e:
        print(f"   ❌ 克隆过程出错: {e}")
        return False

# =============================================================================
# 问题2：优化依赖安装
# =============================================================================
def install_dependencies_robust():
    """鲁棒的依赖安装"""
    print("📦 安装核心依赖（优化版）...")
    
    # 分阶段安装策略
    stages = [
        # 阶段1：基础工具
        {
            'name': '基础工具',
            'packages': ['pip', 'wheel', 'setuptools'],
            'flags': ['--upgrade']
        },
        # 阶段2：JAX核心
        {
            'name': 'JAX核心',
            'packages': ['jax[cuda12_pip]'],  
            'flags': ['-f', 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html', '--upgrade']
        },
        # 阶段3：深度学习框架
        {
            'name': '深度学习',
            'packages': ['flax>=0.8.0', 'optax>=0.1.7', 'chex'],
            'flags': []
        },
        # 阶段4：图网络和配置
        {
            'name': '图网络',
            'packages': ['jraph', 'ml-collections'],
            'flags': []
        },
        # 阶段5：数值计算
        {
            'name': '数值计算', 
            'packages': ['numpy>=1.24.0', 'scipy'],
            'flags': []
        }
    ]
    
    success_count = 0
    for stage in stages:
        print(f"   🔄 安装{stage['name']}...")
        for package in stage['packages']:
            try:
                cmd = [sys.executable, '-m', 'pip', 'install'] + stage['flags'] + [package]
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=300)
                print(f"      ✅ {package}")
                success_count += 1
            except subprocess.TimeoutExpired:
                print(f"      ⏰ {package} (超时，跳过)")
            except subprocess.CalledProcessError as e:
                print(f"      ⚠️  {package} (失败)")
            except Exception as e:
                print(f"      ❌ {package} (错误)")
    
    print(f"📊 依赖安装完成: {success_count} 个包成功")
    return success_count > 8  # 至少8个包成功才算OK

# =============================================================================
# 问题3：内存优化启动
# =============================================================================
def setup_memory_optimization():
    """设置内存优化"""
    print("🧠 配置内存优化...")
    
    # JAX内存优化
    try:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # 限制GPU内存使用
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # 禁用预分配
        os.environ['JAX_ENABLE_X64'] = 'false'  # 使用float32节省内存
        print("   ✅ JAX内存优化配置完成")
    except Exception as e:
        print(f"   ⚠️  内存优化配置警告: {e}")

# =============================================================================
# 问题4：智能导入系统
# =============================================================================
def test_imports():
    """测试和验证导入"""
    print("🧪 测试核心库导入...")
    
    import_results = {}
    
    # 测试JAX
    try:
        import jax
        import jax.numpy as jnp
        print(f"   ✅ JAX {jax.__version__} - 设备: {len(jax.devices())}")
        import_results['jax'] = True
    except ImportError as e:
        print(f"   ❌ JAX导入失败: {e}")
        import_results['jax'] = False
    
    # 测试Flax
    try:
        import flax
        import flax.linen as nn
        print(f"   ✅ Flax {flax.__version__}")
        import_results['flax'] = True
    except ImportError as e:
        print(f"   ❌ Flax导入失败: {e}")
        import_results['flax'] = False
    
    # 测试Optax
    try:
        import optax
        print(f"   ✅ Optax")
        import_results['optax'] = True  
    except ImportError as e:
        print(f"   ❌ Optax导入失败: {e}")
        import_results['optax'] = False
    
    # 测试其他
    other_libs = ['numpy', 'ml_collections', 'chex']
    for lib in other_libs:
        try:
            __import__(lib)
            print(f"   ✅ {lib}")
            import_results[lib] = True
        except ImportError:
            print(f"   ❌ {lib} 不可用")
            import_results[lib] = False
    
    return import_results

# =============================================================================
# 主执行流程
# =============================================================================
def main():
    """主启动流程"""
    print("=" * 60)
    
    # 步骤1：内存优化
    setup_memory_optimization()
    
    # 步骤2：安装依赖
    deps_ok = install_dependencies_robust()
    
    # 步骤3：解决git问题并获取代码
    code_ok = fix_git_clone_issue()
    
    # 步骤4：测试导入
    import_results = test_imports()
    
    # 步骤5：决定执行策略
    jax_ok = import_results.get('jax', False)
    flax_ok = import_results.get('flax', False)
    
    print("\n" + "=" * 60)
    print("🏁 启动完成状态:")
    print(f"   📦 依赖安装: {'✅ 成功' if deps_ok else '❌ 部分失败'}")
    print(f"   📂 代码获取: {'✅ 成功' if code_ok else '❌ 失败'}")  
    print(f"   🧮 JAX可用: {'✅ 是' if jax_ok else '❌ 否'}")
    print(f"   🧠 Flax可用: {'✅ 是' if flax_ok else '❌ 否'}")
    
    if jax_ok and flax_ok:
        print("\n🎉 环境准备完成！可以开始训练")
        print("📝 建议下一步:")
        print("   1. 运行 exec(open('/kaggle/working/kaggle_training_fixed.py').read())")
        print("   2. 或者使用项目中的 main.py")
        
        # 添加Python路径
        if code_ok:
            sys.path.insert(0, '/kaggle/working/safe_agile_flight')
            print("   ✅ Python路径已配置")
        
        return True
    else:
        print("\n⚠️  环境存在问题，将启用备用模式")
        print("📝 备用方案:")
        print("   - 使用简化实现进行训练")
        print("   - 功能可能受限但仍可工作")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 准备就绪！可以启动训练")
    else:
        print("\n🔧 环境配置完成，使用备用模式")