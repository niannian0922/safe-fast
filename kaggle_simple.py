import subprocess
import sys
import os
import shutil

print("🚁 Safe Agile Flight - Kaggle简化版")
print("解决git clone目录已存在问题...")

# 1. 清理已存在目录
if os.path.exists('/kaggle/working/safe_agile_flight'):
    shutil.rmtree('/kaggle/working/safe_agile_flight')
    print("✅ 清理完成")

# 2. 重新克隆
subprocess.run(['git', 'clone', 'https://github.com/niannian0922/safe_agile_flight.git', '/kaggle/working/safe_agile_flight'], check=True)
print("✅ 克隆成功")

# 3. 添加路径
sys.path.append('/kaggle/working/safe_agile_flight')

# 4. 安装依赖
deps = ['jax[cuda12_pip]', 'flax', 'optax', 'ml-collections', 'chex']
for dep in deps:
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True, capture_output=True)
        print(f"✅ {dep}")
    except:
        print(f"⚠️ {dep} 安装失败")

# 5. 执行训练
exec(open('/kaggle/working/safe_agile_flight/kaggle_one_click_train.py').read())