# Kaggle 端到端训练完整指令

## 1. 创建 Kaggle Notebook

1. 登录 Kaggle (https://www.kaggle.com)
2. 点击 "Code" -> "New Notebook"  
3. 选择 "Python" 作为语言
4. 在 Settings 中启用：
   - GPU P100 或 T4 (推荐)
   - Internet 连接

## 2. Notebook 设置

### 环境配置
- **语言**: Python
- **加速器**: GPU (强烈推荐)
- **网络**: Internet On (必需，用于克隆代码)

### 数据集(可选)
如果您有预训练的检查点或配置文件，可以上传为数据集。

## 3. 完整训练步骤

将以下代码分别粘贴到不同的 Cell 中：

### Cell 1: 环境设置
```python
# 安装依赖
import subprocess
import sys

packages = [
    "jax[cuda12_pip]==0.4.20",
    "jaxlib==0.4.20", 
    "flax==0.8.0",
    "jraph==0.0.6.dev0",
    "optax==0.1.7",
    "ml-collections==0.1.1",
    "qpax",
    "chex==0.1.84"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
```

### Cell 2: 克隆项目
```python
import subprocess
import os
import sys

# 克隆项目
if not os.path.exists('/kaggle/working/safe_agile_flight'):
    subprocess.run([
        'git', 'clone', 
        'https://github.com/niannian0922/safe_agile_flight.git',
        '/kaggle/working/safe_agile_flight'
    ], check=True)

# 设置路径
sys.path.append('/kaggle/working/safe_agile_flight')
```

### Cell 3-12: 训练代码
将 `kaggle_training_notebook.py` 中的相应部分分别复制到不同的 Cell。

## 4. 执行训练

1. 运行所有 Cell (Ctrl+Shift+Enter 或点击 "Run All")
2. 监控训练进度和损失曲线
3. 等待训练完成

## 5. 结果保存

训练完成后，以下文件将保存在 Kaggle 输出目录：
- `trained_model.pkl`: 训练好的模型参数
- `training_curves.png`: 训练曲线图
- Notebook 输出: 详细的训练日志

## 6. 预期训练时间

根据配置：
- **GPU P100**: 约 2-4 小时
- **GPU T4**: 约 3-6 小时  
- **CPU**: 不推荐 (时间过长)

## 7. 故障排除

### 常见问题：
1. **JAX CUDA 版本问题**: 确保安装了正确的 CUDA 版本
2. **内存不足**: 减少批次大小或训练时间步
3. **网络连接问题**: 确保启用了 Internet 连接

### 解决方案：
- 重启 Notebook 并重新运行
- 检查 GPU 可用性：`jax.devices()`
- 监控内存使用情况

## 8. 高级选项

### 修改训练配置
在 Cell 4 中，您可以修改配置：
```python
# 修改训练参数
config.training.num_epochs = 2000  # 增加训练轮数
config.training.batch_size = 16    # 调整批次大小
config.training.learning_rate = 3e-4  # 调整学习率
```

### 启用检查点
```python
# 添加定期保存
if epoch % 500 == 0:
    checkpoint_path = f'/kaggle/working/checkpoint_epoch_{epoch}.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(trained_params, f)
```

## 9. 结果分析

训练完成后，您将获得：
- 完整的训练历史曲线
- 模型性能评估报告  
- 保存的模型参数文件
- 详细的训练统计信息

训练成功的标志：
- 损失函数持续下降
- 成功率 > 80%
- 梯度范数稳定
- 无 NaN 或 Inf 值

## 10. 下一步

训练完成后，您可以：
1. 下载训练好的模型
2. 在本地进行进一步测试
3. 部署到真实无人机系统
4. 进行消融研究和性能优化