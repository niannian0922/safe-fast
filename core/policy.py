"""
安全敏捷飞行系统的策略网络。

本模块实现神经网络策略，结合以下研究的见解：
1. GCBF+ : 使用图神经网络的分布式安全控制
2. DiffPhysDrone : 端到端基于视觉的飞行控制

策略架构支持单智能体和多智能体场景，
具有用于时间一致性和安全意识的循环记忆。
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Optional, Tuple, Callable, Any
import chex
from flax import linen as nn
from flax import struct
import optax


# =============================================================================
# 策略状态表示
# =============================================================================

@struct.dataclass
class PolicyState:
    """带记忆的策略网络的状态表示。"""
    rnn_state: chex.Array  # 循环网络的隐藏状态
    step_count: int  # 课程学习的当前步骤计数器
    action_history: chex.Array  # 用于平滑性的先前动作历史


@struct.dataclass 
class PolicyParams:
    """策略网络配置的参数。"""
    # 网络架构
    hidden_dims: Tuple[int, ...] = (256, 256)
    rnn_hidden_size: int = 256
    activation: str = "relu"
    use_rnn: bool = True
    
    # 控制约束
    max_thrust: float = 0.8
    thrust_smoothing: float = 0.95  # 指数平滑因子
    
    # 安全集成
    enable_cbf_integration: bool = True
    safety_margin: float = 0.1
    
    # 训练超参数
    action_penalty_coef: float = 0.01
    smoothness_penalty_coef: float = 0.001


# =============================================================================
# 基础策略网络
# =============================================================================

class MLPBlock(nn.Module):
    """具有可配置激活函数的多层感知器块。"""
    
    features: int
    activation: str = "relu"
    use_bias: bool = True
    dropout_rate: float = 0.0
    
    def setup(self):
        self.dense = nn.Dense(self.features, use_bias=self.use_bias)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = None
            
    def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
        x = self.dense(x)
        
        # 应用激活函数
        if self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "tanh":
            x = nn.tanh(x)
        elif self.activation == "swish":
            x = nn.swish(x)
        elif self.activation == "gelu":
            x = nn.gelu(x)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
            
        # 如果指定则应用dropout
        if self.dropout is not None:
            x = self.dropout(x, deterministic=not training)
            
        return x


class PolicyNetworkMLP(nn.Module):
    """
    单智能体控制的基础MLP策略网络。
    
    基于DiffPhysDrone的轻量级架构，但通过JAX/Flax实现增强，
    并改善了数值稳定性。
    """
    
    params: PolicyParams
    output_dim: int = 3  # 3D推力命令
    
    def setup(self):
        # 使用列表推导式创建MLP层（兼容Flax）
        self.layers = [
            MLPBlock(
                features=features,
                activation=self.params.activation,
                dropout_rate=0.1 if i < len(self.params.hidden_dims) - 1 else 0.0
            )
            for i, features in enumerate(self.params.hidden_dims)
        ]
        
        # 带tanh激活的输出层用于有界控制
        self.output_layer = nn.Dense(self.output_dim)
    
    def __call__(
        self, 
        observations: chex.Array,  # [batch_size, obs_dim] 
        training: bool = False
    ) -> chex.Array:
        """
        通过MLP策略的前向传递。
        
        参数：
            observations: 输入观测值
            training: 是否在训练模式
            
        返回：
            [-1, 1]范围内的控制命令
        """
        x = observations
        
        # 通过隐藏层前向传递
        for layer in self.layers:
            x = layer(x, training=training)
        
        # 输出层
        x = self.output_layer(x)
        
        # 应用tanh获得有界输出
        control_output = nn.tanh(x)
        
        return control_output


# 循环策略网络（受DiffPhysDrone启发）卷积+循环混合架构

class PolicyNetworkRNN(nn.Module):
    """
    用于时间一致性的循环策略网络。
    
    实现DiffPhysDrone的CRNN架构的关键见解：
    - 用于时间记忆和规划一致性的GRU
    - 用于实时部署的轻量级设计
    - 集成动作平滑
    """
    
    params: PolicyParams
    output_dim: int = 3
    
    def setup(self):
        # 使用列表推导式的特征提取层（兼容Flax）
        self.feature_layers = [
            MLPBlock(features=features, activation=self.params.activation)
            for features in self.params.hidden_dims[:-1]  # 除最后一个外的所有
        ]
        
        # 不需要设置RNN层，我们在__call__中直接使用GRUCell
        
        # 输出投影
        final_hidden_dim = self.params.hidden_dims[-1] if self.params.hidden_dims else self.params.rnn_hidden_size
        self.output_projection = nn.Dense(final_hidden_dim)
        self.control_head = nn.Dense(self.output_dim)
        
        # 动作历史集成以获得平滑性
        self.action_history_proj = nn.Dense(self.params.rnn_hidden_size // 4)
    
    def __call__(
        self,
        observations: chex.Array,  # [batch_size, seq_len, obs_dim]
        rnn_state: chex.Array,     # [batch_size, rnn_hidden_size] 
        action_history: Optional[chex.Array] = None,  # [batch_size, history_len, 3]
        training: bool = False
    ) -> Tuple[chex.Array, chex.Array]:
        """
        通过RNN策略的前向传递。
        
        参数：
            observations: 输入观测序列
            rnn_state: 上一个RNN隐藏状态
            action_history: 用于平滑的先前动作历史
            training: 训练模式标志
            
        返回：
            (control_commands, new_rnn_state)
        """
        batch_size = observations.shape[0]
        
        # 特征提取:进行初步的特征提取。这一步将原始的、可能维度很高的观测数据，转换成了更抽象、信息密度更高的特征向量 x
        x = observations
        for layer in self.feature_layers:
            x = layer(x, training=training)
        
        # 处理动作历史以获得平滑性（DiffPhysDrone见解）
        if action_history is not None:
            action_features = self.action_history_proj(
                action_history.reshape(batch_size, -1)
            )
            # 与当前特征组合
            x = jnp.concatenate([x, action_features], axis=-1)
        
        # RNN处理 - 在时间维度上扫描
        rnn_cell = nn.GRUCell(features=self.params.rnn_hidden_size)#实例化一个 GRU（门控循环单元）。GRU 是一种比基础 RNN 更先进的循环单元，它内部有“更新门”和“重置门”，能够更有效地学习长期依赖关系，并缓解梯度消失问题。
        new_rnn_state, rnn_output = rnn_cell(rnn_state, x)#GRU 单元内部进行复杂的门控计算，融合新旧信息。
        #是 GRU 对当前情况的一个高度浓缩的总结，但它还不是最终的控制指令
        # 输出投影
        x = self.output_projection(rnn_output)
        x = nn.relu(x)
        
        # 带有界输出的控制头
        control_output = self.control_head(x)#代码将它通过另外两层 MLP ,将其“解码”成一个 3 维的推力向量 control_output。
        control_output = nn.tanh(control_output)  # 绑定到[-1, 1]
        
        return control_output, new_rnn_state


# =============================================================================
# 策略工厂和实用程序
# =============================================================================

def create_policy_network(
    params: PolicyParams,
    network_type: str = "mlp",
    output_dim: int = 3
) -> nn.Module:
    """
    创建策略网络的工厂函数。
    
    参数：
        params: 策略参数
        network_type: 网络类型（"mlp" 或 "rnn"）
        output_dim: 输出维度
        
    返回：
        策略网络实例
    """
    if network_type == "mlp":
        return PolicyNetworkMLP(params=params, output_dim=output_dim)
    elif network_type == "rnn":
        return PolicyNetworkRNN(params=params, output_dim=output_dim)
    else:
        raise ValueError(f"未知的网络类型: {network_type}")


def init_policy_state(
    policy_params: PolicyParams,
    rng_key: chex.PRNGKey,
    batch_size: int = 1
) -> PolicyState:
    """初始化策略状态。"""
    rnn_state = jnp.zeros((batch_size, policy_params.rnn_hidden_size))
    action_history = jnp.zeros((batch_size, 3, 3))  # 最后3个动作
    
    return PolicyState(
        rnn_state=rnn_state,
        step_count=0,
        action_history=action_history
    )


def apply_control_constraints(
    raw_control: chex.Array,
    params: PolicyParams,
    previous_action: Optional[chex.Array] = None
) -> chex.Array:
    """
    应用控制约束和平滑。
    
    实现来自DiffPhysDrone的控制处理：
    - 推力幅度约束
    - 用于稳定性的时间平滑
    """
    # 缩放到实际推力范围
    control_output = raw_control * params.max_thrust
    
    # 如果有可用的上一个动作，应用指数平滑
    if previous_action is not None:
        control_output = (
            params.thrust_smoothing * previous_action + 
            (1.0 - params.thrust_smoothing) * control_output
        )
    
    # 强制硬约束
    control_output = jnp.clip(control_output, -params.max_thrust, params.max_thrust)
    
    return control_output


# =============================================================================
# 策略评估和实用程序
# =============================================================================

@jax.jit
def evaluate_policy_mlp(
    policy: nn.Module,
    params: chex.Array,
    observations: chex.Array,
    training: bool = False
) -> chex.Array:
    """MLP的JIT编译策略评估。"""
    return policy.apply(params, observations, training=training)


@jax.jit 
def evaluate_policy_rnn(
    policy: nn.Module,
    params: chex.Array,
    observations: chex.Array,
    rnn_state: chex.Array,
    action_history: Optional[chex.Array] = None,
    training: bool = False
) -> Tuple[chex.Array, chex.Array]:
    """RNN的JIT编译策略评估。"""
    return policy.apply(
        params, observations, rnn_state, action_history, training=training
    )


def compute_policy_loss_components(
    predicted_actions: chex.Array,
    target_actions: chex.Array,
    action_history: chex.Array,
    params: PolicyParams
) -> Tuple[chex.Array, dict]:
    """
    遵循DiffPhysDrone方法计算策略损失组件。
    
    参数：
        predicted_actions: 网络输出动作
        target_actions: 目标动作（来自QP求解器）
        action_history: 用于平滑性的先前动作
        params: 策略参数
        
    返回：
        (total_loss, loss_dict)
    """
    # 动作跟踪损失（主要目标）
    action_loss = jnp.mean((predicted_actions - target_actions) ** 2)#jnp.mean 计算了所有这些平方误差的平均值。这就是标准的均方误差 (Mean Squared Error, MSE)，是回归任务中最常用的损失函数。
    
    # 动作幅度惩罚（能量效率）
    magnitude_loss = jnp.mean(jnp.sum(predicted_actions ** 2, axis=-1))
    #将推力向量 [tx, ty, tz] 的每个分量平方，得到 [tx^2, ty^2, tz^2]。
    #jnp.sum(..., axis=-1)：沿着最后一个维度（即 xyz 分量）求和，得到 tx^2 + ty^2 + tz^2。这正是向量模长（距离原点的距离）的平方。
    #jnp.mean(...)：计算批处理中所有动作模长平方的平均值。
    
    # 平滑性惩罚（基于动作导数）
    if action_history.shape[-2] > 1:  # 至少需要2个历史步骤
        action_derivatives = jnp.diff(action_history, axis=-2)#jnp.diff 函数计算了一个数组中沿指定轴的 N 阶差分
        smoothness_loss = jnp.mean(jnp.sum(action_derivatives ** 2, axis=-1))
    else:
        smoothness_loss = 0.0
    
    # 组合损失
    total_loss = (
        action_loss + 
        params.action_penalty_coef * magnitude_loss +
        params.smoothness_penalty_coef * smoothness_loss
    )
    
    loss_dict = {
        "action_loss": action_loss,
        "magnitude_loss": magnitude_loss, 
        "smoothness_loss": smoothness_loss,
        "total_loss": total_loss
    }
    
    return total_loss, loss_dict


# =============================================================================
# 测试和验证
# =============================================================================

def validate_policy_implementation():
    """验证策略网络实现。"""
    print("🧪 验证策略网络实现...")
    
    # 创建测试参数
    params = PolicyParams(
        hidden_dims=(128, 64),
        rnn_hidden_size=128,
        use_rnn=True
    )
    
    # 测试MLP策略
    mlp_policy = create_policy_network(params, "mlp")
    
    # 初始化参数
    key = random.PRNGKey(42)
    dummy_obs = jnp.ones((4, 10))  # 批量4，观测维度10
    
    mlp_params = mlp_policy.init(key, dummy_obs)
    mlp_output = mlp_policy.apply(mlp_params, dummy_obs)
    
    print(f"✅ MLP策略: 输入 {dummy_obs.shape} -> 输出 {mlp_output.shape}")
    assert mlp_output.shape == (4, 3), f"期望(4, 3)，得到{mlp_output.shape}"
    
    # 测试RNN策略
    rnn_policy = create_policy_network(params, "rnn")
    rnn_state = jnp.zeros((4, params.rnn_hidden_size))
    
    rnn_params = rnn_policy.init(key, dummy_obs, rnn_state)
    rnn_output, new_rnn_state = rnn_policy.apply(rnn_params, dummy_obs, rnn_state)
    
    print(f"✅ RNN策略: 输入 {dummy_obs.shape} -> 输出 {rnn_output.shape}")
    assert rnn_output.shape == (4, 3), f"期望(4, 3)，得到{rnn_output.shape}"
    assert new_rnn_state.shape == rnn_state.shape, "RNN状态形状不匹配"
    
    # 测试JIT编译
    jit_mlp = jax.jit(mlp_policy.apply)
    jit_output = jit_mlp(mlp_params, dummy_obs)
    
    print(f"✅ JIT编译: MLP策略编译成功")
    assert jnp.allclose(mlp_output, jit_output), "JIT输出不匹配"
    
    # 测试控制约束
    raw_control = jnp.array([[0.8, -0.6, 1.2], [-0.5, 0.9, -0.3]])
    prev_action = jnp.array([[0.1, -0.1, 0.2], [-0.2, 0.3, -0.1]])
    
    constrained_control = apply_control_constraints(raw_control, params, prev_action)
    print(f"✅ 控制约束: 应用成功")
    
    # 验证边界
    assert jnp.all(jnp.abs(constrained_control) <= params.max_thrust), "控制边界被违反"
    
    print("🎉 策略网络验证: 所有测试通过!")


if __name__ == "__main__":
    validate_policy_implementation()