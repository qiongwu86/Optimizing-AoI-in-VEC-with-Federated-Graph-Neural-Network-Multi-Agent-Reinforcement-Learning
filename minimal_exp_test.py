#!/usr/bin/env python3
"""
最小化测试：只测试关键的exp函数
"""

import numpy as np

# 直接测试sigmoid函数
def sigmoid(x):
    exp_input = -x
    print(f"[DEBUG] sigmoid: exp_input = {exp_input}, max = {np.max(exp_input) if hasattr(exp_input, '__len__') else exp_input}")
    result = 1.0/(1+np.exp(exp_input))
    print(f"[DEBUG] sigmoid: result = {result}")
    return result

# 测试f函数
def f(x, a, powerFactor):
    if a == 0:
        return 0
    exp_term = powerFactor * x
    exp_term = np.clip(exp_term, -50, 50)  # 限制指数范围
    
    print(f"[DEBUG] f function: exp_term = {exp_term}, powerFactor = {powerFactor}")
    result = np.exp(exp_term) / powerFactor
    print(f"[DEBUG] f function: result = {result}")
    
    # 检查结果是否为NaN或无穷大
    if np.isnan(result) or np.isinf(result):
        return 0
    return result

if __name__ == "__main__":
    print("测试 sigmoid 函数:")
    sigmoid(0)
    sigmoid(5)
    sigmoid(-5)
    
    print("\n测试 f 函数:")
    f(1.0, 1.0, 1.0)
    f(10.0, 1.0, 1.0)
    f(100.0, 1.0, 1.0)
    
    print("\n测试 GCN 特征 exp:")
    # 模拟GCN特征
    node_feature = np.random.normal(0, 5, (3, 2))
    print(f"node_feature: {node_feature}")
    
    feature_mean = np.mean(np.abs(node_feature), axis=0)
    print(f"feature_mean: {feature_mean}")
    
    exp_result = np.exp(feature_mean)
    print(f"exp_result: {exp_result}")
    
    print("测试完成！")
