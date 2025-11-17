#!/usr/bin/env python3
"""测试speed_variation_percent参数是否正确添加"""

import sys
import importlib

# 清除可能缓存的模块
if 'EnvironmentGCN_SAC_new_global2_test_env_speed' in sys.modules:
    del sys.modules['EnvironmentGCN_SAC_new_global2_test_env_speed']

# 重新导入
from EnvironmentGCN_SAC_new_global2_test_env_speed import VEC_env
import inspect

# 检查参数签名
sig = inspect.signature(VEC_env.__init__)
params = list(sig.parameters.keys())

print("VEC_env.__init__ 参数列表:")
for i, param in enumerate(params, 1):
    param_obj = sig.parameters[param]
    default = param_obj.default if param_obj.default != inspect.Parameter.empty else "无默认值"
    print(f"  {i}. {param} = {default}")

# 检查speed_variation_percent是否存在
if 'speed_variation_percent' in params:
    print("\n✅ speed_variation_percent 参数存在!")
    print(f"   默认值: {sig.parameters['speed_variation_percent'].default}")
else:
    print("\n❌ speed_variation_percent 参数不存在!")
    print("   可用参数:", params)



