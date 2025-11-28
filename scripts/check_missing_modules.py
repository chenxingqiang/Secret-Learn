#!/usr/bin/env python3
"""
检查遗漏模块中的具体算法
"""

import inspect

missing_modules = [
    'compose',
    'dummy',
    'feature_extraction',
    'frozen',
    'impute',
    'kernel_ridge',
    'model_selection',
    'pipeline',
    'random_projection',
]

def is_estimator(obj):
    """检查对象是否是 sklearn 估计器"""
    try:
        return (
            inspect.isclass(obj) and 
            hasattr(obj, 'fit') and
            not obj.__name__.startswith('_') and
            obj.__module__.startswith('sklearn.')
        )
    except:
        return False

print("="*80)
print("遗漏模块中的算法详情")
print("="*80)
print()

all_missing_algorithms = []

for module_name in missing_modules:
    try:
        module = __import__(f'sklearn.{module_name}', fromlist=[module_name])
        
        algorithms = []
        for name, obj in inspect.getmembers(module):
            if is_estimator(obj):
                algorithms.append(name)
        
        if algorithms:
            print(f"\n【{module_name}】 ({len(algorithms)} 个算法)")
            print("-" * 60)
            for i, algo in enumerate(sorted(algorithms), 1):
                print(f"  {i:2d}. {algo}")
                all_missing_algorithms.append((module_name, algo))
        else:
            print(f"\n【{module_name}】 (无估计器)")
            
    except Exception as e:
        print(f"\n【{module_name}】 错误: {str(e)}")

print("\n" + "="*80)
print("总结")
print("="*80)
print(f"遗漏的模块数: {len(missing_modules)}")
print(f"遗漏的算法数: {len(all_missing_algorithms)}")
print()

if all_missing_algorithms:
    print("遗漏的算法按模块分类:")
    current_module = None
    for module_name, algo in sorted(all_missing_algorithms):
        if module_name != current_module:
            print(f"\n{module_name}:")
            current_module = module_name
        print(f"  - {algo}")

print("\n" + "="*80)
print("建议更新 count_sklearn_algorithms.py")
print("="*80)
print("\n需要在 import 语句中添加:")
for module_name in sorted(missing_modules):
    try:
        module = __import__(f'sklearn.{module_name}', fromlist=[module_name])
        algorithms = [
            name for name, obj in inspect.getmembers(module)
            if is_estimator(obj)
        ]
        if algorithms:
            print(f"    {module_name},  # {len(algorithms)} 个算法")
    except:
        pass

print("\n完整的 import 语句应该是:")
print("\nfrom sklearn import (")
modules_list = [
    'cluster', 'decomposition', 'ensemble', 'linear_model',
    'naive_bayes', 'neighbors', 'neural_network', 'svm',
    'tree', 'discriminant_analysis', 'gaussian_process',
    'manifold', 'preprocessing', 'feature_selection',
    'covariance', 'cross_decomposition', 'mixture',
    'isotonic', 'kernel_approximation', 'multiclass',
    'multioutput', 'semi_supervised', 'calibration'
]

# 添加遗漏的模块
for module_name in sorted(missing_modules):
    try:
        module = __import__(f'sklearn.{module_name}', fromlist=[module_name])
        algorithms = [
            name for name, obj in inspect.getmembers(module)
            if is_estimator(obj)
        ]
        if algorithms and module_name not in modules_list:
            modules_list.append(module_name)
    except:
        pass

# 排序并打印
modules_list = sorted(modules_list)
for i, mod in enumerate(modules_list):
    if i < len(modules_list) - 1:
        print(f"    {mod},")
    else:
        print(f"    {mod}")
print(")")

print("\n" + "="*80)

