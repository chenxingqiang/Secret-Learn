#!/usr/bin/env python3
"""
统计 sklearn 中每个模块的算法数量
"""

import inspect
import sklearn
from sklearn import (
    calibration, cluster, compose, covariance, cross_decomposition,
    decomposition, discriminant_analysis, dummy, ensemble,
    feature_extraction, feature_selection, frozen, gaussian_process,
    impute, isotonic, kernel_approximation, kernel_ridge,
    linear_model, manifold, mixture, model_selection,
    multiclass, multioutput, naive_bayes, neighbors,
    neural_network, pipeline, preprocessing, random_projection,
    semi_supervised, svm, tree
)

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

def count_algorithms_in_module(module, module_name):
    """统计模块中的算法数量"""
    algorithms = []
    
    for name, obj in inspect.getmembers(module):
        if is_estimator(obj):
            algorithms.append(name)
    
    return sorted(algorithms)

# 定义要统计的模块
modules = {
    'Calibration': calibration,
    'Clustering': cluster,
    'Compose': compose,
    'Covariance': covariance,
    'Cross Decomposition': cross_decomposition,
    'Decomposition': decomposition,
    'Discriminant Analysis': discriminant_analysis,
    'Dummy': dummy,
    'Ensemble': ensemble,
    'Feature Extraction': feature_extraction,
    'Feature Selection': feature_selection,
    'Frozen': frozen,
    'Gaussian Process': gaussian_process,
    'Impute': impute,
    'Isotonic': isotonic,
    'Kernel Approximation': kernel_approximation,
    'Kernel Ridge': kernel_ridge,
    'Linear Models': linear_model,
    'Manifold': manifold,
    'Mixture': mixture,
    'Model Selection': model_selection,
    'Multiclass': multiclass,
    'Multioutput': multioutput,
    'Naive Bayes': naive_bayes,
    'Neighbors': neighbors,
    'Neural Network': neural_network,
    'Pipeline': pipeline,
    'Preprocessing': preprocessing,
    'Random Projection': random_projection,
    'Semi Supervised': semi_supervised,
    'SVM': svm,
    'Tree': tree,
}

print("="*80)
print("sklearn 算法统计")
print("="*80)
print()

total_count = 0
category_stats = []

for category, module in modules.items():
    algorithms = count_algorithms_in_module(module, category)
    count = len(algorithms)
    total_count += count
    
    if count > 0:
        category_stats.append((category, count, algorithms))

# 按数量排序
category_stats.sort(key=lambda x: x[1], reverse=True)

# 打印统计结果
print(f"{'类别':<25} {'数量':>6}  示例算法")
print("-"*80)

for category, count, algorithms in category_stats:
    # 显示前3个算法作为示例
    examples = ', '.join(algorithms[:3])
    if len(algorithms) > 3:
        examples += f', ... (+{len(algorithms)-3} more)'
    
    print(f"{category:<25} {count:>6}  {examples}")

print("-"*80)
print(f"{'总计':<25} {total_count:>6}")
print()

# 详细列表
print("\n" + "="*80)
print("详细算法列表")
print("="*80)

for category, count, algorithms in category_stats:
    print(f"\n【{category}】 ({count} 个算法)")
    print("-" * 60)
    for i, algo in enumerate(algorithms, 1):
        print(f"  {i:2d}. {algo}")

print("\n" + "="*80)
print(f"sklearn 版本: {sklearn.__version__}")
print(f"总算法数: {total_count}")
print("="*80)

