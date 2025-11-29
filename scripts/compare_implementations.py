#!/usr/bin/env python3
"""
 sklearn  Secret-Learn 
"""

import os
import glob

# sklearn （ count_sklearn_algorithms.py）
sklearn_stats = {
    'Linear Models': 37,
    'Preprocessing': 19,
    'Ensemble': 18,
    'Clustering': 14,
    'Decomposition': 14,
    'Neighbors': 11,
    'Feature Selection': 11,
    'Covariance': 8,
    'SVM': 7,
    'Naive Bayes': 6,
    'Manifold': 5,
    'Kernel Approximation': 5,
    'Calibration': 5,
    'Tree': 4,
    'Cross Decomposition': 4,
    'Multiclass': 4,
    'Multioutput': 4,
    'Model Selection': 4,
    'Neural Network': 3,
    'Discriminant Analysis': 3,
    'Semi Supervised': 3,
    'Impute': 3,
    'Pipeline': 3,
    'Random Projection': 3,
    'Gaussian Process': 2,
    'Mixture': 2,
    'Compose': 2,
    'Dummy': 2,
    'Feature Extraction': 2,
    'Isotonic': 1,
    'Kernel Ridge': 1,
    'Frozen': 1,
}

#  Secret-Learn 
def count_secretlearn_implementations(base_path):
    """ Secret-Learn """
    
    categories = {}
    
    # FL 
    fl_path = os.path.join(base_path, 'secretlearn/FL')
    if os.path.exists(fl_path):
        for category_dir in os.listdir(fl_path):
            full_path = os.path.join(fl_path, category_dir)
            if os.path.isdir(full_path) and not category_dir.startswith('__'):
                #  .py file（ __init__.py）
                py_files = [f for f in os.listdir(full_path) 
                           if f.endswith('.py') and f != '__init__.py']
                if py_files:
                    categories[category_dir] = len(py_files)
    
    return categories

# 
category_mapping = {
    'cluster': 'Clustering',
    'clustering': 'Clustering',
    'decomposition': 'Decomposition',
    'ensemble': 'Ensemble',
    'linear_model': 'Linear Models',
    'linear_models': 'Linear Models',
    'naive_bayes': 'Naive Bayes',
    'neighbors': 'Neighbors',
    'neural_network': 'Neural Network',
    'svm': 'SVM',
    'tree': 'Tree',
    'discriminant_analysis': 'Discriminant Analysis',
    'gaussian_process': 'Gaussian Process',
    'manifold': 'Manifold',
    'preprocessing': 'Preprocessing',
    'feature_selection': 'Feature Selection',
    'covariance': 'Covariance',
    'cross_decomposition': 'Cross Decomposition',
    'mixture': 'Mixture',
    'isotonic': 'Isotonic',
    'kernel_approximation': 'Kernel Approximation',
    'multiclass': 'Multiclass',
    'multioutput': 'Multioutput',
    'semi_supervised': 'Semi Supervised',
    'calibration': 'Calibration',
    'compose': 'Compose',
    'dummy': 'Dummy',
    'feature_extraction': 'Feature Extraction',
    'frozen': 'Frozen',
    'impute': 'Impute',
    'kernel_ridge': 'Kernel Ridge',
    'model_selection': 'Model Selection',
    'pipeline': 'Pipeline',
    'random_projection': 'Random Projection',
}

base_path = '/Users/xingqiangchen/jax-sklearn'
secretlearn_cats = count_secretlearn_implementations(base_path)

# convert
secretlearn_mapped = {}
for dir_name, count in secretlearn_cats.items():
    display_name = category_mapping.get(dir_name, dir_name.replace('_', ' ').title())
    secretlearn_mapped[display_name] = count

print("="*90)
print("sklearn vs Secret-Learn ")
print("="*90)
print()
print(f"{'':<30} {'sklearn':<10} {'Secret-Learn':<15} {'':<10} {''}")
print("-"*90)

total_sklearn = 0
total_implemented = 0
category_details = []

for category, sklearn_count in sklearn_stats.items():
    implemented = secretlearn_mapped.get(category, 0)
    total_sklearn += sklearn_count
    total_implemented += implemented
    
    if sklearn_count > 0:
        coverage = (implemented / sklearn_count) * 100
    else:
        coverage = 0
    
    if coverage >= 80:
        status = ""
    elif coverage >= 50:
        status = "🟡 "
    elif coverage > 0:
        status = "🟠 "
    else:
        status = " "
    
    category_details.append((category, sklearn_count, implemented, coverage, status))

# 
category_details.sort(key=lambda x: x[3], reverse=True)

for category, sklearn_count, implemented, coverage, status in category_details:
    print(f"{category:<30} {sklearn_count:<10} {implemented:<15} {coverage:>6.1f}%    {status}")

print("-"*90)
total_coverage = (total_implemented / total_sklearn) * 100 if total_sklearn > 0 else 0
print(f"{'':<30} {total_sklearn:<10} {total_implemented:<15} {total_coverage:>6.1f}%")
print()

print("="*90)
print("")
print("="*90)
print(f"sklearn :        {total_sklearn}")
print(f"Secret-Learn :       {total_implemented}")
print(f"pattern:          {total_implemented}")
print(f"pattern:          {total_implemented * 3}")
print(f":                  {total_coverage:.1f}%")
print()

print("="*90)
print("")
print("="*90)
print()
print("🔴 （）：")
missing_high = []
for cat, sk_count, impl, cov, status in category_details:
    if cov < 50 and sk_count >= 5:
        missing_high.append(f"  - {cat}: {impl}/{sk_count} ({cov:.0f}%) -  {sk_count - impl} ")

if missing_high:
    for item in missing_high[:5]:
        print(item)
else:
    print("  ！")

print()
print("🟡 （）：")
missing_med = []
for cat, sk_count, impl, cov, status in category_details:
    if 50 <= cov < 80 and sk_count >= 3:
        missing_med.append(f"  - {cat}: {impl}/{sk_count} ({cov:.0f}%) -  {sk_count - impl} ")

if missing_med:
    for item in missing_med[:5]:
        print(item)
else:
    print("  ！")

print()
print("="*90)
print("Secret-Learn ")
print("="*90)
for category, count in sorted(secretlearn_mapped.items(), key=lambda x: x[1], reverse=True):
    print(f"{category:<30} {count}  × 3 pattern = {count * 3} ")

print()
print("="*90)

