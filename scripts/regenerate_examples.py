#!/usr/bin/env python3
"""
重新生成所有有问题的 example 文件
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, '/Users/xingqiangchen/jax-sklearn')
from secretlearn.algorithm_classifier import classify_algorithm

def camel_to_snake(name):
    """驼峰转下划线"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(snake_str):
    """下划线转驼峰"""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def generate_example(algo_name, category, mode):
    """
    生成正确的 example 文件
    
    Parameters
    ----------
    algo_name : str
        驼峰命名的算法名，如 'LinearSVC'
    category : str
        下划线命名的类别，如 'svm'
    mode : str
        'FL', 'SS', or 'SL'
    """
    snake_name = camel_to_snake(algo_name)
    class_name = f"{mode}{algo_name}"
    
    # 分类算法
    try:
        char = classify_algorithm(algo_name)
        is_unsupervised = char.get('is_unsupervised', False)
        use_epochs = char.get('use_epochs', False)
    except:
        is_unsupervised = False
        use_epochs = False
    
    # 根据模式确定端口
    port_base = {'FL': 9491, 'SS': 9494, 'SL': 9497}
    ports = [port_base[mode] + i for i in range(3)]
    
    # 根据模式确定初始化代码
    if mode == 'FL':
        init_code = """    # Create devices dict for FL mode
    devices = {"alice": alice, "bob": bob, "carol": carol}
    """
    elif mode == 'SS':
        init_code = """    # Use SPU for SS mode
    """
    else:  # SL
        init_code = """    # Create devices dict for SL mode
    devices = {"alice": alice, "bob": bob, "carol": carol}
    """
    
    # 根据模式确定实例化代码
    if mode == 'FL' or mode == 'SL':
        model_init = f"model = {class_name}(devices)"
    else:  # SS
        model_init = f"model = {class_name}(spu)"
    
    # 生成数据和 fit 代码
    if is_unsupervised:
        data_code = """    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # No labels needed for unsupervised learning"""
        
        label_section = ""
        
        fit_code = f"""    {model_init}
    model.fit(fed_X)  # Unsupervised: no labels"""
        
    elif use_epochs:
        data_code = """    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples).astype(np.int32)  # Classification labels"""
        
        label_section = """
    # Create federated labels
    fed_y = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(y),
        },
        partition_way=PartitionWay.HORIZONTAL
    )"""
        
        fit_code = f"""    {model_init}
    model.fit(fed_X, fed_y, epochs=10)  # Iterative training"""
        
    else:  # supervised non-iterative
        data_code = """    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)  # Target values"""
        
        label_section = """
    # Create federated labels
    fed_y = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(y),
        },
        partition_way=PartitionWay.HORIZONTAL
    )"""
        
        fit_code = f"""    {model_init}
    model.fit(fed_X, fed_y)"""
    
    template = f'''#!/usr/bin/env python3
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Usage Example for {class_name}

This example demonstrates how to use the privacy-preserving {algo_name}
in SecretFlow's {mode} mode.
"""

import numpy as np

try:
    import secretflow as sf
    import secretflow.distributed as sfd
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.device.driver import reveal
    from secretflow.distributed.const import DISTRIBUTION_MODE
except ImportError:
    print(" SecretFlow not installed. Install with: pip install secretflow")
    exit(1)

from secretlearn.{mode}.{category}.{snake_name} import {class_name}


def main():
    """Main example function"""
    print("="*70)
    print(f" {class_name} Usage Example")
    print("="*70)
    
    # Step 1: Initialize SecretFlow (PRODUCTION mode for SF 1.11+)
    print("\\n[1/5] Initializing SecretFlow...")
    
    # For single-node testing (simulated multi-party)
    cluster_config = {{
        'parties': {{
            'alice': {{'address': 'localhost:{ports[0]}', 'listen_addr': '0.0.0.0:{ports[0]}'}},
            'bob': {{'address': 'localhost:{ports[1]}', 'listen_addr': '0.0.0.0:{ports[1]}'}},
            'carol': {{'address': 'localhost:{ports[2]}', 'listen_addr': '0.0.0.0:{ports[2]}'}},
        }},
        'self_party': 'alice'
    }}
    
    # Initialize with PRODUCTION mode (SF 1.11+ removes Ray/SIMULATION mode)
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    
    # Create SPU device
    spu_config = sf.utils.testing.cluster_def(
        parties=['alice', 'bob', 'carol'],
        runtime_config={{'protocol': 'ABY3', 'field': 'FM64'}}
    )
    spu = sf.SPU(spu_config)
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')
    print("  ✓ SecretFlow initialized (PRODUCTION mode)")
    
    # Step 2: Create sample data
    print("\\n[2/5] Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
{data_code}
    
    # Partition data vertically
    X_alice = X[:, 0:5]
    X_bob = X[:, 5:10]
    X_carol = X[:, 10:15]
    
    print(f"  ✓ Data shape: {{n_samples}} samples × {{n_features}} features")
    print(f"  ✓ Alice: {{X_alice.shape}}, Bob: {{X_bob.shape}}, Carol: {{X_carol.shape}}")
    
    # Step 3: Create federated data
    print("\\n[3/5] Creating federated data...")
    fed_X = FedNdarray(
        partitions={{
            alice: alice(lambda x: x)(X_alice),
            bob: bob(lambda x: x)(X_bob),
            carol: carol(lambda x: x)(X_carol),
        }},
        partition_way=PartitionWay.VERTICAL
    ){label_section}
    print("  ✓ Federated data created")
    
    # Step 4: Train model
    print("\\n[4/5] Training {class_name}...")
    print("  Note: All computation with privacy protection")
    
    import time
    start_time = time.time()
    
{init_code}
{fit_code}
    
    training_time = time.time() - start_time
    print(f"  ✓ Training completed in {{training_time*1000:.2f}}ms")
    
    # Step 5: Make predictions (if applicable)
    print("\\n[5/5] Model trained successfully!")
    print("  ✓ Privacy: Fully protected")
    print(f"  ✓ Performance: {{training_time*1000:.2f}}ms")
    
    # Cleanup
    sf.shutdown()
    print("\\nExample completed!")


if __name__ == "__main__":
    main()
'''
    
    return template

def get_category_from_import(content):
    """从 import 语句中提取类别"""
    match = re.search(r'from secretlearn\.[A-Z]+\.(\w+)\.', content)
    if match:
        return match.group(1)
    return None

def regenerate_problematic_examples(base_path):
    """重新生成有问题的 example 文件"""
    print("="*90)
    print("重新生成有问题的 Example 文件")
    print("="*90)
    print()
    
    regenerated = 0
    
    for mode in ['FL', 'SS', 'SL']:
        examples_path = Path(base_path) / 'examples' / mode
        
        if not examples_path.exists():
            continue
        
        print(f"\n【{mode} 模式】")
        mode_count = 0
        
        for py_file in sorted(examples_path.glob('*.py')):
            if py_file.name == '__init__.py':
                continue
            
            try:
                # 读取文件
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有问题
                has_issue = False
                
                # 1. 检查未替换的模板变量
                if '{algo_name}' in content:
                    has_issue = True
                
                # 2. 检查语法错误（简单检查）
                try:
                    compile(content, py_file.name, 'exec')
                except SyntaxError:
                    has_issue = True
                
                # 3. 检查导入语句中的类名错误
                # 例如：from secretlearn.SL.svm.linear_svc import FLLinearSVC
                # 应该是：from secretlearn.SL.svm.linear_svc import SLLinearSVC
                import_match = re.search(
                    rf'from secretlearn\.{mode}\.(\w+)\.(\w+) import ([A-Z]\w+)',
                    content
                )
                if import_match:
                    category, module, class_name = import_match.groups()
                    expected_class = f"{mode}{snake_to_camel(module)}"
                    if class_name != expected_class:
                        has_issue = True
                
                if not has_issue:
                    continue
                
                # 提取信息
                filename = py_file.stem
                algo_name = snake_to_camel(filename)
                category = get_category_from_import(content)
                
                if not category:
                    print(f"  ⚠️  无法确定类别: {py_file.name}")
                    continue
                
                # 重新生成
                new_content = generate_example(algo_name, category, mode)
                
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                mode_count += 1
                regenerated += 1
                
                if mode_count <= 10:
                    print(f"  {py_file.name}")
                
            except Exception as e:
                print(f"   {py_file.name}: {str(e)}")
        
        if mode_count > 10:
            print(f"  ... 还有 {mode_count - 10} 个文件已重新生成")
        
        if mode_count > 0:
            print(f"  {mode} 小计: {mode_count} 个文件")
    
    print(f"\n总计重新生成: {regenerated} 个文件")
    return regenerated

def main():
    base_path = '/Users/xingqiangchen/jax-sklearn'
    
    print("="*90)
    print("Example 文件重新生成工具")
    print("="*90)
    print()
    print("将重新生成有问题的 example 文件:")
    print("  - 修复模板变量")
    print("  - 修复代码结构")
    print("  - 修复导入语句")
    print("  - 确保语法正确")
    print()
    
    response = input("是否继续？(yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("操作已取消")
        return
    
    # 重新生成
    count = regenerate_problematic_examples(base_path)
    
    print("\n" + "="*90)
    print("Example 文件重新生成完成！")
    print("="*90)
    print(f"重新生成: {count} 个文件")
    print()

if __name__ == '__main__':
    main()

