# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(2024)

# Define number of students
num_students = 10000

# Generate raw scores for Chemistry and Geography
chemistry_raw = np.random.normal(loc=65, scale=12, size=num_students)
geography_raw = np.random.normal(loc=60, scale=15, size=num_students)

# Create short variable names for compatibility with later code
chem_raw = chemistry_raw
geo_raw = geography_raw

# Print first 5 values to verify
print("Chemistry Raw (first 5):", chemistry_raw[:5])
print("Geography Raw (first 5):", geography_raw[:5])

def scale_scores(raw_scores):
    """
    Convert raw scores to scaled scores with guaranteed rank preservation.
    Simple, robust method: sort everything, then assign scores based on exact rank.
    """
    # 1. 获取原始分的稳定排序索引（从低到高）
    # 使用稳定排序确保相同分数处理一致
    sorted_indices = np.argsort(raw_scores, kind='stable')  # 关键修改：稳定排序
    n = len(raw_scores)
    
    # 2. 定义等级边界（基于排名位置）
    # 各等级的累计人数
    grade_counts = {
        'E': int(n * 0.02),      # 2%
        'D': int(n * 0.15),      # 13% + 2%
        'C': int(n * 0.50),      # 35% + 15%
        'B': int(n * 0.85),      # 35% + 50%
        'A': n                   # 15% + 85%
    }
    
    # 各等级分数区间
    grade_ranges = {
        'E': (30, 40),
        'D': (41, 55),
        'C': (56, 70),
        'B': (71, 85),
        'A': (86, 100)
    }
    
    # 3. 初始化结果数组
    scaled_scores = np.zeros(n, dtype=int)
    
    # 4. 按等级填充分数
    start_idx = 0
    for grade in ['E', 'D', 'C', 'B', 'A']:
        end_idx = grade_counts[grade]
        if start_idx >= end_idx:
            continue
            
        grade_size = end_idx - start_idx
        low, high = grade_ranges[grade]
        
        if grade_size > 0:
            # 生成该等级内的分数（从低到高）
            # 使用arange确保每个分数都出现
            if grade_size == 1:
                grade_scores = [low]
            else:
                # 线性间隔，确保覆盖整个区间
                grade_scores = np.linspace(low, high, grade_size)
            
            # 四舍五入并转换为整数
            grade_scores_int = np.round(grade_scores).astype(int)
            
            # 获取该等级考生的索引（已排序）
            grade_student_indices = sorted_indices[start_idx:end_idx]
            
            # 分配分数：排名最低的得最低分，排名最高的得最高分
            scaled_scores[grade_student_indices] = grade_scores_int
        
        start_idx = end_idx
    
    return scaled_scores
    """
    Convert raw scores to scaled scores with guaranteed rank preservation.
    This version sorts scores first, then assigns scores based on exact rank positions.
    """
    # 1. 获取原始分的排序索引（从小到大）
    sorted_indices = np.argsort(raw_scores)
    sorted_scores = raw_scores[sorted_indices]
    n = len(raw_scores)
    
    # 2. 定义等级划分（基于排序位置，更精确）
    # 各等级的累计比例（与之前一致）
    cumulative_props = {'E': 0.02, 'D': 0.15, 'C': 0.50, 'B': 0.85, 'A': 1.00}
    # 各等级赋分区间（与之前一致）
    grade_points = {'E': (30, 40), 'D': (41, 55), 'C': (56, 70), 'B': (71, 85), 'A': (86, 100)}
    
    # 3. 计算每个等级的分界索引（考生序号）
    grade_boundaries = {}
    for grade, prop in cumulative_props.items():
        # 向下取整，确保边界明确
        boundary_idx = int(np.floor(prop * n)) - 1
        grade_boundaries[grade] = boundary_idx
    
    # 4. 初始化结果数组
    scaled_scores = np.zeros_like(raw_scores, dtype=int)
    
    # 5. 按等级逐段赋分
    prev_idx = -1
    for grade in ['E', 'D', 'C', 'B', 'A']:
        curr_idx = grade_boundaries[grade]
        if curr_idx <= prev_idx:
            continue
            
        # 该等级内的考生索引范围
        grade_indices = sorted_indices[prev_idx+1:curr_idx+1]
        grade_size = len(grade_indices)
        
        if grade_size > 0:
            low, high = grade_points[grade]
            # 在该等级内线性赋分（确保一分一档）
            # 生成该等级内从low到high的等间隔分数
            grade_scores = np.linspace(low, high, grade_size)
            # 四舍五入到整数
            grade_scores_rounded = np.round(grade_scores).astype(int)
            # 按原始顺序分配分数
            scaled_scores[grade_indices] = grade_scores_rounded
        
        prev_idx = curr_idx
    
    return scaled_scores


    """
    转换为标准分（Z分数），然后线性映射。
    """
    z_scores = (raw_scores - np.mean(raw_scores)) / np.std(raw_scores)
    scaled = target_mean + z_scores * target_std
    scaled = np.round(scaled).astype(int)
    scaled = np.clip(scaled, 0, 150)  # 限定一个合理范围，避免极端值
    return scaled



# Apply scaling transformation
chemistry_scaled = scale_scores(chemistry_raw)
geography_scaled = scale_scores(geography_raw)

# Create short variable names for scaled scores
chem_scaled = chemistry_scaled
geo_scaled = geography_scaled

# Print first 5 scaled scores for verification
print("Chemistry Scaled (first 5):", chem_scaled[:5])
print("Geography Scaled (first 5):", geo_scaled[:5])

def calculate_stats(name, raw, scaled):
    """Calculate and print descriptive statistics"""
    stats_df = pd.DataFrame({
        'Raw_Score': raw,
        'Scaled_Score': scaled
    })
    print(f"\n=== {name} ===")
    print(stats_df.describe().round(2))
    print(f"Raw Std: {raw.std():.2f}, Scaled Std: {scaled.std():.2f}")

# Calculate statistics for both subjects
calculate_stats("Chemistry", chem_raw, chem_scaled)
calculate_stats("Geography", geo_raw, geo_scaled)

# ==================== PART 5: VISUALIZATION ====================
# Figure 1: Distribution comparison (2x2 grid)
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle('Distribution Comparison: Before vs After Scaling', fontsize=16)

# Top-left: Chemistry raw scores
axes1[0, 0].hist(chem_raw, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes1[0, 0].set_title('Chemistry - Raw Scores')
axes1[0, 0].set_xlabel('Score')
axes1[0, 0].set_ylabel('Frequency')

# Top-right: Chemistry scaled scores
axes1[0, 1].hist(chem_scaled, bins=30, color='lightcoral', edgecolor='black', alpha=0.7, range=(30, 100))
axes1[0, 1].set_title('Chemistry - Scaled Scores')
axes1[0, 1].set_xlabel('Score')
axes1[0, 1].set_ylabel('Frequency')

# Bottom-left: Geography raw scores
axes1[1, 0].hist(geo_raw, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes1[1, 0].set_title('Geography - Raw Scores')
axes1[1, 0].set_xlabel('Score')
axes1[1, 0].set_ylabel('Frequency')

# Bottom-right: Geography scaled scores
axes1[1, 1].hist(geo_scaled, bins=30, color='gold', edgecolor='black', alpha=0.7, range=(30, 100))
axes1[1, 1].set_title('Geography - Scaled Scores')
axes1[1, 1].set_xlabel('Score')
axes1[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('score_distribution_comparison.png', dpi=300)
plt.show()

# Figure 2: Score transformation scatter plot
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(chem_raw, chem_scaled, alpha=0.5, s=10, label='Chemistry', color='blue')
ax2.scatter(geo_raw, geo_scaled, alpha=0.5, s=10, label='Geography', color='red')
ax2.set_xlabel('Raw Score')
ax2.set_ylabel('Scaled Score')
ax2.set_title('Score Transformation: Raw to Scaled')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('score_transformation_scatter.png', dpi=300)
plt.show()

print("\nAll charts have been generated and saved as PNG files.")


# ==================== VALIDATION MODULE ====================
print("\n" + "="*60)
print("MODEL VALIDATION REPORT")
print("="*60)

# -----------------------------------------------------------------
# 1. 基础参数定义 (必须与scale_scores函数内部一致)
# -----------------------------------------------------------------
grade_points = {'A': (86, 100), 'B': (71, 85), 'C': (56, 70), 'D': (41, 55), 'E': (30, 40)}
grade_cutoffs = {'A': 0.85, 'B': 0.50, 'C': 0.15, 'D': 0.02, 'E': 0.00}

# -----------------------------------------------------------------
# 2. 验证函数定义
# -----------------------------------------------------------------
def validate_score_ranges(subject_name, scaled_scores, grade_points):
    """验证所有赋分结果是否都在定义的等级区间内"""
    all_scores = set()
    for low, high in grade_points.values():
        all_scores.update(range(low, high + 1))
    
    unique_scores = set(scaled_scores)
    invalid_scores = unique_scores - all_scores
    
    if not invalid_scores:
        print(f"[OK] {subject_name}: All scores are within defined grade ranges.")
        return True
    else:
        print(f"[ERROR] {subject_name}: Found invalid scores: {sorted(invalid_scores)[:10]}")
        return False

def validate_rank_preservation(subject_name, raw_scores, scaled_scores):
    """验证排名保持性 - 核心检查"""
    # 方法1: 计算排名差异总和
    raw_ranks = pd.Series(raw_scores).rank(method='min')
    scaled_ranks = pd.Series(scaled_scores).rank(method='min')
    rank_diff = (raw_ranks - scaled_ranks).abs().sum()
    
    # 方法2: 直接检查排序后的赋分是否单调不减
    sorted_indices = np.argsort(raw_scores)
    sorted_scaled = scaled_scores[sorted_indices]
    is_monotonic = np.all(np.diff(sorted_scaled) >= 0)
    
    print(f"\n{subject_name} Rank Preservation Check:")
    print(f"  Rank difference sum: {rank_diff:.6f}")
    print(f"  Sorted scaled scores monotonic: {is_monotonic}")
    
    if rank_diff < 1e-9 and is_monotonic:
        print(f"  [PASS] Perfect rank preservation")
        return True
    else:
        print(f"  [FAIL] Rank issues detected")
        # 诊断：检查具体哪些位置有问题
        if not is_monotonic:
            problematic_indices = np.where(np.diff(sorted_scaled) < 0)[0]
            print(f"  Found {len(problematic_indices)} non-monotonic points")
            if len(problematic_indices) > 0:
                idx = problematic_indices[0]
                print(f"  Example at index {idx}: {sorted_scaled[idx]} -> {sorted_scaled[idx+1]}")
        return False

def analyze_grade_distribution(subject_name, scaled_scores, grade_points, grade_cutoffs):
    """分析各等级分布情况"""
    print(f"\n{subject_name} Grade Distribution:")
    
    # 统计各等级人数
    total = len(scaled_scores)
    for grade in ['A', 'B', 'C', 'D', 'E']:
        low, high = grade_points[grade]
        count = ((scaled_scores >= low) & (scaled_scores <= high)).sum()
        actual_pct = count / total * 100
        
        # 计算期望比例
        if grade == 'E':
            expected_pct = 2.0
        elif grade == 'D':
            expected_pct = 13.0
        elif grade == 'C':
            expected_pct = 35.0
        elif grade == 'B':
            expected_pct = 35.0
        else:  # A
            expected_pct = 15.0
            
        diff = actual_pct - expected_pct
        status = "OK" if abs(diff) < 1.0 else "WARN"
        print(f"  Grade {grade} ({low:3d}-{high:3d}): {count:5d} ({actual_pct:5.1f}%) "
              f"[Exp: {expected_pct:4.1f}%, Diff: {diff:6.2f}%] [{status}]")

def analyze_score_continuity(subject_name, scaled_scores, grade_points):
    """分析分数连续性"""
    unique_scores = sorted(set(scaled_scores))
    all_possible = set()
    
    for low, high in grade_points.values():
        all_possible.update(range(low, high + 1))
    
    missing = sorted(set(all_possible) - set(unique_scores))
    
    print(f"\n{subject_name} Score Continuity:")
    print(f"  Unique scores: {len(unique_scores)} out of {len(all_possible)} possible")
    print(f"  Score range: [{min(unique_scores)}, {max(unique_scores)}]")
    
    if missing:
        print(f"  Missing scores ({len(missing)}): {missing[:15]}" + 
              ("..." if len(missing) > 15 else ""))
        # 检查缺失分数是否在等级边界
        boundary_gaps = []
        for grade in ['E', 'D', 'C', 'B']:
            low1, high1 = grade_points[grade]
            low2, high2 = grade_points[chr(ord(grade)-1)] if grade != 'E' else (0, 0)
            gap = high1 + 1
            if gap in missing:
                boundary_gaps.append(f"{high1}-{high1+1}")
        
        if boundary_gaps:
            print(f"  Note: Missing scores include grade boundaries: {boundary_gaps}")
    else:
        print(f"  [OK] All possible scores are represented")

# -----------------------------------------------------------------
# 3. 执行验证
# -----------------------------------------------------------------
print("\n" + "-"*60)
print("1. SCORE RANGE VALIDATION")
print("-"*60)
validate_score_ranges("Chemistry", chem_scaled, grade_points)
validate_score_ranges("Geography", geo_scaled, grade_points)

print("\n" + "-"*60)
print("2. RANK PRESERVATION ANALYSIS")
print("-"*60)
chem_rank_ok = validate_rank_preservation("Chemistry", chem_raw, chem_scaled)
geo_rank_ok = validate_rank_preservation("Geography", geo_raw, geo_scaled)

print("\n" + "-"*60)
print("3. DISTRIBUTION ANALYSIS")
print("-"*60)
analyze_grade_distribution("Chemistry", chem_scaled, grade_points, grade_cutoffs)
analyze_grade_distribution("Geography", geo_scaled, grade_points, grade_cutoffs)

print("\n" + "-"*60)
print("4. SCORE CONTINUITY ANALYSIS")
print("-"*60)
analyze_score_continuity("Chemistry", chem_scaled, grade_points)
analyze_score_continuity("Geography", geo_scaled, grade_points)

# -----------------------------------------------------------------
# 4. 总体结论
# -----------------------------------------------------------------
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)

# 检查关键指标
key_metrics = []
if chem_rank_ok and geo_rank_ok:
    key_metrics.append("[PASS] Rank preservation: PERFECT")
else:
    key_metrics.append("[FAIL] Rank preservation: FAILED")

# 检查分数范围
if ((chem_scaled.min() >= 30) and (chem_scaled.max() <= 100) and
    (geo_scaled.min() >= 30) and (geo_scaled.max() <= 100)):
    key_metrics.append("[OK] Score ranges: Valid (30-100)")
else:
    key_metrics.append("[ERROR] Score ranges: Invalid")

# 检查分布比例
chem_a_count = ((chem_scaled >= 86) & (chem_scaled <= 100)).sum()
geo_a_count = ((geo_scaled >= 86) & (geo_scaled <= 100)).sum()
if abs(chem_a_count/10000 - 0.15) < 0.01 and abs(geo_a_count/10000 - 0.15) < 0.01:
    key_metrics.append("[OK] Grade proportions: Approximately correct")
else:
    key_metrics.append("[WARN] Grade proportions: Significant deviation")

for metric in key_metrics:
    print(f"  {metric}")

print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("-"*60)
print("1. Rank preservation is the MOST IMPORTANT criterion.")
print("2. Missing scores within grade ranges are normal and depend on")
print("   the distribution of raw scores.")
print("3. Visual 'gaps' in histograms between different grades (e.g.,")
print("   between 70 and 71) are expected and correct - they represent")
print("   the boundaries between different grade tiers.")
print("="*60)
print("\n" + "="*60)
print("RANK CORRELATION ANALYSIS (More Robust Measure)")
print("="*60)

from scipy.stats import spearmanr, kendalltau

# 计算化学的排名相关系数
chem_spearman, chem_spearman_p = spearmanr(chem_raw, chem_scaled)
chem_kendall, chem_kendall_p = kendalltau(chem_raw, chem_scaled)

# 计算地理的排名相关系数
geo_spearman, geo_spearman_p = spearmanr(geo_raw, geo_scaled)
geo_kendall, geo_kendall_p = kendalltau(geo_raw, geo_scaled)

print(f"\nChemistry:")
print(f"  Spearman rank correlation: {chem_spearman:.6f} (p={chem_spearman_p:.2e})")
print(f"  Kendall's tau correlation: {chem_kendall:.6f} (p={chem_kendall_p:.2e})")

print(f"\nGeography:")
print(f"  Spearman rank correlation: {geo_spearman:.6f} (p={geo_spearman_p:.2e})")
print(f"  Kendall's tau correlation: {geo_kendall:.6f} (p={geo_kendall_p:.2e})")

# 解读
print(f"\nINTERPRETATION:")
print(f"- Correlation of 1.0 means perfect rank preservation.")
print(f"- Values above 0.999 indicate nearly perfect preservation.")
print(f"- The tiny p-values (e.g., {chem_spearman_p:.2e}) confirm the correlation")
print(f"  is statistically significant (not due to chance).")
# 额外的诊断：检查原始分和赋分的关系
print("\n" + "="*60)
print("ADDITIONAL DIAGNOSTICS")
print("="*60)
# 检查是否有相同的原始分得到不同的赋分
chem_df = pd.DataFrame({'raw': chem_raw, 'scaled': chem_scaled})
chem_grouped = chem_df.groupby('raw')['scaled'].nunique()
problematic_raw_scores = chem_grouped[chem_grouped > 1]
if len(problematic_raw_scores) > 0:
    print(f"Chemistry: Found {len(problematic_raw_scores)} raw scores with multiple scaled values")
    for raw_score in problematic_raw_scores.index[:3]:
        values = chem_df[chem_df['raw'] == raw_score]['scaled'].unique()
        print(f"  Raw score {raw_score:.2f} -> Scaled: {values}")
else:
    print("Chemistry: Each raw score maps to a unique scaled score (GOOD)")

# 同样检查地理
geo_df = pd.DataFrame({'raw': geo_raw, 'scaled': geo_scaled})
geo_grouped = geo_df.groupby('raw')['scaled'].nunique()
problematic_raw_scores_geo = geo_grouped[geo_grouped > 1]
if len(problematic_raw_scores_geo) > 0:
    print(f"\nGeography: Found {len(problematic_raw_scores_geo)} raw scores with multiple scaled values")
    for raw_score in problematic_raw_scores_geo.index[:3]:
        values = geo_df[geo_df['raw'] == raw_score]['scaled'].unique()
        print(f"  Raw score {raw_score:.2f} -> Scaled: {values}")
else:
    print("\nGeography: Each raw score maps to a unique scaled score (GOOD)")