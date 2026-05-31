#!/usr/bin/env python3
"""
关键词筛选工具：从源数据中筛选包含特定关键词的样本
用法: python filter.py --keywords_file output.txt --source_file medical/finetune/train_zh_0.json
"""
import json
import argparse
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import time
from tqdm import tqdm
from collections import defaultdict


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    """逐行读取JSONL文件"""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: JSON解析错误: {e}")
                continue
    return data


def read_keywords(keywords_file: Path) -> List[str]:
    """读取关键词列表"""
    with keywords_file.open("r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]
    print(f"已读取 {len(keywords)} 个关键词")
    return keywords


def sample_contains_keywords(sample: Dict[str, Any], keywords: List[str]) -> List[str]:
    """检查样本是否包含关键词，返回包含的关键词列表"""
    # 合并instruction、input和output字段
    text = ""
    for field in ['instruction', 'input', 'output']:
        if field in sample and sample[field]:
            text += str(sample[field]) + " "
    
    # 检查文本中是否包含关键词
    matched_keywords = []
    for keyword in keywords:
        if keyword in text:
            matched_keywords.append(keyword)
    
    return matched_keywords


def sample_by_keywords_stratified(
    source_file: Path, 
    keywords: List[str],
    samples_per_keyword: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    分层抽样：为每个关键词抽取指定数量的样本
    
    返回: 字典 {关键词: [样本列表]}
    """
    print(f"开始从 {source_file} 中抽取数据...")
    print(f"每个关键词抽取 {samples_per_keyword} 个样本")
    
    # 初始化数据结构
    keyword_samples = defaultdict(list)  # 每个关键词的样本
    all_matched_samples = []  # 所有匹配到的样本
    keyword_counters = defaultdict(int)  # 每个关键词已抽取的样本数
    
    # 先读取所有数据
    print("正在读取源数据...")
    all_samples = iter_jsonl(source_file)
    print(f"源数据共有 {len(all_samples)} 个样本")
    
    # 第一次遍历：找出包含关键词的样本
    print("正在查找包含关键词的样本...")
    for sample in tqdm(all_samples, desc="扫描样本"):
        matched_keywords = sample_contains_keywords(sample, keywords)
        if matched_keywords:
            # 记录样本和匹配的关键词
            sample_data = {
                'sample': sample,
                'matched_keywords': matched_keywords
            }
            all_matched_samples.append(sample_data)
    
    print(f"找到 {len(all_matched_samples)} 个包含关键词的样本")
    
    # 第二次遍历：为每个关键词抽取样本
    print("正在为每个关键词抽取样本...")
    
    # 打乱所有匹配的样本，避免顺序偏差
    random.shuffle(all_matched_samples)
    
    # 统计每个关键词的可用样本
    keyword_available_samples = defaultdict(list)
    for sample_data in all_matched_samples:
        sample = sample_data['sample']
        matched_keywords = sample_data['matched_keywords']
        for keyword in matched_keywords:
            keyword_available_samples[keyword].append(sample)
    
    # 为每个关键词抽取样本
    for keyword in tqdm(keywords, desc="处理关键词"):
        available = keyword_available_samples.get(keyword, [])
        
        if not available:
            print(f"警告: 关键词 '{keyword}' 在数据中没有找到样本")
            continue
        
        # 计算需要抽取的数量
        n_needed = min(samples_per_keyword, len(available))
        
        # 抽取样本
        selected = random.sample(available, n_needed) if n_needed <= len(available) else available
        
        # 记录抽取的样本
        keyword_samples[keyword] = selected
        keyword_counters[keyword] = len(selected)
    
    return dict(keyword_samples), dict(keyword_counters)


def sample_by_keywords_single_pass(
    source_file: Path, 
    keywords: List[str],
    samples_per_keyword: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    单遍抽样：在读取数据的同时为每个关键词收集样本
    更节省内存，适合大数据集
    """
    print(f"开始从 {source_file} 中抽取数据（单遍扫描）...")
    print(f"每个关键词抽取 {samples_per_keyword} 个样本")
    
    keyword_samples = defaultdict(list)
    
    # 将关键词转换为集合提高查找速度
    keyword_set = set(keywords)
    
    with source_file.open("r", encoding="utf-8") as f:
        # 首先获取文件总行数（用于进度条）
        print("正在计算文件大小...")
        total_lines = sum(1 for _ in f)
        f.seek(0)  # 重置文件指针
        
        print(f"文件共有 {total_lines} 行")
        
        with tqdm(total=total_lines, desc="扫描文件") as pbar:
            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    pbar.update(1)
                    continue
                
                # 检查样本是否包含关键词
                text = ""
                for field in ['instruction', 'input', 'output']:
                    if field in sample and sample[field]:
                        text += str(sample[field]) + " "
                
                # 为每个关键词检查是否还需要更多样本
                for keyword in keyword_set:
                    if keyword in text:
                        if len(keyword_samples[keyword]) < samples_per_keyword:
                            keyword_samples[keyword].append(sample)
                
                pbar.update(1)
                
                # 如果所有关键词都已收集足够样本，可以提前退出
                all_complete = all(
                    len(keyword_samples[k]) >= samples_per_keyword 
                    for k in keyword_set
                )
                if all_complete:
                    print("所有关键词已收集足够样本，提前结束扫描")
                    break
    
    return dict(keyword_samples)


def write_output_files(
    keyword_samples: Dict[str, List[Dict[str, Any]]],
    output_dir: Path
) -> Dict[str, Any]:
    """将结果写入文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "total_keywords": len(keyword_samples),
        "samples_per_keyword": {},
        "total_samples": 0,
        "output_files": []
    }
    
    # 合并所有样本（去重）
    all_unique_samples = []
    seen_samples = set()
    
    for keyword, samples in keyword_samples.items():
        for sample in samples:
            # 使用样本内容的哈希值来去重
            sample_str = json.dumps(sample, sort_keys=True, ensure_ascii=False)
            sample_hash = hash(sample_str)
            
            if sample_hash not in seen_samples:
                seen_samples.add(sample_hash)
                all_unique_samples.append(sample)
        
        manifest["samples_per_keyword"][keyword] = len(samples)
    
    manifest["total_samples"] = len(all_unique_samples)
    
    # 写入合并的文件
    merged_file = output_dir / "filtered_sft_data.jsonl"
    with merged_file.open("w", encoding="utf-8") as f:
        for sample in all_unique_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    manifest["output_files"].append(str(merged_file))
    
    # 写入每个关键词的单独文件（可选）
    for keyword, samples in keyword_samples.items():
        if samples:  # 只写入有样本的关键词
            keyword_file = output_dir / f"keyword_{keyword[:20]}.jsonl"  # 限制文件名长度
            with keyword_file.open("w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            manifest["output_files"].append(str(keyword_file))
    
    # 写入manifest
    manifest_file = output_dir / "filter_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n输出目录: {output_dir}")
    print(f"合并文件: {merged_file} ({len(all_unique_samples)} 个样本)")
    print(f"清单文件: {manifest_file}")
    
    return manifest


def estimate_time(source_file: Path, keywords: List[str]) -> float:
    """
    粗略估计运行时间
    基于：文件大小、关键词数量、硬盘读取速度
    """
    # 获取文件大小（MB）
    file_size_mb = source_file.stat().st_size / (1024 * 1024)
    
    # 估计参数
    read_speed_mb_per_sec = 100  # MB/秒，SSD的典型速度
    processing_speed_samples_per_sec = 1000  # 每秒处理样本数
    
    # 获取总样本数
    with source_file.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    # 计算时间
    read_time = file_size_mb / read_speed_mb_per_sec
    processing_time = total_lines / processing_speed_samples_per_sec
    
    # 加上固定开销
    total_time_seconds = read_time + processing_time + 5  # 5秒固定开销
    
    return total_time_seconds


def main():
    parser = argparse.ArgumentParser(description="从源数据中筛选包含关键词的样本")
    parser.add_argument("--keywords_file", type=Path, required=True, 
                       help="关键词文件，每行一个关键词")
    parser.add_argument("--source_file", type=Path, default=None,
                       help="源数据文件。默认: medical/finetune/train_zh_0.json")
    parser.add_argument("--output_dir", type=Path, default=project_root() / "filtered_data",
                       help="输出目录")
    parser.add_argument("--samples_per_keyword", type=int, default=10,
                       help="每个关键词抽取的样本数")
    parser.add_argument("--method", choices=["single_pass", "stratified"], default="single_pass",
                       help="抽样方法: single_pass(单遍扫描，省内存) 或 stratified(分层抽样，更准确)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 设置默认源文件
    if args.source_file is None:
        args.source_file = project_root() / "medical" / "finetune" / "train_zh_0.json"
    
    # 检查文件是否存在
    if not args.keywords_file.exists():
        print(f"错误: 关键词文件不存在: {args.keywords_file}")
        sys.exit(1)
    
    if not args.source_file.exists():
        print(f"错误: 源数据文件不存在: {args.source_file}")
        sys.exit(1)
    
    # 估算时间
    print("正在估算运行时间...")
    keywords = read_keywords(args.keywords_file)
    estimated_time = estimate_time(args.source_file, keywords)
    print(f"预计运行时间: {estimated_time:.1f} 秒 ({estimated_time/60:.1f} 分钟)")
    print()
    
    # 实际运行
    start_time = time.time()
    
    # 读取关键词
    keywords = read_keywords(args.keywords_file)
    
    # 根据选择的方法进行抽样
    if args.method == "single_pass":
        keyword_samples = sample_by_keywords_single_pass(
            args.source_file, keywords, args.samples_per_keyword
        )
    else:  # stratified
        keyword_samples, keyword_counters = sample_by_keywords_stratified(
            args.source_file, keywords, args.samples_per_keyword
        )
    
    # 写入输出文件
    manifest = write_output_files(keyword_samples, args.output_dir)
    
    # 统计结果
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印摘要
    print("\n" + "="*60)
    print("筛选完成！")
    print("="*60)
    print(f"总关键词数: {len(keyword_samples)}")
    
    # 统计每个关键词的实际样本数
    coverage_stats = {}
    for keyword, samples in keyword_samples.items():
        coverage_stats[keyword] = len(samples)
    
    # 找出覆盖不足的关键词
    poorly_covered = [(k, v) for k, v in coverage_stats.items() 
                     if v < args.samples_per_keyword]
    
    print(f"\n样本覆盖情况:")
    print(f"  完全覆盖的关键词: {len(keyword_samples) - len(poorly_covered)}")
    print(f"  覆盖不足的关键词: {len(poorly_covered)}")
    
    if poorly_covered:
        print(f"\n覆盖不足的关键词 (前10个):")
        for keyword, count in poorly_covered[:10]:
            print(f"  {keyword}: {count}/{args.samples_per_keyword} 个样本")
    
    print(f"\n实际运行时间: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print(f"预计时间 vs 实际时间: {estimated_time:.1f}秒 vs {elapsed_time:.1f}秒")
    
    # 建议
    if len(poorly_covered) > len(keyword_samples) * 0.3:  # 超过30%的关键词覆盖不足
        print(f"\n⚠️  警告: 大量关键词覆盖不足！")
        print("建议:")
        print("1. 检查关键词是否在正确的字段中")
        print("2. 尝试增加源数据量")
        print("3. 考虑使用语义匹配而不仅是字面匹配")


if __name__ == "__main__":
    main()