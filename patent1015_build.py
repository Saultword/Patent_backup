#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
from tkinter import Tk, filedialog, Label, Button, Frame, StringVar, messagebox, Checkbutton, BooleanVar

class PatentProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("专利分析工具")
        master.geometry("500x400")  # 增加高度以适应新控件
        
        # 输入输出路径变量
        self.input_folder = StringVar()
        self.output_folder = StringVar()
        self.generate_images = BooleanVar(value=True)  # 默认勾选生成图片
        self.enable_advanced_analysis = BooleanVar(value=True)  # 高级分析选项
        
        # 创建界面元素
        self.create_widgets()
        
        # 处理器实例
        self.processor = PatentProcessor()
    
    def create_widgets(self):
        # 主框架
        main_frame = Frame(self.master, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # 标题
        Label(main_frame, text="专利分析工具", font=('Arial', 16)).pack(pady=10)
        
        # 输入文件夹选择
        input_frame = Frame(main_frame)
        input_frame.pack(fill='x', pady=5)
        Label(input_frame, text="输入文件夹:").pack(side='left')
        Button(input_frame, text="浏览...", command=self.select_input_folder).pack(side='right', padx=5)
        self.input_label = Label(input_frame, textvariable=self.input_folder, relief='sunken', width=40, anchor='w')
        self.input_label.pack(side='right', expand=True, fill='x')
        
        # 输出文件夹选择
        output_frame = Frame(main_frame)
        output_frame.pack(fill='x', pady=5)
        Label(output_frame, text="输出文件夹:").pack(side='left')
        Button(output_frame, text="浏览...", command=self.select_output_folder).pack(side='right', padx=5)
        self.output_label = Label(output_frame, textvariable=self.output_folder, relief='sunken', width=40, anchor='w')
        self.output_label.pack(side='right', expand=True, fill='x')
        
        # 选项框架
        option_frame = Frame(main_frame)
        option_frame.pack(fill='x', pady=10)
        
        Checkbutton(option_frame, text="生成网络图片", variable=self.generate_images).pack(side='left', padx=10)
        Checkbutton(option_frame, text="启用高级网络分析", variable=self.enable_advanced_analysis).pack(side='left', padx=10)
        
        # 操作按钮
        button_frame = Frame(main_frame, pady=20)
        button_frame.pack()
        
        Button(button_frame, text="开始处理", command=self.start_processing, width=15).pack(side='left', padx=10)
        Button(button_frame, text="从CSV生成图片", command=self.generate_images_from_csv, width=15).pack(side='left', padx=10)
        Button(button_frame, text="退出", command=self.master.quit, width=15).pack(side='right', padx=10)
        
        # 状态信息
        self.status_label = Label(main_frame, text="就绪", relief='sunken', anchor='w')
        self.status_label.pack(fill='x', pady=5)
    
    def select_input_folder(self):
        folder = filedialog.askdirectory(title="选择输入文件夹")
        if folder:
            self.input_folder.set(folder)
    
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder.set(folder)
    
    def start_processing(self):
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        
        if not input_folder or not output_folder:
            messagebox.showerror("错误", "请先选择输入和输出文件夹！")
            return
        
        # 确认对话框
        confirm = messagebox.askyesno("确认", 
            f"即将开始处理:\n输入文件夹: {input_folder}\n输出文件夹: {output_folder}\n\n确定要继续吗?")
        
        if confirm:
            try:
                self.status_label.config(text="处理中...")
                self.master.update()
                
                # 执行处理
                self.processor.process_folder(input_folder, output_folder)
                
                if self.enable_advanced_analysis.get():
                    # 使用新代码的高级分析功能
                    stats = self.processor.analyze_networks(output_folder)
                    
                    # 读取统计结果用于显示
                    summary_path = os.path.join(output_folder, 'pruned_network_summary.csv')
                    if os.path.exists(summary_path):
                        summary_df = pd.read_csv(summary_path)
                        pruned_nodes = summary_df[summary_df['metric'] == 'pruned_network_nodes']['value'].values[0] if not summary_df.empty else 'N/A'
                        component_nodes = summary_df[summary_df['metric'] == 'pruned_largest_component_nodes']['value'].values[0] if not summary_df.empty else 'N/A'
                    else:
                        pruned_nodes = 'N/A'
                        component_nodes = 'N/A'
                    
                    message = (f"处理完成！\n\n"
                              f"专利家族记录: {len(self.processor.families)} 条\n"
                              f"引用关系记录: {len(self.processor.citations)} 条\n\n"
                              f"高级网络分析:\n"
                              f"- 精简网络节点数: {pruned_nodes}\n"
                              f"- 最大连通子图节点数: {component_nodes}\n"
                              f"- 节点影响分析已完成\n"
                              f"- 熵值权重分析已完成")
                else:
                    # 使用原有分析功能
                    stats = self.processor.basic_analyze_networks(output_folder, generate_images=self.generate_images.get())
                    message = (f"处理完成！\n\n"
                              f"专利家族记录: {len(self.processor.families)} 条\n"
                              f"引用关系记录: {len(self.processor.citations)} 条\n\n"
                              f"网络统计:\n"
                              f"- 总节点数: {stats['full_network_nodes']}\n"
                              f"- 最大连通子图节点数: {stats['largest_component_nodes']}\n"
                              f"- 覆盖率: {stats['coverage_ratio_nodes']:.2%}")
                
                self.status_label.config(text="处理完成")
                messagebox.showinfo("完成", message)
                
            except Exception as e:
                self.status_label.config(text="处理出错")
                messagebox.showerror("错误", f"处理过程中发生错误:\n{str(e)}")
    
    def generate_images_from_csv(self):
        output_folder = self.output_folder.get()
        
        if not output_folder:
            messagebox.showerror("错误", "请先选择输出文件夹！")
            return
        
        try:
            self.status_label.config(text="生成图片中...")
            self.master.update()
            
            # 检查并生成精简网络图片
            pruned_network_path = os.path.join(output_folder, 'pruned_network.csv')
            largest_component_path = os.path.join(output_folder, 'largest_component.csv')
            
            if os.path.exists(pruned_network_path) and os.path.exists(largest_component_path):
                # 从CSV构建网络
                pruned_network = self.processor.load_network_from_csv(pruned_network_path)
                largest_component = self.processor.load_network_from_csv(largest_component_path)
                
                # 生成图片
                self.processor.visualize_network(pruned_network, os.path.join(output_folder, 'pruned_network.png'), 
                                              title="Pruned Citation Network")
                self.processor.visualize_network(largest_component, os.path.join(output_folder, 'pruned_largest_component.png'),
                                              title="Largest Component of Pruned Network")
            
            # 检查并生成完整网络图片
            full_network_path = os.path.join(output_folder, 'full_network.csv')
            basic_component_path = os.path.join(output_folder, 'largest_component.csv')
            
            if os.path.exists(full_network_path) and os.path.exists(basic_component_path):
                full_network = self.processor.load_network_from_csv(full_network_path)
                basic_component = self.processor.load_network_from_csv(basic_component_path)
                
                self.processor.visualize_network(full_network, os.path.join(output_folder, 'full_network.png'), 
                                              title="Full Citation Network (Cited → Citing)")
                self.processor.visualize_network(basic_component, os.path.join(output_folder, 'largest_component.png'),
                                              title="Largest Connected Component (Cited → Citing)")
            
            self.status_label.config(text="图片生成完成")
            messagebox.showinfo("完成", "网络图片已从CSV文件成功生成！")
        except Exception as e:
            self.status_label.config(text="图片生成出错")
            messagebox.showerror("错误", f"生成图片时发生错误:\n{str(e)}")

class PatentProcessor:
    def __init__(self):
        self.families = []
        self.citations = []
        self.seen_hashes = set()
    
    def _is_valid_patent(self, patent_str):
        return bool(re.match(
            r'^[A-Z]{2}\d+[-][A-Z]?\d*$|^[A-Z]{2}\d+[-][A-Z]\d*[-][A-Z]\d+$',
            patent_str))
    
    def process_folder(self, input_folder, output_folder):
        """主处理流程"""
        print(f"开始扫描文件夹: {input_folder}")
        file_count = 0
        processed_blocks = 0
        
        for filename in os.listdir(input_folder):
            if not filename.endswith('.txt'):
                continue
            file_count += 1
            filepath = os.path.join(input_folder, filename)
            print(f"正在处理文件: {filename}...")
            processed_blocks += self._process_file(filepath)
        
        print(f"完成处理! 共处理 {file_count} 个文件，{processed_blocks} 个专利数据块")
        self._save_results(output_folder)
    
    def _process_file(self, filepath):
        """处理单个文件"""
        block_count = 0
        current_block = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('PT '):
                    current_block = [line]
                elif line.startswith('ER'):
                    current_block.append(line)
                    self._parse_block('\n'.join(current_block))
                    block_count += 1
                    current_block = []
                elif current_block:
                    current_block.append(line)
        return block_count
    
    def _parse_block(self, block_text):
        """解析单个专利数据块"""
        block_hash = hash(block_text.strip())
        if block_hash in self.seen_hashes:
            return
        self.seen_hashes.add(block_hash)
        
        try:
            family, citations = self._extract_relations(block_text)
            self.families.append(family)
            self.citations.extend(citations)
        except Exception as e:
            print(f"⚠️ 解析失败: {str(e)}\n片段预览: {block_text[:150]}...")
    
    def _save_results(self, output_folder):
        """保存结果到CSV"""
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成家族关系表
        family_records = []
        for family in self.families:
            for member in family['members']:
                member_citations = [
                    c['target'] for c in self.citations 
                    if c['source'] == member
                ]
                family_records.append({
                    'PatentFamily': family['family_id'],
                    'PatentNumber': member,
                    'CitedPatents': ';'.join(member_citations) or ''
                })
        
        # 生成引用关系表
        citation_records = [{
            'SourcePatent': c['source'],
            'CitedPatent': c['target']
        } for c in self.citations]
        
        # 保存文件
        family_path = os.path.join(output_folder, 'patent_families.csv')
        pd.DataFrame(family_records).to_csv(family_path, index=False)
        
        citation_path = os.path.join(output_folder, 'citation_relations.csv')
        pd.DataFrame(citation_records).to_csv(citation_path, index=False)
        
        print(f"结果已保存到: {output_folder}")
        print(f" - 专利家族记录: {len(family_records)} 条")
        print(f" - 引用关系记录: {len(citation_records)} 条")
    
    def _extract_relations(self, text):
        family = {'members': [], 'family_id': None}
        citations = []
        current_source = None  # 当前处理的源专利
        indent_level = 0       # 当前缩进级别
        
        for line in text.split('\n'):
            if not line.strip():
                continue
            
            # 检测字段标识
            if not line.startswith(' ') and len(line) >= 2:
                field = line[:2].strip()
                if field == 'PN':
                    family['members'] = [p.strip() for p in re.split(r';\s*', line[2:].strip()) if self._is_valid_patent(p)]
                    family['family_id'] = ';'.join(family['members'])
                elif field == 'CP':
                    current_source = re.split(r'\s+', line[2:].strip())[0]
                    indent_level = 0
                continue
            
            # 处理引用关系（带缩进分析）
            if line.startswith('      '):  # 4空格：被引用专利
                if current_source and indent_level == 1:
                    target = re.split(r'\s+', line.strip())[0]
                    if self._is_valid_patent(target):
                        citations.append({'source': current_source, 'target': target})
            elif line.startswith('  '):    # 2空格：次级源专利
                new_source = re.split(r'\s+', line.strip())[0]
                if self._is_valid_patent(new_source):
                    current_source = new_source
                    indent_level = 1

        return family, citations
    
    def build_citation_network(self):
        print("构建反向引用网络...")
        G = nx.DiGraph()
    
        for cite in self.citations:
            # 独特方向：被引专利(target) → 施引专利(source)
            # 表示被引专利"指向"引用它的专利
            if G.has_edge(cite['source'], cite['target']):
                G[cite['source']][cite['target']]['weight'] += 1
            else:
                G.add_edge(cite['source'], cite['target'], weight=1)
    
        print(f"反向网络构建完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
        return G
    
    def prune_network(self, G):
        """移除入度为0或出度为0的节点，形成精简网络"""
        if len(G.nodes()) == 0:
            print("警告: 网络为空，无法精简")
            return G
            
        print("开始精简网络：移除孤立节点...")
        initial_nodes = len(G.nodes())
        
        # 识别需要移除的节点（入度或出度为0）
        nodes_to_remove = [
            node for node in G.nodes() 
            if G.in_degree(node) == 0 or G.out_degree(node) == 0
        ]
        
        # 创建精简网络副本
        pruned_G = G.copy()
        pruned_G.remove_nodes_from(nodes_to_remove)
        
        final_nodes = len(pruned_G.nodes())
        print(f"网络精简完成: 移除 {len(nodes_to_remove)} 个节点 "
              f"({initial_nodes} → {final_nodes})")
        
        return pruned_G
    
    def analyze_networks(self, output_folder):
        """新代码的高级网络分析功能"""
        # 构建原始网络
        G = self.build_citation_network()
        
        # 精简网络（移除入度或出度为0的节点）
        pruned_G = self.prune_network(G)
        
        # 检查精简后的网络是否为空
        if len(pruned_G.nodes()) == 0:
            print("警告: 精简后网络为空，跳过后续分析")
            return {
                'status': 'error',
                'message': 'Pruned network is empty'
            }
        
        largest_cc = self.get_largest_component(pruned_G)
        
        # 保存网络数据（精简后）
        self.save_network_data(pruned_G, largest_cc, output_folder)
        
        # 计算并保存节点指标（精简网络）
        self.save_node_metrics(pruned_G, output_folder)
        
        # 可视化精简网络
        self.visualize_network(pruned_G, os.path.join(output_folder, 'pruned_network.png'), 
                            title="Pruned Citation Network")
        self.visualize_network(largest_cc, os.path.join(output_folder, 'pruned_largest_component.png'),
                            title="Largest Component of Pruned Network")
        
        # 计算网络统计指标（精简后）
        pruned_size = len(pruned_G.nodes())
        component_size = len(largest_cc.nodes())
        pruned_edges = len(pruned_G.edges())
        component_edges = len(largest_cc.edges())
        
        # 计算鲁棒性指标
        robustness_metrics = self.calculate_robustness_metrics(largest_cc)
        
        # 创建汇总统计
        summary_stats = pd.DataFrame({
            'metric': [
                'pruned_network_nodes', 'pruned_largest_component_nodes',
                'pruned_network_edges', 'pruned_largest_component_edges',
                'component_size_ratio', 'component_edges_ratio',
                'robustness_efficiency', 'robustness_connectivity', 'robustness_clustering'
            ],
            'value': [
                pruned_size, component_size,
                pruned_edges, component_edges,
                component_size/pruned_size if pruned_size > 0 else 0,
                component_edges/pruned_edges if pruned_edges > 0 else 0,
                robustness_metrics['efficiency'],
                robustness_metrics['connectivity'],
                robustness_metrics['clustering']
            ]
        })
        
        # 保存汇总统计
        summary_path = os.path.join(output_folder, 'pruned_network_summary.csv')
        summary_stats.to_csv(summary_path, index=False)
        print(f"精简网络统计已保存到: {summary_path}")
        
        # 节点影响分析（在精简网络的最大连通子图上）
        self.analyze_node_impact(largest_cc, output_folder)
        
        return summary_stats
    
    def analyze_node_impact(self, G, output_folder):
        if len(G.nodes()) == 0:
            print("警告：网络为空，跳过节点影响分析")
            return []

        # 创建结果文件
        impact_path = os.path.join(output_folder, 'node_impact_analysis.csv')
        os.makedirs(output_folder, exist_ok=True)

        # 获取原始最大连通子网
        original_largest_cc = self.get_largest_component(G)
        original_nodes = set(original_largest_cc.nodes())
        original_node_count = len(original_nodes)
 
        # 预先计算原始最大子网的指标
        undirected_original = original_largest_cc.to_undirected()
        initial_efficiency = nx.global_efficiency(undirected_original) if len(undirected_original.nodes()) > 1 else 0
        initial_clustering = nx.average_clustering(undirected_original) if len(undirected_original.nodes()) > 1 else 0

        print(f"开始节点影响分析: {original_node_count} 个节点")
        print(f"初始最大子网: {original_node_count}节点, 效率={initial_efficiency:.4f}, 聚类系数={initial_clustering:.4f}")

        # 创建结果文件
        with open(impact_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'node', 'networkEfficiency_before', 'ClusteringCoefficient_before',
                'networkEfficiency_after', 'ClusteringCoefficient_after',
                'connectivity_after', 'efficiency_change', 'clustering_change'
            ])
            writer.writeheader()
        
            processed_count = 0
            for i, node in enumerate(original_nodes):
                if (i+1) % 50 == 0 or (i+1) == original_node_count:
                    print(f"处理进度: {i+1}/{original_node_count} ({((i+1)/original_node_count)*100:.1f}%)")
            
                # 创建副本并移除节点
                G_removed = G.copy()
                G_removed.remove_node(node)
            
                # 获取移除节点后的最大连通子网
                largest_cc_after = self.get_largest_component(G_removed)
                after_node_count = len(largest_cc_after.nodes())
            
                # 计算连接度（剩余最大子网节点比例）
                connectivity_after = after_node_count / original_node_count
            
                # 计算效率变化（基于剩余最大子网）
                undirected_after = largest_cc_after.to_undirected()
                if after_node_count > 1:
                    try:
                        eff_after = nx.global_efficiency(undirected_after)
                    except:
                        eff_after = 0
                
                    # 计算平均聚类系数
                    clustering_after = nx.average_clustering(undirected_after)
                else:
                    eff_after = 0
                    clustering_after = 0
            
                # 计算变化量
                result = {
                    'node': node,
                    'networkEfficiency_before': initial_efficiency,
                    'ClusteringCoefficient_before': initial_clustering,
                    'networkEfficiency_after': eff_after,
                    'ClusteringCoefficient_after': clustering_after,
                    'connectivity_after': connectivity_after,
                    'efficiency_change': initial_efficiency - eff_after,
                    'clustering_change': initial_clustering - clustering_after
                }
            
                # 写入结果
                writer.writerow(result)
                processed_count += 1

        print(f"节点影响分析完成! 共处理 {processed_count} 个节点")
    
        # 新增功能：计算百分位排序
        self._calculate_percentile_ranks(impact_path, output_folder)
        self.calculate_entropy_weights_scores(output_folder)
        return processed_count
    
    def _calculate_percentile_ranks(self, impact_file_path, output_folder):
        print("开始计算百分位排序...")
    
        try:
            # 读取节点影响分析结果
            df = pd.read_csv(impact_file_path)
            
            # 定义需要计算百分位的三个change列
            change_columns = ['efficiency_change', 'clustering_change', 'connectivity_after']
            
            # 为每个change列计算百分位排名
            for col in change_columns:
                if col in df.columns:
                    percentile_col = f'{col}_percentile'
                    # 使用pct=True获取百分位排名（0到1之间）
                    df[percentile_col] = df[col].rank(pct=True)
                    print(f"已计算 {col} 的百分位排名: {percentile_col}")
            
            # 保存带有百分位排名的新文件
            ranked_file_path = os.path.join(output_folder, 'node_impact_analysis_with_percentiles.csv')
            df.to_csv(ranked_file_path, index=False)
            
            print(f"百分位排序完成! 结果已保存到: {ranked_file_path}")
            
            # 输出一些统计信息
            for col in change_columns:
                if col in df.columns:
                    percentile_col = f'{col}_percentile'
                    print(f"{col} 百分位统计:")
                    print(f"  - 最小值: {df[percentile_col].min():.3f}")
                    print(f"  - 中位数: {df[percentile_col].median():.3f}")
                    print(f"  - 最大值: {df[percentile_col].max():.3f}")
                    print(f"  - 平均值: {df[percentile_col].mean():.3f}")
                    
        except Exception as e:
            print(f"计算百分位排序时出错: {str(e)}")
    
    def calculate_entropy_weights_scores(self, output_folder):
        print("开始计算熵值、权重和综合得分...")
    
        try:
            # 读取带有百分位的数据文件
            file_path = os.path.join(output_folder, 'node_impact_analysis_with_percentiles.csv')
            df = pd.read_csv(file_path)
            
            # 三个百分位指标列
            percentile_columns = ['efficiency_change_percentile', 'clustering_change_percentile', 'connectivity_after_percentile']
            
            entropy_results = {}
            weights_results = {}
            
            # 第一步：计算每个指标的熵值
            for col in percentile_columns:
                if col not in df.columns:
                    print(f"警告: 未找到列 {col}，跳过该指标")
                    continue
                    
                data = df[col].dropna()
                n = len(data)
                
                if n == 0:
                    print(f"警告: 列 {col} 无有效数据，跳过")
                    continue
                
                # 归一化处理（百分位数据已经在[0,1]范围内）
                normalized_data = data
                
                # 计算 p_ij = Z_ij / sum(Z_ij)
                p_ij = normalized_data / normalized_data.sum()
                
                # 计算信息熵 e_j
                k = 1 / np.log(n)
                entropy = 0
                for p in p_ij:
                    if p > 0:
                        entropy += p * np.log(p)
                
                e_j = -k * entropy
                
                entropy_results[col] = {
                    'entropy': e_j,
                    'sample_size': n,
                    'data_range': (data.min(), data.max()),
                    'mean_value': data.mean()
                }
            
            # 第二步：计算权重
            if entropy_results:
                # 计算差异度 d_j = 1 - e_j
                total_d = 0
                weights_data = []
                
                for col, entropy_info in entropy_results.items():
                    e_j = entropy_info['entropy']
                    d_j = 1 - e_j  # 差异度
                    total_d += d_j
                    weights_data.append({
                        'metric': col,
                        'entropy': e_j,
                        'divergence': d_j
                    })
                
                # 计算权重 w_j = d_j / sum(d_j)
                for item in weights_data:
                    item['weight'] = item['divergence'] / total_d if total_d > 0 else 0
                
                # 第三步：计算每个节点的综合得分 S_i = Σ(w_j * Z_ij)
                # 创建权重字典便于计算
                weight_dict = {item['metric']: item['weight'] for item in weights_data}
                
                # 计算每个节点的综合得分
                df['comprehensive_score'] = 0
                for col in percentile_columns:
                    if col in df.columns and col in weight_dict:
                        df['comprehensive_score'] += weight_dict[col] * df[col]
                
                # 为综合得分也计算百分位排名
                df['comprehensive_score_percentile'] = df['comprehensive_score'].rank(pct=True)
                
                # 保存完整结果（包含综合得分）
                comprehensive_path = os.path.join(output_folder, 'comprehensive_node_scores.csv')
                df.to_csv(comprehensive_path, index=False)
                
                # 保存权重结果
                weights_df = pd.DataFrame(weights_data)
                weights_path = os.path.join(output_folder, 'entropy_weight_analysis.csv')
                weights_df.to_csv(weights_path, index=False)
                
                # 输出详细结果
                print("\n=== 熵值、权重和综合得分分析结果 ===")
                for item in weights_data:
                    print(f"{item['metric']}:")
                    print(f"  信息熵 e_j = {item['entropy']:.6f}")
                    print(f"  差异度 d_j = {item['divergence']:.6f}")
                    print(f"  权重 w_j = {item['weight']:.6f} ({item['weight']*100:.2f}%)")
                
                print(f"\n权重总和: {sum(item['weight'] for item in weights_data):.6f}")
                
                # 输出综合得分统计
                print(f"\n综合得分统计:")
                print(f"  最小值: {df['comprehensive_score'].min():.6f}")
                print(f"  最大值: {df['comprehensive_score'].max():.6f}")
                print(f"  平均值: {df['comprehensive_score'].mean():.6f}")
                print(f"  中位数: {df['comprehensive_score'].median():.6f}")
                
                # 输出排名前10的节点
                top_nodes = df.nlargest(10, 'comprehensive_score')[['node', 'comprehensive_score', 'comprehensive_score_percentile']]
                print(f"\n综合得分前10的节点:")
                for _, row in top_nodes.iterrows():
                    print(f"  {row['node']}: {row['comprehensive_score']:.6f} (百分位: {row['comprehensive_score_percentile']:.3f})")
                
                print(f"\n结果已保存到:")
                print(f"  - 完整节点得分: {comprehensive_path}")
                print(f"  - 权重分析: {weights_path}")
                
                return {
                    'entropy_results': entropy_results,
                    'weight_results': weights_data,
                    'comprehensive_scores': df[['node', 'comprehensive_score', 'comprehensive_score_percentile']].to_dict('records'),
                    'top_nodes': top_nodes.to_dict('records')
                }
            else:
                print("警告: 未计算任何指标的熵值")
                return {}
                
        except Exception as e:
            print(f"计算熵值、权重和得分时出错: {str(e)}")
            return {}
    
    def basic_analyze_networks(self, output_folder, generate_images=True):
        """原有的基础网络分析功能"""
        G = self.build_citation_network()
        largest_cc = self.get_largest_component(G)
        
        # 保存网络数据
        self.save_basic_network_data(G, largest_cc, output_folder)
        
        # 计算并保存节点指标
        self.save_basic_node_metrics(G, output_folder)
        
        # 可选: 可视化完整网络和最大连通子网
        if generate_images:
            self.visualize_network(G, os.path.join(output_folder, 'full_network.png'), 
                                title="Full Citation Network (Cited → Citing)")
            self.visualize_network(largest_cc, os.path.join(output_folder, 'largest_component.png'),
                                title="Largest Connected Component (Cited → Citing)")
        
        # 计算网络统计指标
        total_size = len(G.nodes())
        component_size = len(largest_cc.nodes())
        total_edges = len(G.edges())
        component_edges = len(largest_cc.edges())
        
        # 创建汇总统计DataFrame
        summary_stats = pd.DataFrame({
            'metric': ['total_size', 'component_size', 'total_edges', 'component_edges', 
                      'component_size_ratio', 'component_edges_ratio'],
            'value': [total_size, component_size, total_edges, component_edges,
                     component_size/total_size if total_size > 0 else 0,
                     component_edges/total_edges if total_edges > 0 else 0]
        })
        
        # 保存汇总统计到CSV
        summary_path = os.path.join(output_folder, 'network_summary_stats.csv')
        summary_stats.to_csv(summary_path, index=False)
        print(f"网络汇总统计已保存到: {summary_path}")
        
        # 返回统计信息
        return {
            "full_network_nodes": total_size,
            "largest_component_nodes": component_size,
            "full_network_edges": total_edges,
            "largest_component_edges": component_edges,
            "coverage_ratio_nodes": component_size / total_size if total_size > 0 else 0,
            "coverage_ratio_edges": component_edges / total_edges if total_edges > 0 else 0
        }
    
    def visualize_network(self, G, output_path, title=None):
        if len(G.nodes()) == 0:
            print(f"警告: 无法可视化空网络 - {output_path}")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 计算布局
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
        
        # 绘制节点
        in_degrees = dict(G.in_degree())
        node_sizes = [in_degrees[n] * 50 + 10 for n in G.nodes()]
        
        # 添加节点颜色基于聚类系数
        clustering = nx.clustering(G.to_undirected())
        node_colors = [clustering[n] for n in G.nodes()]
         
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            alpha=0.8,
            vmin=0, vmax=1
        )
        
        # 添加颜色条
        plt.colorbar(nodes, label='Clustering Coefficient')
        
        # 绘制边
        edge_widths = [d.get('weight', 1) * 0.8 for _, _, d in G.edges(data=True)]
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color='gray',
            arrowsize=10,
            arrowstyle='->'
        )
        
        # 只标注重要节点
        if len(G.nodes()) > 0:
            important_nodes = [n for n in G.nodes() if in_degrees[n] > np.percentile(list(in_degrees.values()), 90)]
            labels = {n: n for n in important_nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # 设置标题
        if title is None:
            title = "Patent Citation Network (Cited → Citing)"
        plt.title(title, fontsize=14)
        
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_largest_component(self, G):
        """获取最大弱连通子图"""
        if len(G.nodes()) == 0:
            return G
            
        undirected = G.to_undirected()
        largest_cc_nodes = max(nx.connected_components(undirected), key=len)
        return G.subgraph(largest_cc_nodes).copy()
    
    def save_network_data(self, G, largest_cc, output_folder):
        """保存完整网络和最大子图到CSV（新代码版本）"""
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存完整网络
        full_edges = [
            {"Source": u, "Target": v, "Weight": d['weight']}
            for u, v, d in G.edges(data=True)
        ]
        pd.DataFrame(full_edges).to_csv(
            os.path.join(output_folder, 'pruned_network.csv'),
            index=False
        )
        print(f"保存精简网络: {len(full_edges)} 条边")
        
        # 保存最大连通子图
        lcc_edges = [
            {"Source": u, "Target": v, "Weight": d['weight']}
            for u, v, d in largest_cc.edges(data=True)
        ]
        pd.DataFrame(lcc_edges).to_csv(
            os.path.join(output_folder, 'largest_component.csv'),
            index=False
        )
        print(f"保存最大子网: {len(lcc_edges)} 条边")
    
    def save_basic_network_data(self, G, largest_cc, output_folder):
        """保存完整网络和最大子图到CSV（原有版本）"""
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存完整网络
        full_edges = [
            {"Source": u, "Target": v, "Weight": d['weight']}
            for u, v, d in G.edges(data=True)
        ]
        pd.DataFrame(full_edges).to_csv(
            os.path.join(output_folder, 'full_network.csv'),
            index=False
        )
        
        # 保存最大连通子图
        lcc_edges = [
            {"Source": u, "Target": v, "Weight": d['weight']}
            for u, v, d in largest_cc.edges(data=True)
        ]
        pd.DataFrame(lcc_edges).to_csv(
            os.path.join(output_folder, 'largest_component.csv'),
            index=False
        )
    
    def save_node_metrics(self, G, output_folder):
        """保存节点指标（新代码版本）"""
        if len(G.nodes()) == 0:
            print("警告: 无法保存空网络的节点指标")
            return
            
        metrics = []
        
        # 预先计算全局指标
        undirected = G.to_undirected()
        global_eff = nx.global_efficiency(undirected) if len(undirected.nodes()) > 1 else 0
        betweenness = nx.betweenness_centrality(G)
        
        # 修复接近中心性计算
        if nx.is_strongly_connected(G):
            closeness = nx.closeness_centrality(G)
        else:
            closeness = {n: 0 for n in G.nodes()}
            
        clustering = nx.clustering(undirected)
        
        # 获取所有连通组件
        components = list(nx.connected_components(undirected))
        component_dict = {n: c for c in components for n in c}
        
        # 预先计算每个子网的效率
        subgraph_efficiencies = {}
        for component in components:
            if len(component) > 1:
                subgraph = undirected.subgraph(component)
                eff = nx.global_efficiency(subgraph)
            else:
                eff = 0
            for node in component:
                subgraph_efficiencies[node] = eff

        # 修复循环缩进问题
        for node in G.nodes():
            component = component_dict.get(node, {node})
            component_size = len(component)
            connectivity = component_size / len(G.nodes()) if len(G.nodes()) > 0 else 0

            metrics.append({
                'PatentNumber': node,
                'CitedByOthers': G.out_degree(node),
                'CitesOthers': G.in_degree(node),
                'TotalDegree': G.in_degree(node) + G.out_degree(node),
                'GlobalEfficiency': global_eff,
                'SubgraphEfficiency': subgraph_efficiencies.get(node, 0),
                'BetweennessCentrality': betweenness.get(node, 0),
                'ClosenessCentrality': closeness.get(node, 0),
                'ClusteringCoefficient': clustering.get(node, 0),
                'ComponentSize': component_size,
                'Connectivity': round(connectivity, 4)
            })

        # 保存到CSV
        pd.DataFrame(metrics).to_csv(
            os.path.join(output_folder, 'node_metrics.csv'),
            index=False
        )
        print(f"节点指标已保存到: {output_folder} (共 {len(metrics)} 条记录)")
    
    def save_basic_node_metrics(self, G, output_folder):
        """计算节点指标（原有版本）"""
        metrics = []
        
        # 预先计算全局指标
        global_eff = nx.global_efficiency(G.to_undirected())
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G) if nx.is_strongly_connected(G) else {n:0 for n in G.nodes()}
        clustering = nx.clustering(G.to_undirected())  # 计算聚类系数
        
        # 获取所有连通组件（优化性能，避免重复计算）
        undirected = G.to_undirected()
        components = {n: c for c in nx.connected_components(undirected) for n in c}
        total_nodes = len(G.nodes())

        for node in G.nodes():
            # 计算该节点所在连通子图的连接度
            component_size = len(components[node])
            connectivity = component_size / total_nodes if total_nodes > 0 else 0

            metrics.append({
                'PatentNumber': node,
                'InDegree': G.in_degree(node),
                'OutDegree': G.out_degree(node),
                'TotalDegree': G.in_degree(node) + G.out_degree(node),
                'GlobalEfficiency': global_eff,
                'BetweennessCentrality': betweenness.get(node, 0),
                'ClosenessCentrality': closeness.get(node, 0),
                'ClusteringCoefficient': clustering.get(node, 0),  # 新增聚类系数
                'ComponentSize': component_size,          # 所在连通子图的节点数
                'Connectivity': round(connectivity, 4)    # 连接度百分比
            })

        # 保存到CSV
        pd.DataFrame(metrics).to_csv(
            os.path.join(output_folder, 'node_metrics.csv'),
            index=False
        )
        print(f"节点指标已保存到: {output_folder}")

    def calculate_robustness_metrics(self, G):
        """计算三个关键鲁棒性指标"""
        if len(G.nodes()) == 0:
            return {
                'efficiency': 0,
                'connectivity': 0,
                'clustering': 0
            }
            
        undirected = G.to_undirected()
        
        # 1. 全局效率
        efficiency = nx.global_efficiency(undirected) if len(G.nodes()) > 1 else 0
        
        # 2. 连通性
        connectivity = len(G.nodes()) / len(undirected.nodes()) if len(undirected.nodes()) > 0 else 0
        
        # 3. 平均聚类系数
        clustering = nx.average_clustering(undirected)
        
        return {
            'efficiency': round(efficiency, 4),
            'connectivity': round(connectivity, 4),
            'clustering': round(clustering, 4)
        }
    
    def load_network_from_csv(self, csv_path):
        """从CSV文件加载网络"""
        if not os.path.exists(csv_path):
            print(f"警告: 文件不存在 {csv_path}")
            return nx.DiGraph()
            
        df = pd.read_csv(csv_path)
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            source = row['Source']
            target = row['Target']
            weight = row.get('Weight', 1)
            
            if G.has_edge(source, target):
                G[source][target]['weight'] += weight
            else:
                G.add_edge(source, target, weight=weight)
        
        print(f"从 {csv_path} 加载网络: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
        return G

def main():
    root = Tk()
    app = PatentProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()