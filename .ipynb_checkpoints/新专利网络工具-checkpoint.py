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
            if G.has_edge( cite['source'],cite['target']):
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
        return processed_count
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
        """保存完整网络和最大子图到CSV"""
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存完整网络 - 修复文件名与调用一致
        full_edges = [
            {"Source": u, "Target": v, "Weight": d['weight']}
            for u, v, d in G.edges(data=True)
        ]
        pd.DataFrame(full_edges).to_csv(
            os.path.join(output_folder, 'pruned_network.csv'),  # 修复文件名
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
    
    def save_node_metrics(self, G, output_folder):
        """修复缩进问题"""
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
    


# 实际调用示例
if __name__ == '__main__':
    # 创建处理器实例
    processor = PatentProcessor()
    
    # 设置输入输出路径
    input_folder = 'data'  # 包含专利txt文件的文件夹
    output_folder = input_folder + '-results'  # 结果输出文件夹
    
    # 执行处理流程
    print("=== 开始处理专利数据 ===")
    processor.process_folder(input_folder, output_folder)
    
    print("\n=== 开始分析引用网络 ===")
    stats = processor.analyze_networks(output_folder)
    
    print("\n处理完成！所有结果已保存到", output_folder)