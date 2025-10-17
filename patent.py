#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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
        """构建反向引用网络（被引 → 施引）"""
        G = nx.DiGraph()
        citation_map = defaultdict(list)
        
        for cite in self.citations:
            citation_map[cite['target']].append(cite['source'])  # 反向记录
        
        for target, sources in citation_map.items():
            source_counts = defaultdict(int)
            for src in sources:
                source_counts[src] += 1
            
            for src, weight in source_counts.items():
                G.add_edge(target, src, weight=weight)
        
        return G
    
    def visualize_network(self, G, output_path):
        """可视化反向引用网络"""
        plt.figure(figsize=(15, 10))
        
        # 计算布局
        pos = nx.spring_layout(G, k=0.2, iterations=50)
        
        # 绘制节点（按入度大小调整）
        in_degrees = dict(G.in_degree())
        node_sizes = [in_degrees[n] * 50 + 10 for n in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.8
        )
        
        # 绘制边（按权重调整）
        edge_widths = [d['weight'] * 0.8 for _, _, d in G.edges(data=True)]
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color='gray',
            arrowsize=15,
            arrowstyle='->'
        )
        
        # 标注高权重边
        edge_labels = {
            (u, v): d['weight']
            for u, v, d in G.edges(data=True)
            if d['weight'] > 2
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        plt.title("Patent Citation Network (Cited → Citing)", fontsize=14)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_networks(self, output_folder):
     G = self.build_citation_network()
     largest_cc = self.get_largest_component(G)
    
    # 保存网络数据
     self.save_network_data(G, largest_cc, output_folder)
    
    # 计算并保存节点指标
     self.save_node_metrics(G, output_folder)
    
    # 可视化完整网络和最大连通子网
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
     plt.figure(figsize=(15, 10))
    
    # 计算布局 - 使用spring_layout但调整参数以获得更好的可视化效果
     pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)  # 添加seed保证可重复性
    
    # 绘制节点（按入度大小调整）
     in_degrees = dict(G.in_degree())
     node_sizes = [in_degrees[n] * 50 + 10 for n in G.nodes()]
    
    # 添加节点颜色基于聚类系数
     clustering = nx.clustering(G.to_undirected())
     node_colors = [clustering[n] for n in G.nodes()]  # 使用聚类系数作为颜色
     
     nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,  # 使用颜色映射
        alpha=0.8,
        vmin=0, vmax=1  # 聚类系数范围0-1
    )
    
    # 添加颜色条
     plt.colorbar(nodes, label='Clustering Coefficient')
    
    # 绘制边（按权重调整）
     edge_widths = [d.get('weight', 1) * 0.8 for _, _, d in G.edges(data=True)]
    
     nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',
        arrowsize=10,  # 调小箭头大小
        arrowstyle='->'
    )
    
    # 只标注重要节点（避免图像过于拥挤）
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
        undirected = G.to_undirected()
        largest_cc_nodes = max(nx.connected_components(undirected), key=len)
        return G.subgraph(largest_cc_nodes).copy()
    
    def save_network_data(self, G, largest_cc, output_folder):
        """保存完整网络和最大子图到CSV"""
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
     """计算节点指标（包含聚类系数）"""
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

# 实际调用示例
if __name__ == '__main__':
    # 创建处理器实例
    processor = PatentProcessor()
    
    # 设置输入输出路径（根据实际情况修改）
    input_folder = 'data'  # 包含专利txt文件的文件夹
    output_folder = input_folder+'-results'  # 结果输出文件夹
    
    # 执行处理流程
    print("=== 开始处理专利数据 ===")
    processor.process_folder(input_folder, output_folder)
    
    print("\n=== 开始分析引用网络 ===")
    stats = processor.analyze_networks(output_folder)
    
    
    print("\n处理完成！所有结果已保存到", output_folder)