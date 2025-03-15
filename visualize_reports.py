import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from tqdm import tqdm
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FinancialReportVisualizer:
    def __init__(self, data_dir='data', output_dir='visualizations'):
        """初始化财报可视化器"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.years = []
        self.data_files = []
        self.data = {}
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self._load_file_list()
    
    def _load_file_list(self):
        """加载数据文件列表"""
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                year = file.split('.')[0]
                if year.isdigit():
                    self.years.append(int(year))
                    self.data_files.append(os.path.join(self.data_dir, file))
        
        # 按年份排序
        sorted_data = sorted(zip(self.years, self.data_files))
        self.years, self.data_files = zip(*sorted_data)
        print(f"找到{len(self.years)}个年份的财报数据: {', '.join(map(str, self.years))}")
    
    def load_sample_data(self, sample_size=1000, random_state=42):
        """从每个年份加载样本数据"""
        for year, file_path in zip(self.years, self.data_files):
            print(f"加载{year}年数据样本...")
            try:
                # 尝试读取CSV文件，假设第一列是文本数据
                df = pd.read_csv(file_path, header=None, encoding='utf-8')
                
                # 如果数据是以制表符分隔的
                if len(df.columns) == 1 and '\t' in str(df.iloc[0, 0]):
                    df = pd.read_csv(file_path, header=None, sep='\t', encoding='utf-8')
                
                # 随机抽样
                if len(df) > sample_size:
                    df = df.sample(sample_size, random_state=random_state)
                
                self.data[year] = df
                print(f"  成功加载{len(df)}条记录")
            except Exception as e:
                print(f"  加载{year}年数据时出错: {str(e)}")
    
    def visualize_file_sizes(self):
        """可视化各年份文件大小"""
        file_sizes = []
        for file_path in self.data_files:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
            file_sizes.append(size_mb)
        
        plt.figure(figsize=(12, 6))
        plt.bar(self.years, file_sizes, color='skyblue')
        plt.title('各年份财报数据文件大小')
        plt.xlabel('年份')
        plt.ylabel('文件大小 (MB)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(self.years, rotation=45)
        
        # 添加数值标签
        for i, v in enumerate(file_sizes):
            plt.text(i, v + 1, f'{v:.1f}MB', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'file_sizes.png'), dpi=300)
        plt.close()
        print("文件大小可视化完成")
    
    def visualize_text_complexity(self):
        """可视化文本复杂度"""
        if not self.data:
            print("请先加载数据")
            return
        
        avg_word_lengths = []
        avg_sentence_lengths = []
        
        for year in self.years:
            if year not in self.data:
                continue
                
            df = self.data[year]
            text_col = df.columns[-1]
            texts = df[text_col].astype(str).tolist()
            
            # 计算平均单词长度
            all_words = []
            for text in texts:
                words = re.findall(r'\b[a-zA-Z]{1,}\b', text.lower())
                all_words.extend(words)
            
            if all_words:
                avg_word_len = sum(len(word) for word in all_words) / len(all_words)
                avg_word_lengths.append(avg_word_len)
            else:
                avg_word_lengths.append(0)
            
            # 计算平均句子长度
            all_sentences = []
            for text in texts:
                sentences = re.split(r'[.!?]', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                all_sentences.extend(sentences)
            
            if all_sentences:
                avg_sent_len = sum(len(s.split()) for s in all_sentences) / len(all_sentences)
                avg_sentence_lengths.append(avg_sent_len)
            else:
                avg_sentence_lengths.append(0)
        
        # 绘制平均单词长度和句子长度
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('年份')
        ax1.set_ylabel('平均单词长度', color=color)
        ax1.plot(self.years, avg_word_lengths, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('平均句子长度 (单词数)', color=color)
        ax2.plot(self.years, avg_sentence_lengths, color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('财报文本复杂度随时间变化')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(self.years, rotation=45)
        
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'text_complexity.png'), dpi=300)
        plt.close()
        print("文本复杂度可视化完成")
    
    def visualize_topic_distribution(self, n_topics=5):
        """可视化主题分布"""
        if not self.data:
            print("请先加载数据")
            return
        
        # 定义主题关键词
        topics = {
            '财务状况': ['financial', 'condition', 'balance', 'sheet', 'assets', 'liabilities'],
            '经营业绩': ['operations', 'performance', 'results', 'income', 'revenue', 'sales'],
            '风险因素': ['risk', 'uncertainty', 'challenge', 'competition', 'market', 'economic'],
            '发展战略': ['strategy', 'growth', 'development', 'future', 'plan', 'expansion'],
            '公司治理': ['governance', 'management', 'board', 'executive', 'committee', 'director']
        }
        
        topic_mentions = {topic: [] for topic in topics}
        
        for year in self.years:
            if year not in self.data:
                continue
                
            df = self.data[year]
            text_col = df.columns[-1]
            all_text = ' '.join(df[text_col].astype(str).lower())
            
            for topic, keywords in topics.items():
                count = sum(all_text.count(keyword) for keyword in keywords)
                topic_mentions[topic].append(count)
        
        # 绘制主题分布
        plt.figure(figsize=(14, 8))
        
        for topic, counts in topic_mentions.items():
            plt.plot(self.years, counts, marker='o', linewidth=2, label=topic)
        
        plt.title('财报主题分布随时间变化')
        plt.xlabel('年份')
        plt.ylabel('关键词提及次数')
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'topic_distribution.png'), dpi=300)
        plt.close()
        print("主题分布可视化完成")
    
    def visualize_word_trends(self, target_words=None):
        """可视化特定词汇的趋势"""
        if not self.data:
            print("请先加载数据")
            return
        
        if target_words is None:
            target_words = [
                'technology', 'digital', 'innovation', 'growth', 
                'profit', 'revenue', 'cost', 'expense',
                'risk', 'challenge', 'opportunity', 'strategy'
            ]
        
        word_counts = {word: [] for word in target_words}
        
        for year in self.years:
            if year not in self.data:
                continue
                
            df = self.data[year]
            text_col = df.columns[-1]
            all_text = ' '.join(df[text_col].astype(str).lower())
            
            for word in target_words:
                # 使用正则表达式匹配完整单词
                pattern = r'\b' + word + r'\b'
                count = len(re.findall(pattern, all_text))
                word_counts[word].append(count)
        
        # 绘制词汇趋势
        plt.figure(figsize=(14, 8))
        
        for word, counts in word_counts.items():
            plt.plot(self.years, counts, marker='o', linewidth=2, label=word)
        
        plt.title('财报关键词趋势随时间变化')
        plt.xlabel('年份')
        plt.ylabel('词汇出现次数')
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'word_trends.png'), dpi=300)
        plt.close()
        print("词汇趋势可视化完成")
    
    def visualize_text_similarity(self):
        """可视化不同年份文本的相似性"""
        if not self.data:
            print("请先加载数据")
            return
        
        # 提取每年的文本样本
        year_texts = {}
        for year in self.years:
            if year not in self.data:
                continue
                
            df = self.data[year]
            text_col = df.columns[-1]
            texts = df[text_col].astype(str).tolist()
            
            # 合并所有文本
            year_texts[year] = ' '.join(texts)
        
        # 使用TF-IDF向量化文本
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(list(year_texts.values()))
        
        # 计算相似度矩阵 (余弦相似度)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(X)
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=list(year_texts.keys()),
                   yticklabels=list(year_texts.keys()))
        
        plt.title('不同年份财报文本相似度')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'text_similarity.png'), dpi=300)
        plt.close()
        
        # 使用降维可视化文本相似性
        if len(year_texts) >= 3:  # 至少需要3个点才能进行有意义的降维
            # 使用PCA降维
            pca = PCA(n_components=2)
            X_dense = X.toarray()
            X_pca = pca.fit_transform(X_dense)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], s=100)
            
            # 添加年份标签
            for i, year in enumerate(year_texts.keys()):
                plt.annotate(str(year), (X_pca[i, 0], X_pca[i, 1]), fontsize=12)
            
            plt.title('财报文本内容相似性 (PCA降维)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'text_similarity_pca.png'), dpi=300)
            plt.close()
        
        print("文本相似性可视化完成")
    
    def visualize_readability_scores(self):
        """可视化可读性分数"""
        if not self.data:
            print("请先加载数据")
            return
        
        # 简化的Flesch-Kincaid可读性评分计算
        def calculate_readability(text):
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0
            
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            if not words:
                return 0
            
            # 计算平均句子长度
            avg_sentence_length = len(words) / len(sentences)
            
            # 计算平均单词长度
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # 简化的Flesch-Kincaid Grade Level公式
            readability = 0.39 * avg_sentence_length + 11.8 * avg_word_length - 15.59
            
            return readability
        
        readability_scores = []
        
        for year in self.years:
            if year not in self.data:
                continue
                
            df = self.data[year]
            text_col = df.columns[-1]
            
            # 计算每个文档的可读性分数
            scores = []
            for text in df[text_col].astype(str):
                if len(text) > 100:  # 只计算足够长的文本
                    score = calculate_readability(text)
                    scores.append(score)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                readability_scores.append(avg_score)
            else:
                readability_scores.append(0)
        
        # 绘制可读性分数
        plt.figure(figsize=(12, 6))
        plt.plot(self.years, readability_scores, marker='o', linewidth=2, color='purple')
        
        plt.title('财报可读性分数随时间变化')
        plt.xlabel('年份')
        plt.ylabel('Flesch-Kincaid Grade Level')
        plt.grid(True)
        plt.xticks(self.years, rotation=45)
        
        # 添加解释
        plt.figtext(0.5, 0.01, 
                   '注: 分数越高表示文本越复杂，需要更高的阅读水平理解', 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'readability_scores.png'), dpi=300)
        plt.close()
        print("可读性分数可视化完成")
    
    def visualize_financial_ratios(self):
        """可视化财务比率提及"""
        if not self.data:
            print("请先加载数据")
            return
        
        # 定义常见财务比率
        ratios = {
            'ROI/ROE/ROA': r'\b(roi|roe|roa|return on (investment|equity|assets))\b',
            'P/E比率': r'\b(p/e|price[- ]to[- ]earnings)\b',
            '利润率': r'\b(profit margin|gross margin|net margin)\b',
            '资产负债率': r'\b(debt[- ]to[- ]equity|leverage ratio|debt ratio)\b',
            '流动比率': r'\b(current ratio|liquidity ratio)\b',
            'EPS': r'\b(eps|earnings per share)\b'
        }
        
        ratio_mentions = {ratio: [] for ratio in ratios}
        
        for year in self.years:
            if year not in self.data:
                continue
                
            df = self.data[year]
            text_col = df.columns[-1]
            all_text = ' '.join(df[text_col].astype(str).lower())
            
            for ratio, pattern in ratios.items():
                count = len(re.findall(pattern, all_text))
                ratio_mentions[ratio].append(count)
        
        # 绘制财务比率提及
        plt.figure(figsize=(14, 8))
        
        for ratio, counts in ratio_mentions.items():
            plt.plot(self.years, counts, marker='o', linewidth=2, label=ratio)
        
        plt.title('财务比率提及频率随时间变化')
        plt.xlabel('年份')
        plt.ylabel('提及次数')
        plt.legend()
        plt.grid(True)
        plt.xticks(self.years, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'financial_ratios.png'), dpi=300)
        plt.close()
        print("财务比率提及可视化完成")
    
    def run_all_visualizations(self, sample_size=1000):
        """运行所有可视化"""
        print("开始全面可视化财报数据...")
        self.load_sample_data(sample_size=sample_size)
        
        self.visualize_file_sizes()
        self.visualize_text_complexity()
        self.visualize_topic_distribution()
        self.visualize_word_trends()
        self.visualize_text_similarity()
        self.visualize_readability_scores()
        self.visualize_financial_ratios()
        
        print(f"全部可视化完成！结果保存在 {self.output_dir} 目录")

if __name__ == "__main__":
    visualizer = FinancialReportVisualizer(output_dir='report_visualizations')
    visualizer.run_all_visualizations(sample_size=2000) 