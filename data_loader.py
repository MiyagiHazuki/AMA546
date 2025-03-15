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
import jieba
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FinancialReportAnalyzer:
    def __init__(self, data_dir='data'):
        """初始化财报分析器"""
        self.data_dir = data_dir
        self.years = []
        self.data_files = []
        self.data = {}
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
    
    def analyze_data_distribution(self):
        """分析数据分布"""
        if not self.data:
            print("请先加载数据")
            return
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. 每年数据量分布
        plt.subplot(2, 2, 1)
        years = list(self.data.keys())
        counts = [len(df) for df in self.data.values()]
        plt.bar(years, counts)
        plt.title('各年份数据量分布')
        plt.xlabel('年份')
        plt.ylabel('数据量')
        plt.xticks(rotation=45)
        
        # 2. 文本长度分布
        plt.subplot(2, 2, 2)
        all_lengths = []
        for year, df in self.data.items():
            # 假设文本在最后一列
            text_col = df.columns[-1]
            lengths = df[text_col].astype(str).apply(len)
            all_lengths.extend(lengths)
        
        sns.histplot(all_lengths, bins=50, kde=True)
        plt.title('文本长度分布')
        plt.xlabel('文本长度')
        plt.ylabel('频率')
        
        # 3. 每年平均文本长度
        plt.subplot(2, 2, 3)
        avg_lengths = []
        for year, df in self.data.items():
            text_col = df.columns[-1]
            avg_length = df[text_col].astype(str).apply(len).mean()
            avg_lengths.append(avg_length)
        
        plt.bar(years, avg_lengths)
        plt.title('各年份平均文本长度')
        plt.xlabel('年份')
        plt.ylabel('平均长度')
        plt.xticks(rotation=45)
        
        # 4. 文本长度随时间变化的箱线图
        plt.subplot(2, 2, 4)
        length_data = []
        year_labels = []
        for year, df in self.data.items():
            text_col = df.columns[-1]
            lengths = df[text_col].astype(str).apply(len)
            length_data.append(lengths)
            year_labels.extend([str(year)] * len(lengths))
        
        plt.boxplot(length_data, labels=years)
        plt.title('各年份文本长度分布')
        plt.xlabel('年份')
        plt.ylabel('文本长度')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300)
        plt.close()
        print("数据分布分析完成，已保存为 data_distribution.png")
    
    def analyze_text_features(self, year=None, top_n=20):
        """分析文本特征"""
        if not self.data:
            print("请先加载数据")
            return
        
        if year is not None and year not in self.data:
            print(f"未找到{year}年的数据")
            return
        
        years_to_analyze = [year] if year else list(self.data.keys())
        
        for year in years_to_analyze:
            print(f"分析{year}年文本特征...")
            df = self.data[year]
            text_col = df.columns[-1]
            texts = df[text_col].astype(str).tolist()
            
            # 提取所有单词
            all_words = []
            for text in texts:
                # 使用正则表达式提取英文单词
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                all_words.extend(words)
            
            # 计算词频
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(top_n)
            
            # 绘制词频图
            plt.figure(figsize=(12, 6))
            words, counts = zip(*top_words)
            plt.barh(range(len(words)), counts, align='center')
            plt.yticks(range(len(words)), words)
            plt.title(f'{year}年财报高频词汇')
            plt.xlabel('出现次数')
            plt.tight_layout()
            plt.savefig(f'word_freq_{year}.png', dpi=300)
            plt.close()
            
            # 生成词云
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                 max_words=100, contour_width=3, contour_color='steelblue')
            wordcloud.generate_from_frequencies(dict(word_counts))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{year}年财报词云')
            plt.tight_layout()
            plt.savefig(f'wordcloud_{year}.png', dpi=300)
            plt.close()
            
            print(f"  {year}年文本特征分析完成")
    
    def analyze_financial_metrics(self):
        """分析财务指标"""
        if not self.data:
            print("请先加载数据")
            return
        
        # 尝试提取常见财务指标
        metrics = {
            'revenue': r'revenue|sales|net sales',
            'profit': r'profit|gross profit|net income',
            'assets': r'assets|total assets',
            'liabilities': r'liabilities|total liabilities',
            'cash': r'cash|cash flow|cash and cash equivalents',
            'expenses': r'expenses|operating expenses'
        }
        
        results = {metric: [] for metric in metrics}
        years = []
        
        for year, df in self.data.items():
            years.append(year)
            text_col = df.columns[-1]
            all_text = ' '.join(df[text_col].astype(str))
            
            for metric, pattern in metrics.items():
                # 计算每个指标的提及次数
                mentions = len(re.findall(pattern, all_text.lower()))
                results[metric].append(mentions)
        
        # 绘制财务指标提及频率
        plt.figure(figsize=(12, 8))
        for metric, counts in results.items():
            plt.plot(years, counts, marker='o', label=metric)
        
        plt.title('财务指标提及频率随时间变化')
        plt.xlabel('年份')
        plt.ylabel('提及次数')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('financial_metrics.png', dpi=300)
        plt.close()
        print("财务指标分析完成，已保存为 financial_metrics.png")
    
    def analyze_sentiment_over_time(self):
        """分析情感随时间变化"""
        if not self.data:
            print("请先加载数据")
            return
        
        # 简单的情感词典
        positive_words = ['increase', 'growth', 'profit', 'success', 'positive', 'improved', 
                         'higher', 'gain', 'favorable', 'strong', 'opportunity']
        negative_words = ['decrease', 'decline', 'loss', 'failure', 'negative', 'reduced', 
                         'lower', 'risk', 'unfavorable', 'weak', 'challenge']
        
        sentiment_scores = []
        years_list = []
        
        for year, df in self.data.items():
            text_col = df.columns[-1]
            texts = df[text_col].astype(str).tolist()
            
            year_scores = []
            for text in texts:
                text_lower = text.lower()
                positive_count = sum(text_lower.count(word) for word in positive_words)
                negative_count = sum(text_lower.count(word) for word in negative_words)
                
                # 简单的情感分数计算
                if positive_count + negative_count > 0:
                    score = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    score = 0
                
                year_scores.append(score)
            
            sentiment_scores.append(year_scores)
            years_list.append(year)
        
        # 绘制情感分数箱线图
        plt.figure(figsize=(12, 6))
        plt.boxplot(sentiment_scores, labels=years_list)
        plt.title('财报情感分数随时间变化')
        plt.xlabel('年份')
        plt.ylabel('情感分数 (正面 > 0, 负面 < 0)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sentiment_over_time.png', dpi=300)
        plt.close()
        print("情感分析完成，已保存为 sentiment_over_time.png")
    
    def run_all_analyses(self, sample_size=1000):
        """运行所有分析"""
        print("开始全面分析财报数据...")
        self.load_sample_data(sample_size=sample_size)
        self.analyze_data_distribution()
        self.analyze_text_features()
        self.analyze_financial_metrics()
        self.analyze_sentiment_over_time()
        print("全部分析完成！")

if __name__ == "__main__":
    analyzer = FinancialReportAnalyzer()
    analyzer.run_all_analyses(sample_size=2000)
