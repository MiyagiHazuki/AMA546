import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import warnings
import random
warnings.filterwarnings('ignore')

# 确保NLTK资源已下载
def download_nltk_resources():
    """下载NLTK所需资源"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        print("NLTK resources already downloaded.")
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        print("NLTK resources downloaded successfully.")

# 下载NLTK资源
download_nltk_resources()

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
        
        # 获取英文停用词
        self.stop_words = set(stopwords.words('english'))
        # 添加标点符号和常见财报中的无意义词汇
        self.stop_words.update(string.punctuation)
        self.stop_words.update(['company', 'year', 'report', 'financial', 'million', 'billion', 
                               'quarter', 'fiscal', 'annual', 'period', 'ended', 'inc', 'corporation',
                               'corp', 'ltd', 'limited', 'group', 'plc', 'holdings', 'international'])
        
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
        print(f"Found financial report data for {len(self.years)} years: {', '.join(map(str, self.years))}")
    
    def load_data_samples(self, max_sample_size=2000, sample_ratio=0.15, min_sample_size=10, random_state=42):
        """从每个年份加载样本数据
        
        参数:
            max_sample_size (int): 每年最大样本数量
            sample_ratio (float): 抽样比例，占总数据量的百分比
            min_sample_size (int): 每年最小样本数量
            random_state (int): 随机种子
        """
        total_samples = 0
        
        for year, file_path in zip(self.years, self.data_files):
            print(f"Loading {year} data samples...")
            try:
                # 获取文件大小（MB）
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  File size: {file_size_mb:.2f} MB")
                
                # 尝试读取CSV文件的前几行来确定分隔符
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                
                # 确定分隔符
                if '\t' in first_line:
                    sep = '\t'
                elif ',' in first_line:
                    sep = ','
                else:
                    sep = None  # 让pandas自动检测
                
                # 估计文件总行数
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # 读取前1000行来估计平均行长度
                    sample_lines = [f.readline() for _ in range(1000) if f.readline()]
                    if sample_lines:
                        avg_line_length = sum(len(line) for line in sample_lines) / len(sample_lines)
                        # 估计总行数
                        estimated_lines = int((file_size_mb * 1024 * 1024) / avg_line_length)
                    else:
                        estimated_lines = 0
                
                # 计算动态样本大小
                dynamic_sample_size = min(max(int(estimated_lines * sample_ratio), min_sample_size), max_sample_size)
                print(f"  Estimated total lines: {estimated_lines}, Dynamic sample size: {dynamic_sample_size}")
                
                # 读取样本数据
                df = pd.read_csv(file_path, sep=sep, header=None, encoding='utf-8', 
                                 on_bad_lines='skip', nrows=dynamic_sample_size*2)
                
                # 随机抽样
                if len(df) > dynamic_sample_size:
                    df = df.sample(dynamic_sample_size, random_state=random_state)
                
                # 确保数据框至少有一列
                if df.shape[1] == 0:
                    print(f"  Warning: {year} data has no columns")
                    continue
                
                # 假设最后一列是文本内容
                self.data[year] = df
                total_samples += len(df)
                print(f"  Successfully loaded {len(df)} records with {df.shape[1]} columns")
            except Exception as e:
                print(f"  Error loading {year} data: {str(e)}")
        
        print(f"Total loaded samples across all years: {total_samples}")
    
    def preprocess_text(self, text):
        """预处理文本：转小写，去除标点，去除停用词"""
        if not isinstance(text, str):
            return ""
        
        # 转小写
        text = text.lower()
        
        # 去除非字母数字字符
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 去除停用词
        filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(filtered_tokens)
    
    def visualize_data_distribution(self):
        """可视化数据分布"""
        if not self.data:
            print("Please load data first")
            return
        
        print("Visualizing data distribution...")
        
        # 创建图表
        plt.figure(figsize=(16, 12))
        
        # 1. 每年数据量分布
        plt.subplot(2, 2, 1)
        years = list(self.data.keys())
        counts = [len(df) for df in self.data.values()]
        
        bars = plt.bar(years, counts, color=plt.cm.viridis(np.linspace(0, 1, len(years))))
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.title('Data Volume by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Reports', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 文本长度分布
        plt.subplot(2, 2, 2)
        all_lengths = []
        year_labels = []
        
        for year, df in self.data.items():
            # 假设文本在最后一列
            text_col = df.columns[-1]
            lengths = df[text_col].astype(str).apply(len)
            all_lengths.extend(lengths)
            year_labels.extend([year] * len(lengths))
        
        # 使用KDE图显示文本长度分布
        sns.histplot(all_lengths, bins=50, kde=True, color='skyblue')
        plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Text Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        
        # 添加统计信息
        avg_len = np.mean(all_lengths)
        median_len = np.median(all_lengths)
        plt.axvline(avg_len, color='red', linestyle='--', label=f'Mean: {avg_len:.1f}')
        plt.axvline(median_len, color='green', linestyle='--', label=f'Median: {median_len:.1f}')
        plt.legend()
        
        # 3. 每年平均文本长度
        plt.subplot(2, 2, 3)
        avg_lengths = []
        for year, df in self.data.items():
            text_col = df.columns[-1]
            avg_length = df[text_col].astype(str).apply(len).mean()
            avg_lengths.append(avg_length)
        
        bars = plt.bar(years, avg_lengths, color=plt.cm.cool(np.linspace(0, 1, len(years))))
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.title('Average Text Length by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Length (characters)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. 文本长度随时间变化的箱线图
        plt.subplot(2, 2, 4)
        length_data = []
        
        for year in years:
            df = self.data[year]
            text_col = df.columns[-1]
            lengths = df[text_col].astype(str).apply(len)
            length_data.append(lengths)
        
        # 绘制箱线图
        boxplot = plt.boxplot(length_data, labels=years, patch_artist=True)
        
        # 设置箱线图颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(years)))
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Text Length Distribution by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Text Length (characters)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/data_distribution.png', dpi=300)
        plt.close()
        print("Data distribution visualization completed, saved as data_distribution.png")
    
    def visualize_word_frequency(self, top_n=30):
        """可视化词频分布"""
        if not self.data:
            print("Please load data first")
            return
        
        print("Visualizing word frequency...")
        
        # 合并所有年份的文本
        all_texts = []
        for year, df in self.data.items():
            text_col = df.columns[-1]
            all_texts.extend(df[text_col].astype(str).tolist())
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in tqdm(all_texts, desc="Processing texts")]
        
        # 统计词频
        word_counts = Counter()
        for text in processed_texts:
            words = text.split()
            word_counts.update(words)
        
        # 获取最常见的词
        top_words = word_counts.most_common(top_n)
        
        # 绘制词频图
        plt.figure(figsize=(12, 8))
        words, counts = zip(*top_words)
        
        # 水平条形图
        bars = plt.barh(range(len(words)), counts, align='center', color=plt.cm.viridis(np.linspace(0, 1, len(words))))
        plt.yticks(range(len(words)), words)
        
        # 添加数据标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 5, bar.get_y() + bar.get_height()/2, f'{width}', 
                    ha='left', va='center', fontsize=9)
        
        plt.title('Most Frequent Words in Financial Reports', fontsize=16, fontweight='bold')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/word_frequency.png', dpi=300)
        plt.close()
        
        # 生成词云
        print("Generating word cloud...")
        wordcloud = WordCloud(width=1000, height=600, 
                             background_color='white', 
                             max_words=100, 
                             contour_width=3, 
                             contour_color='steelblue')
        
        wordcloud.generate_from_frequencies(dict(word_counts))
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Financial Reports Word Cloud', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/wordcloud.png', dpi=300)
        plt.close()
        
        print("Word frequency analysis completed, saved as word_frequency.png and wordcloud.png")
    
    def visualize_yearly_word_trends(self, keywords=None):
        """可视化关键词随年份的变化趋势"""
        if not self.data:
            print("Please load data first")
            return
        
        if keywords is None:
            # 默认关键词列表 (英文)
            keywords = ['profit', 'revenue', 'growth', 'decline', 'risk', 'innovation', 
                       'investment', 'debt', 'shareholder', 'strategy']
        
        print(f"Analyzing keyword trends: {', '.join(keywords)}...")
        
        # 初始化结果字典
        trend_data = {keyword: [] for keyword in keywords}
        years = sorted(self.data.keys())
        
        # 统计每年关键词出现频率
        for year in years:
            df = self.data[year]
            text_col = df.columns[-1]
            
            # 预处理所有文本
            processed_texts = [self.preprocess_text(text) for text in df[text_col].astype(str)]
            all_text = ' '.join(processed_texts)
            
            for keyword in keywords:
                # 计算关键词出现次数 (使用正则表达式匹配整个单词)
                pattern = r'\b' + keyword + r'\b'
                count = len(re.findall(pattern, all_text))
                
                # 归一化处理（每千字）
                total_words = len(all_text.split())
                normalized_count = count * 1000 / (total_words if total_words > 0 else 1)
                trend_data[keyword].append(normalized_count)
        
        # 绘制趋势图
        plt.figure(figsize=(14, 8))
        
        for keyword, counts in trend_data.items():
            plt.plot(years, counts, marker='o', linewidth=2, label=keyword)
        
        plt.title('Keyword Usage Frequency Trends (per 1000 words)', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Frequency (per 1000 words)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(years, rotation=45)
        
        # 添加数据标签
        for keyword, counts in trend_data.items():
            for i, count in enumerate(counts):
                plt.text(years[i], count, f'{count:.2f}', 
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/keyword_trends.png', dpi=300)
        plt.close()
        print("Keyword trend analysis completed, saved as keyword_trends.png")
    
    def visualize_sentiment_analysis(self):
        """可视化情感分析结果"""
        if not self.data:
            print("Please load data first")
            return
        
        print("Performing sentiment analysis...")
        
        # 英文情感词典
        positive_words = ['increase', 'growth', 'profit', 'success', 'positive', 'improved', 
                         'higher', 'gain', 'favorable', 'strong', 'opportunity', 'advantage',
                         'innovation', 'achieve', 'exceed', 'benefit', 'progress', 'enhance']
        
        negative_words = ['decrease', 'decline', 'loss', 'failure', 'negative', 'reduced', 
                         'lower', 'risk', 'unfavorable', 'weak', 'challenge', 'difficult',
                         'problem', 'concern', 'uncertainty', 'adverse', 'deteriorate', 'deficit']
        
        sentiment_scores = []
        years_list = sorted(self.data.keys())
        
        for year in years_list:
            df = self.data[year]
            text_col = df.columns[-1]
            
            # 预处理文本
            processed_texts = [self.preprocess_text(text) for text in df[text_col].astype(str)]
            
            year_scores = []
            for text in processed_texts:
                words = text.split()
                
                # 计算情感词出现次数
                positive_count = sum(1 for word in words if word in positive_words)
                negative_count = sum(1 for word in words if word in negative_words)
                
                # 简单的情感分数计算
                if positive_count + negative_count > 0:
                    score = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    score = 0
                
                year_scores.append(score)
            
            sentiment_scores.append(year_scores)
        
        # 计算每年的平均情感分数
        avg_scores = [np.mean(scores) for scores in sentiment_scores]
        
        # 绘制情感分数趋势图
        plt.figure(figsize=(14, 10))
        
        # 1. 情感分数箱线图
        plt.subplot(2, 1, 1)
        boxplot = plt.boxplot(sentiment_scores, labels=years_list, patch_artist=True)
        
        # 设置箱线图颜色
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(years_list)))
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Sentiment Score Distribution in Financial Reports', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Sentiment Score (Negative < 0 < Positive)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. 平均情感分数趋势
        plt.subplot(2, 1, 2)
        plt.plot(years_list, avg_scores, marker='o', linewidth=2, color='blue')
        
        # 添加数据标签
        for i, score in enumerate(avg_scores):
            plt.text(years_list[i], score, f'{score:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Average Sentiment Score Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_analysis.png', dpi=300)
        plt.close()
        print("Sentiment analysis completed, saved as sentiment_analysis.png")
    
    def visualize_topic_distribution(self, n_topics=5):
        """使用TF-IDF和降维技术可视化主题分布"""
        if not self.data:
            print("Please load data first")
            return
        
        print("Analyzing topic distribution...")
        
        # 合并所有文本并进行预处理
        all_texts = []
        year_labels = []
        
        for year, df in self.data.items():
            text_col = df.columns[-1]
            texts = df[text_col].astype(str).tolist()
            
            # 预处理文本
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            all_texts.extend(processed_texts)
            year_labels.extend([year] * len(processed_texts))
        
        # 使用TF-IDF向量化文本
        print("Performing TF-IDF vectorization...")
        vectorizer = TfidfVectorizer(max_features=1000, min_df=5)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 使用降维技术进行可视化
        print("Performing dimensionality reduction...")
        svd = TruncatedSVD(n_components=2)
        reduced_data = svd.fit_transform(tfidf_matrix)
        
        # 绘制散点图
        plt.figure(figsize=(14, 10))
        
        # 为每年使用不同颜色
        years = sorted(set(year_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
        
        for i, year in enumerate(years):
            mask = [y == year for y in year_labels]
            plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                       c=[colors[i]], label=str(year), alpha=0.7)
        
        plt.title('Topic Distribution in Financial Reports (TF-IDF + SVD)', fontsize=16, fontweight='bold')
        plt.xlabel('Topic Dimension 1', fontsize=12)
        plt.ylabel('Topic Dimension 2', fontsize=12)
        plt.legend(title='Year')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/topic_distribution.png', dpi=300)
        plt.close()
        print("Topic distribution analysis completed, saved as topic_distribution.png")
    
    def visualize_lexical_diversity(self):
        """可视化词汇多样性随时间的变化"""
        if not self.data:
            print("Please load data first")
            return
        
        print("Analyzing lexical diversity...")
        
        years = sorted(self.data.keys())
        avg_diversity = []
        diversity_data = []
        
        for year in years:
            df = self.data[year]
            text_col = df.columns[-1]
            
            # 预处理文本
            processed_texts = [self.preprocess_text(text) for text in df[text_col].astype(str)]
            
            # 计算每个文本的词汇多样性 (不同词汇数/总词汇数)
            text_diversity = []
            for text in processed_texts:
                words = text.split()
                if len(words) > 0:
                    diversity = len(set(words)) / len(words)
                    text_diversity.append(diversity)
            
            if text_diversity:
                avg_diversity.append(np.mean(text_diversity))
                diversity_data.append(text_diversity)
            else:
                avg_diversity.append(0)
                diversity_data.append([0])
        
        # 绘制词汇多样性图表
        plt.figure(figsize=(14, 10))
        
        # 1. 词汇多样性箱线图
        plt.subplot(2, 1, 1)
        boxplot = plt.boxplot(diversity_data, labels=years, patch_artist=True)
        
        # 设置箱线图颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Lexical Diversity Distribution by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Lexical Diversity (Unique Words / Total Words)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # 2. 平均词汇多样性趋势
        plt.subplot(2, 1, 2)
        plt.plot(years, avg_diversity, marker='o', linewidth=2, color='blue')
        
        # 添加数据标签
        for i, diversity in enumerate(avg_diversity):
            plt.text(years[i], diversity, f'{diversity:.3f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Average Lexical Diversity Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Lexical Diversity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/lexical_diversity.png', dpi=300)
        plt.close()
        print("Lexical diversity analysis completed, saved as lexical_diversity.png")
    
    def run_all_visualizations(self, max_sample_size=2000, sample_ratio=0.1, min_sample_size=500):
        """运行所有可视化分析
        
        参数:
            max_sample_size (int): 每年最大样本数量
            sample_ratio (float): 抽样比例，占总数据量的百分比
            min_sample_size (int): 每年最小样本数量
        """
        print("Starting comprehensive visualization of financial report data...")
        self.load_data_samples(max_sample_size=max_sample_size, sample_ratio=sample_ratio, min_sample_size=min_sample_size)
        self.visualize_data_distribution()
        self.visualize_word_frequency()
        self.visualize_yearly_word_trends()
        self.visualize_sentiment_analysis()
        self.visualize_topic_distribution()
        self.visualize_lexical_diversity()
        print(f"All visualizations completed! Results saved in {self.output_dir} directory")

if __name__ == "__main__":
    random.seed(42)
    visualizer = FinancialReportVisualizer()
    # 使用动态抽样策略
    visualizer.run_all_visualizations(max_sample_size=4000, sample_ratio=0.25, min_sample_size=10) 