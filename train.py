import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import xgboost as xgb_lib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
from scipy import stats
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from scipy import sparse

# 下载NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# 导入BERT模型和数据集
#from pretrain import BERTRegressor, FinancialReportDataset, set_seed, load_and_preprocess_data
from pretrain_without_logv12 import BERTRegressor, FinancialReportDataset, set_seed, load_and_preprocess_data
from transformers import AutoTokenizer

# 文本预处理函数（用于传统机器学习方法）
def preprocess_text_for_traditional_ml(text_list):
    """
    对文本进行预处理，包括停用词过滤和词干提取
    
    参数:
    - text_list: 文本列表
    
    返回:
    - 预处理后的文本列表
    """
    print("对传统机器学习方法进行文本预处理（停用词过滤和词干提取）...")
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    processed_texts = []
    
    for text in tqdm(text_list, desc='预处理文本'):
        if not isinstance(text, str) or pd.isna(text):
            processed_texts.append('')
            continue
            
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 停用词过滤和词干提取
        filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        
        # 重新组合为文本
        processed_text = ' '.join(filtered_tokens)
        processed_texts.append(processed_text)
    
    return processed_texts

# 保存预处理后的特征数据
def save_processed_features(feature_matrices, feature_extractors, train_texts, test_texts, train_df, test_df, output_dir='data_processed'):
    """
    保存预处理后的特征数据
    
    参数:
    - feature_matrices: 特征矩阵字典
    - feature_extractors: 特征提取器字典
    - train_texts: 训练集预处理后的文本
    - test_texts: 测试集预处理后的文本
    - train_df: 训练集DataFrame
    - test_df: 测试集DataFrame
    - output_dir: 输出目录
    """
    print(f"\n保存预处理后的特征数据到 {output_dir} 目录...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预处理后的文本
    with open(os.path.join(output_dir, 'train_texts_processed.pkl'), 'wb') as f:
        pickle.dump(train_texts, f)
    
    with open(os.path.join(output_dir, 'test_texts_processed.pkl'), 'wb') as f:
        pickle.dump(test_texts, f)
    
    # 保存特征提取器
    for feature_type, vectorizer in feature_extractors.items():
        with open(os.path.join(output_dir, f'{feature_type}_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
    
    # 保存特征矩阵
    for feature_type, (X_train, X_test) in feature_matrices.items():
        # 保存稀疏矩阵
        sparse.save_npz(os.path.join(output_dir, f'{feature_type}_train.npz'), X_train)
        sparse.save_npz(os.path.join(output_dir, f'{feature_type}_test.npz'), X_test)
    
    # 保存带有预处理文本的DataFrame
    train_df_with_processed = train_df.copy()
    test_df_with_processed = test_df.copy()
    
    train_df_with_processed['processed_text'] = train_texts
    test_df_with_processed['processed_text'] = test_texts
    
    train_df_with_processed.to_csv(os.path.join(output_dir, 'train_df_with_processed.csv'), index=False)
    test_df_with_processed.to_csv(os.path.join(output_dir, 'test_df_with_processed.csv'), index=False)
    
    print(f"预处理后的特征数据已保存到 {output_dir} 目录")

# 从BERT模型中提取特征
def extract_bert_features(model, data_loader, device):
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='提取BERT特征'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 获取BERT的输出
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 使用[CLS]标记的输出作为文本表示
            # 注意：不同模型可能有不同的输出结构
            if hasattr(outputs, 'pooler_output'):
                pooled_output = outputs.pooler_output
            else:
                # 对于某些模型（如Longformer），可能需要使用last_hidden_state的第一个token
                pooled_output = outputs.last_hidden_state[:, 0, :]
            
            # 将特征移到CPU并转换为NumPy数组
            features = pooled_output.cpu().numpy()
            
            all_features.append(features)
    
    # 合并所有批次的特征
    all_features = np.vstack(all_features)
    
    return all_features

# 提取不同类型的文本特征
def extract_text_features(train_texts, test_texts, feature_type='tf', max_features=10000):
    """
    提取不同类型的文本特征
    
    参数:
    - train_texts: 训练集文本
    - test_texts: 测试集文本
    - feature_type: 特征类型，可选值：
        - 'tf': log(1+TF)
        - 'tfidf': log(1+TF)-IDF
        - 'bigram_tf': Bigram + log(1+TF)
        - 'bigram_tfidf': Bigram + log(1+TF)-IDF
    - max_features: 最大特征数量
    
    返回:
    - X_train: 训练集特征矩阵 (稀疏格式)
    - X_test: 测试集特征矩阵 (稀疏格式)
    - vectorizer: 特征提取器
    """
    print(f"提取 {feature_type} 特征...")
    
    if feature_type == 'tf':
        # 1. log(1+TF)
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 1)  # 只使用单词
        )
        X_train = vectorizer.fit_transform(train_texts)
        # 应用log(1+TF)变换，保持稀疏格式
        X_train = X_train.log1p()
        
        X_test = vectorizer.transform(test_texts)
        X_test = X_test.log1p()
        
    elif feature_type == 'tfidf':
        # 2. log(1+TF)-IDF
        vectorizer = TfidfVectorizer(
            use_idf=True,
            max_features=max_features,
            ngram_range=(1, 1),  # 只使用单词
            sublinear_tf=True  # 使用log(1+tf)
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        
    elif feature_type == 'bigram_tf':
        # 3. Bigram + log(1+TF)
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)  # 使用单词和二元组
        )
        X_train = vectorizer.fit_transform(train_texts)
        # 应用log(1+TF)变换，保持稀疏格式
        X_train = X_train.log1p()
        
        X_test = vectorizer.transform(test_texts)
        X_test = X_test.log1p()
        
    elif feature_type == 'bigram_tfidf':
        # 4. Bigram + log(1+TF)-IDF
        vectorizer = TfidfVectorizer(
            use_idf=True,
            max_features=max_features,
            ngram_range=(1, 2),  # 使用单词和二元组
            sublinear_tf=True  # 使用log(1+tf)
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
    
    else:
        raise ValueError(f"不支持的特征类型: {feature_type}")
    
    print(f"{feature_type} 特征维度: {X_train.shape[1]}")
    return X_train, X_test, vectorizer

# 训练XGBoost模型
def train_xgboost(X_train, y_train, X_val=None, y_val=None, params=None, num_rounds=100):
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1,
            'lambda': 1,
            'seed': 42
        }
    
    # 创建DMatrix对象
    try:
        dtrain = xgb_lib.DMatrix(X_train, label=y_train)
    except Exception as e:
        print(f"创建DMatrix时出错: {str(e)}")
        print("尝试使用替代方法...")
        dtrain = xgb_lib.DMatrix(data=X_train, label=y_train)
    
    # 如果有验证集
    evals = []
    if X_val is not None and y_val is not None:
        try:
            dval = xgb_lib.DMatrix(X_val, label=y_val)
        except Exception as e:
            print(f"创建验证集DMatrix时出错: {str(e)}")
            print("尝试使用替代方法...")
            dval = xgb_lib.DMatrix(data=X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'val')]
    else:
        evals = [(dtrain, 'train')]
    
    # 训练模型
    print("训练XGBoost模型...")
    model = xgb_lib.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    return model

# 评估模型
def evaluate_model(model, X_test, y_test):
    try:
        dtest = xgb_lib.DMatrix(X_test)
    except Exception as e:
        print(f"创建测试集DMatrix时出错: {str(e)}")
        print("尝试使用替代方法...")
        dtest = xgb_lib.DMatrix(data=X_test)
    
    predictions = model.predict(dtest)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f'测试结果:')
    print(f'  MSE: {mse:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R²: {r2:.4f}')
    
    return mse, rmse, r2, predictions

# 绘制特征重要性
def plot_feature_importance(model, feature_names=None, output_file='feature_importance.png'):
    plt.figure(figsize=(12, 8))
    xgb_lib.plot_importance(model, max_num_features=20, height=0.8, importance_type='gain')
    plt.title('Feature Importance of XGBoost')
    plt.tight_layout()
    plt.savefig(output_file)

# 绘制评估结果
def plot_evaluation_results(predictions, targets, model_name='model'):
    # 创建一个包含多个子图的大图
    plt.figure(figsize=(15, 12))
    
    # 1. 预测值与真实值的散点图
    plt.subplot(2, 2, 1)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Predicted Value vs True Value')
    
    # 2. 残差图（预测误差）
    plt.subplot(2, 2, 2)
    residuals = np.array(predictions) - np.array(targets)
    plt.scatter(targets, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Value')
    plt.ylabel('Residual (Predicted Value - True Value)')
    plt.title('Residual Plot')
    
    # 3. 误差分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Error')
    
    # 4. 预测值与残差的关系图
    plt.subplot(2, 2, 4)
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Relationship between Predicted Value and Residual')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_regression_results.png')
    
    # 额外创建一个图表：真实值和预测值的时间序列对比
    if len(targets) > 100:
        # 如果样本太多，只显示前100个样本
        sample_size = 100
    else:
        sample_size = len(targets)
    
    plt.figure(figsize=(12, 6))
    indices = range(sample_size)
    plt.plot(indices, targets[:sample_size], 'b-', label='True Value', alpha=0.7)
    plt.plot(indices, predictions[:sample_size], 'r-', label='Predicted Value', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Comparison of True Value and Predicted Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_name}_prediction_comparison.png')
    
    # 创建Q-Q图（量化-量化图）检查误差的正态性
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q plot: Check the normality of the error')
    plt.savefig(f'{model_name}_qq_plot.png')

# 训练ElasticNet模型
def train_elasticnet(X_train, y_train, params=None):
    if params is None:
        params = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'max_iter': 1000
        }
    
    print("训练ElasticNet模型...")
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    
    # 如果X_train是稀疏矩阵，使用稀疏矩阵兼容的标准化器
    if hasattr(X_train, 'toarray') or sparse.issparse(X_train):
        # 对于稀疏矩阵，我们只标准化非零元素
        # 不使用均值中心化，因为这会破坏稀疏性
        try:
            scaler = StandardScaler(with_mean=False)  # 对于稀疏矩阵，不使用均值中心化
            X_train_scaled = scaler.fit_transform(X_train)
        except Exception as e:
            print(f"标准化稀疏矩阵时出错: {str(e)}")
            print("跳过标准化步骤...")
            X_train_scaled = X_train
            scaler = None
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    
    # 训练模型
    model = ElasticNet(**params)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# 评估带有标准化的模型
def evaluate_scaled_model(model, scaler, X_test, y_test):
    # 标准化测试数据
    if scaler is not None:
        if hasattr(X_test, 'toarray') or sparse.issparse(X_test):
            try:
                X_test_scaled = scaler.transform(X_test)
            except Exception as e:
                print(f"标准化测试集时出错: {str(e)}")
                print("跳过标准化步骤...")
                X_test_scaled = X_test
        else:
            X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # 预测
    predictions = model.predict(X_test_scaled)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f'测试结果:')
    print(f'  MSE: {mse:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R²: {r2:.4f}')
    
    return mse, rmse, r2, predictions

def main():
    parser = argparse.ArgumentParser(description='特征组合评估 - 金融文本回归任务')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--bert_model', type=str, default='allenai/longformer-base-4096', help='BERT预训练模型名称')
    parser.add_argument('--bert_model_path', type=str, default='longformer_pretrained_model_without_logv12.pt', help='BERT模型路径')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_rounds', type=int, default=100, help='XGBoost训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--force_extract', action='store_true', help='强制重新提取特征')
    parser.add_argument('--begin_year', type=int, default=1996, help='开始年份')
    parser.add_argument('--test_year', type=int, default=2001, help='测试年份')
    parser.add_argument('--max_features', type=int, default=10000, help='最大特征数量')
    parser.add_argument('--skip_preprocessing', action='store_true', help='跳过传统机器学习的文本预处理')
    parser.add_argument('--processed_data_dir', type=str, default='data_processed', help='预处理数据保存目录')
    parser.add_argument('--load_processed', action='store_true', help='加载已预处理的特征数据')
    parser.add_argument('--load_processed_text', action='store_true', help='加载已预处理的文本数据（停用词过滤、词干提取后）')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载和预处理数据
    print("加载和预处理数据...")
    train_df, test_df, scaler = load_and_preprocess_data(args.data_dir, args.begin_year, args.test_year)
    print(f'训练数据大小: {len(train_df)}')
    print(f'测试数据大小: {len(test_df)}')
    
    # 准备文本数据用于特征提取
    if 'tok' in train_df.columns:
        train_texts_original = train_df['tok'].fillna('').astype(str).tolist()
        test_texts_original = test_df['tok'].fillna('').astype(str).tolist()
    elif 'cleaned_text' in train_df.columns:
        train_texts_original = train_df['cleaned_text'].fillna('').astype(str).tolist()
        test_texts_original = test_df['cleaned_text'].fillna('').astype(str).tolist()
    else:
        raise KeyError("数据中既没有'tok'列也没有'cleaned_text'列")
    
    # 检查是否加载已预处理的特征数据
    if args.load_processed and os.path.exists(args.processed_data_dir):
        print(f"加载已预处理的特征数据从 {args.processed_data_dir}...")
        
        # 加载预处理后的文本
        processed_text_path = os.path.join(args.processed_data_dir, 'train_texts_processed.pkl')
        if os.path.exists(processed_text_path):
            with open(processed_text_path, 'rb') as f:
                train_texts_for_ml = pickle.load(f)
            
            with open(os.path.join(args.processed_data_dir, 'test_texts_processed.pkl'), 'rb') as f:
                test_texts_for_ml = pickle.load(f)
            
            print("已加载预处理后的文本数据")
        else:
            print("未找到预处理后的文本数据，将使用原始文本")
            train_texts_for_ml = train_texts_original
            test_texts_for_ml = test_texts_original
        
        # 加载特征提取器和特征矩阵
        feature_extractors = {}
        feature_matrices = {}
        
        # 定义要加载的特征类型
        feature_types = [
            'tf',           # 1. log(1+TF)
            'tfidf',        # 2. log(1+TF)-IDF
            'bigram_tf',    # 3. Bigram + log(1+TF)
            'bigram_tfidf'  # 4. Bigram + log(1+TF)-IDF
        ]
        
        # 加载特征提取器
        for feature_type in feature_types:
            vectorizer_path = os.path.join(args.processed_data_dir, f'{feature_type}_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    feature_extractors[feature_type] = pickle.load(f)
            
            # 加载特征矩阵
            train_path = os.path.join(args.processed_data_dir, f'{feature_type}_train.npz')
            test_path = os.path.join(args.processed_data_dir, f'{feature_type}_test.npz')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                X_train = sparse.load_npz(train_path)
                X_test = sparse.load_npz(test_path)
                feature_matrices[feature_type] = (X_train, X_test)
        
        print(f"已加载 {len(feature_matrices)} 种特征类型的预处理数据")
    # 只加载预处理后的文本数据，但重新提取特征
    elif args.load_processed_text and os.path.exists(args.processed_data_dir):
        print(f"加载已预处理的文本数据从 {args.processed_data_dir}...")
        
        # 加载预处理后的文本
        processed_text_path = os.path.join(args.processed_data_dir, 'train_texts_processed.pkl')
        if os.path.exists(processed_text_path):
            with open(processed_text_path, 'rb') as f:
                train_texts_for_ml = pickle.load(f)
            
            with open(os.path.join(args.processed_data_dir, 'test_texts_processed.pkl'), 'rb') as f:
                test_texts_for_ml = pickle.load(f)
            
            print("已加载预处理后的文本数据")
        else:
            print("未找到预处理后的文本数据，将进行文本预处理")
            if not args.skip_preprocessing:
                train_texts_for_ml = preprocess_text_for_traditional_ml(train_texts_original)
                test_texts_for_ml = preprocess_text_for_traditional_ml(test_texts_original)
            else:
                print("跳过传统机器学习的文本预处理...")
                train_texts_for_ml = train_texts_original
                test_texts_for_ml = test_texts_original
        
        # 提取不同类型的文本特征 - 使用预处理后的文本
        feature_types = [
            'tf',           # 1. log(1+TF)
            'tfidf',        # 2. log(1+TF)-IDF
            'bigram_tf',    # 3. Bigram + log(1+TF)
            'bigram_tfidf'  # 4. Bigram + log(1+TF)-IDF
        ]
        
        # 存储所有特征提取器和特征矩阵
        feature_extractors = {}
        feature_matrices = {}
        
        # 提取所有类型的文本特征 - 使用预处理后的文本
        for feature_type in feature_types:
            X_train, X_test, vectorizer = extract_text_features(
                train_texts_for_ml, 
                test_texts_for_ml, 
                feature_type=feature_type, 
                max_features=args.max_features
            )
            feature_extractors[feature_type] = vectorizer
            feature_matrices[feature_type] = (X_train, X_test)
        
        # 保存特征提取器和特征矩阵
        print(f"\n保存特征提取器和特征矩阵到 {args.processed_data_dir}...")
        os.makedirs(args.processed_data_dir, exist_ok=True)
        
        for feature_type, vectorizer in feature_extractors.items():
            with open(os.path.join(args.processed_data_dir, f'{feature_type}_vectorizer.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)
        
        for feature_type, (X_train, X_test) in feature_matrices.items():
            sparse.save_npz(os.path.join(args.processed_data_dir, f'{feature_type}_train.npz'), X_train)
            sparse.save_npz(os.path.join(args.processed_data_dir, f'{feature_type}_test.npz'), X_test)
    else:
        # 对传统机器学习方法进行文本预处理
        if not args.skip_preprocessing:
            # 对传统机器学习方法的文本进行预处理
            train_texts_for_ml = preprocess_text_for_traditional_ml(train_texts_original)
            test_texts_for_ml = preprocess_text_for_traditional_ml(test_texts_original)
        else:
            print("跳过传统机器学习的文本预处理...")
            train_texts_for_ml = train_texts_original
            test_texts_for_ml = test_texts_original
        
        # 提取不同类型的文本特征 - 使用预处理后的文本
        feature_types = [
            'tf',           # 1. log(1+TF)
            'tfidf',        # 2. log(1+TF)-IDF
            'bigram_tf',    # 3. Bigram + log(1+TF)
            'bigram_tfidf'  # 4. Bigram + log(1+TF)-IDF
        ]
        
        # 存储所有特征提取器和特征矩阵
        feature_extractors = {}
        feature_matrices = {}
        
        # 提取所有类型的文本特征 - 使用预处理后的文本
        for feature_type in feature_types:
            X_train, X_test, vectorizer = extract_text_features(
                train_texts_for_ml, 
                test_texts_for_ml, 
                feature_type=feature_type, 
                max_features=args.max_features
            )
            feature_extractors[feature_type] = vectorizer
            feature_matrices[feature_type] = (X_train, X_test)
        
        # 保存预处理后的特征数据
        save_processed_features(
            feature_matrices, 
            feature_extractors, 
            train_texts_for_ml, 
            test_texts_for_ml, 
            train_df, 
            test_df, 
            output_dir=args.processed_data_dir
        )
    
    # 准备目标变量
    y_train = train_df['logvol+12'].values
    y_test = test_df['logvol+12'].values
    
    # 准备logvol-12特征
    logvol_minus_12_train = train_df['logvol-12'].values.reshape(-1, 1)
    logvol_minus_12_test = test_df['logvol-12'].values.reshape(-1, 1)
    
    # 提取BERT特征 - 使用原始文本，不进行额外预处理
    print('\n提取BERT特征...')
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    # 创建数据集 - 使用原始文本
    train_dataset = FinancialReportDataset(train_df, tokenizer, args.max_length)
    test_dataset = FinancialReportDataset(test_df, tokenizer, args.max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 检查BERT模型是否存在
    if not os.path.isfile(args.bert_model_path):
        print(f'错误: BERT模型文件 {args.bert_model_path} 不存在')
        print('请先运行pretrain_without_logv12.py训练BERT模型')
        return
    
    # 加载BERT模型
    print(f'加载BERT模型: {args.bert_model_path}')
    bert_model = BERTRegressor(args.bert_model)
    
    try:
        bert_model.load_state_dict(torch.load(args.bert_model_path, map_location=device))
    except RuntimeError as e:
        print(f"标准加载失败，错误: {str(e)}")
        print("尝试使用strict=False加载模型...")
        bert_model.load_state_dict(torch.load(args.bert_model_path, map_location=device), strict=False)
        print("模型已加载，但某些权重可能未匹配")
    
    bert_model.to(device)
    print('BERT模型加载成功')
    
    # 提取BERT特征
    bert_train_features_file = f'bert_train_features_{args.begin_year}_{args.test_year-1}.npz'
    bert_test_features_file = f'bert_test_features_{args.test_year}.npz'
    
    if not args.force_extract and os.path.exists(bert_train_features_file) and os.path.exists(bert_test_features_file):
        print('加载已保存的BERT特征...')
        bert_train_features_data = np.load(bert_train_features_file)
        bert_test_features_data = np.load(bert_test_features_file)
        bert_train_features = bert_train_features_data['features']
        bert_test_features = bert_test_features_data['features']
    else:
        print('提取新的BERT特征...')
        bert_train_features = extract_bert_features(bert_model, train_loader, device)
        bert_test_features = extract_bert_features(bert_model, test_loader, device)
        
        # 保存特征
        np.savez(bert_train_features_file, features=bert_train_features)
        np.savez(bert_test_features_file, features=bert_test_features)
    
    print(f"BERT特征维度: {bert_train_features.shape[1]}")
    
    # 创建第5种特征组合: Longformer-Embedding + Bigram
    print("\n创建Longformer-Embedding + Bigram特征组合...")
    # 修复解包错误
    bigram_train, bigram_test = feature_matrices['bigram_tf']
    
    # 使用新的函数创建组合特征
    # 将密集特征转换为CSR稀疏矩阵
    bert_train_sparse = sparse.csr_matrix(bert_train_features)
    bert_test_sparse = sparse.csr_matrix(bert_test_features)
    # 水平连接两个稀疏矩阵
    longformer_bigram_train = sparse.hstack([bert_train_sparse, bigram_train])
    longformer_bigram_test = sparse.hstack([bert_test_sparse, bigram_test])
    
    feature_matrices['longformer_bigram'] = (longformer_bigram_train, longformer_bigram_test)
    print(f"Longformer-Embedding + Bigram特征维度: {longformer_bigram_train.shape[1]}")
    
    # 添加新的特征组合
    
    # 6. 单独使用longformer回归logv+12
    print("\n创建单独使用Longformer特征组合...")
    # 将密集特征转换为稀疏格式以保持一致性
    feature_matrices['longformer_only'] = (bert_train_sparse, bert_test_sparse)
    print(f"Longformer特征维度: {bert_train_features.shape[1]}")
    
    # 7. 单独使用logv-12回归logv+12
    print("\n创建单独使用logv-12特征组合...")
    # 将logv-12转换为稀疏格式以保持一致性
    logv_minus_12_train_sparse = sparse.csr_matrix(logvol_minus_12_train)
    logv_minus_12_test_sparse = sparse.csr_matrix(logvol_minus_12_test)
    feature_matrices['logv_minus_12_only'] = (logv_minus_12_train_sparse, logv_minus_12_test_sparse)
    print(f"logv-12特征维度: {logvol_minus_12_train.shape[1]}")
    
    # 8-12. 原有5个特征组合各自加入logv-12特征
    for feature_type in list(feature_matrices.keys()):
        # 跳过新添加的特征组合
        if feature_type in ['longformer_only', 'logv_minus_12_only']:
            continue
            
        # 获取原始特征
        X_train, X_test = feature_matrices[feature_type]
        
        # 创建新的特征组合名称
        new_feature_type = f"{feature_type}_with_logv12"
        
        print(f"\n创建{new_feature_type}特征组合...")
        
        # 将logv-12转换为稀疏格式
        logv_minus_12_train_sparse = sparse.csr_matrix(logvol_minus_12_train)
        logv_minus_12_test_sparse = sparse.csr_matrix(logvol_minus_12_test)
        
        # 合并特征和logv-12
        X_train_with_logv = sparse.hstack([X_train, logv_minus_12_train_sparse])
        X_test_with_logv = sparse.hstack([X_test, logv_minus_12_test_sparse])
        
        # 存储新的特征组合
        feature_matrices[new_feature_type] = (X_train_with_logv, X_test_with_logv)
        print(f"{new_feature_type}特征维度: {X_train_with_logv.shape[1]}")
    
    # 保存所有特征组合到预处理数据目录
    print(f"\n保存所有特征组合到 {args.processed_data_dir}...")
    for feature_type, (X_train, X_test) in feature_matrices.items():
        # 保存稀疏矩阵
        sparse.save_npz(os.path.join(args.processed_data_dir, f'{feature_type}_train.npz'), X_train)
        sparse.save_npz(os.path.join(args.processed_data_dir, f'{feature_type}_test.npz'), X_test)
    
    # 存储所有模型的评估结果
    results = []
    
    # 评估所有特征组合
    for feature_name, (X_train, X_test) in feature_matrices.items():
        print(f"\n\n{'='*50}")
        print(f"评估特征组合: {feature_name}")
        print(f"{'='*50}")
        
        # 训练XGBoost模型
        print(f"\n训练 {feature_name} + XGBoost 模型...")
        model = train_xgboost(X_train, y_train, num_rounds=args.num_rounds)
        
        # 评估模型
        print(f"\n评估 {feature_name} + XGBoost 模型...")
        mse, rmse, r2, predictions = evaluate_model(model, X_test, y_test)
        
        # 保存结果
        results.append({
            'feature_type': feature_name,
            'model_type': 'XGBoost',
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'feature_dim': X_train.shape[1]  # 添加特征维度信息
        })
        
        # 绘制评估结果
        plot_evaluation_results(predictions, y_test, model_name=f"{feature_name}_xgboost")
        
        # 如果是XGBoost模型，绘制特征重要性
        try:
            plot_feature_importance(model, output_file=f"{feature_name}_xgboost_importance.png")
        except Exception as e:
            print(f"绘制特征重要性时出错: {str(e)}")
        
        # 训练ElasticNet模型
        print(f"\n训练 {feature_name} + ElasticNet 模型...")
        elastic_model, elastic_scaler = train_elasticnet(X_train, y_train)
        
        # 评估ElasticNet模型
        print(f"\n评估 {feature_name} + ElasticNet 模型...")
        elastic_mse, elastic_rmse, elastic_r2, elastic_predictions = evaluate_scaled_model(
            elastic_model, elastic_scaler, X_test, y_test
        )
        
        # 保存结果
        results.append({
            'feature_type': feature_name,
            'model_type': 'ElasticNet',
            'mse': elastic_mse,
            'rmse': elastic_rmse,
            'r2': elastic_r2,
            'feature_dim': X_train.shape[1]  # 添加特征维度信息
        })
        
        # 绘制评估结果
        plot_evaluation_results(elastic_predictions, y_test, model_name=f"{feature_name}_elasticnet")
        
        # 保存最佳模型
        if feature_name in ['longformer_bigram', 'longformer_bigram_with_logv12']:
            with open(f"{feature_name}_xgboost_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            with open(f"{feature_name}_elasticnet_model.pkl", 'wb') as f:
                pickle.dump((elastic_model, elastic_scaler), f)
    
    # 将结果转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='mse')
    
    # 打印结果表格
    print("\n\n特征组合评估结果 (按MSE排序):")
    print("="*100)
    print(results_df.to_string(index=False))
    
    # 保存结果到CSV
    results_df.to_csv('feature_evaluation_results.csv', index=False)
    print("\n结果已保存到 feature_evaluation_results.csv")
    
    # 找出最佳模型
    best_model = results_df.iloc[0]
    print(f"\n最佳模型: {best_model['feature_type']} + {best_model['model_type']}")
    print(f"MSE: {best_model['mse']:.4f}, RMSE: {best_model['rmse']:.4f}, R²: {best_model['r2']:.4f}")
    
    # 按特征类型分组，比较不同特征组合的效果
    print("\n\n不同特征组合的效果比较:")
    print("="*100)
    
    # 创建特征类型分组
    feature_groups = {
        "基础特征": ["tf", "tfidf", "bigram_tf", "bigram_tfidf", "longformer_only", "logv_minus_12_only"],
        "组合特征": ["longformer_bigram"],
        "加入logv-12的特征": ["tf_with_logv12", "tfidf_with_logv12", "bigram_tf_with_logv12", 
                      "bigram_tfidf_with_logv12", "longformer_bigram_with_logv12"]
    }
    
    # 对每个分组，找出最佳模型
    for group_name, feature_types in feature_groups.items():
        group_results = results_df[results_df['feature_type'].isin(feature_types)]
        if not group_results.empty:
            best_in_group = group_results.iloc[0]
            print(f"\n{group_name}组中的最佳模型:")
            print(f"  特征类型: {best_in_group['feature_type']}")
            print(f"  模型类型: {best_in_group['model_type']}")
            print(f"  MSE: {best_in_group['mse']:.4f}, RMSE: {best_in_group['rmse']:.4f}, R²: {best_in_group['r2']:.4f}")
            print(f"  特征维度: {best_in_group['feature_dim']}")
    
    # 比较加入logv-12前后的效果变化
    print("\n\n加入logv-12前后的效果变化:")
    print("="*100)
    
    for base_feature in ["tf", "tfidf", "bigram_tf", "bigram_tfidf", "longformer_bigram"]:
        with_logv = f"{base_feature}_with_logv12"
        
        # 找出XGBoost模型的结果
        base_xgb = results_df[(results_df['feature_type'] == base_feature) & 
                             (results_df['model_type'] == 'XGBoost')]
        with_logv_xgb = results_df[(results_df['feature_type'] == with_logv) & 
                                  (results_df['model_type'] == 'XGBoost')]
        
        if not base_xgb.empty and not with_logv_xgb.empty:
            base_mse = base_xgb.iloc[0]['mse']
            with_logv_mse = with_logv_xgb.iloc[0]['mse']
            improvement = (base_mse - with_logv_mse) / base_mse * 100
            
            print(f"\n{base_feature} + XGBoost:")
            print(f"  原始MSE: {base_mse:.4f}")
            print(f"  加入logv-12后MSE: {with_logv_mse:.4f}")
            print(f"  改进: {improvement:.2f}%")

if __name__ == '__main__':
    main()
