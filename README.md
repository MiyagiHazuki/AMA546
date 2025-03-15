# 英文财报数据可视化工具

这个项目提供了一个全面的英文财报数据可视化工具，用于分析和展示多年财报数据的分布、特征和趋势。

## 功能特点

该工具提供以下可视化分析功能：

1. **数据分布可视化**：
   - 各年份数据量分布
   - 文本长度分布
   - 各年份平均文本长度
   - 各年份文本长度箱线图

2. **词频分析**：
   - 高频词汇条形图（已去除停用词）
   - 词云图

3. **关键词趋势分析**：
   - 关键财务词汇随时间的使用频率变化

4. **情感分析**：
   - 各年份财报情感分数分布
   - 平均情感分数随时间变化趋势

5. **主题分布分析**：
   - 使用TF-IDF和SVD降维技术可视化主题分布

6. **词汇多样性分析**：
   - 各年份词汇多样性分布
   - 平均词汇多样性随时间变化趋势

## 环境要求

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- wordcloud
- scikit-learn
- tqdm

## 安装依赖

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn tqdm
```

## 使用方法

1. 确保数据文件放在`data`目录下，文件名格式为`年份.csv`（如`1996.csv`、`2006.csv`等）

2. 运行可视化程序：

```bash
python financial_report_visualizer.py
```

3. 查看生成的可视化结果，所有图表将保存在`visualizations`目录下

## 智能动态抽样策略

该工具采用智能动态抽样策略，根据每年财报数据的总量自动调整抽样大小：

1. **基于文件大小估计**：程序会估计每个年份CSV文件的总行数
2. **比例抽样**：根据设定的抽样比例（默认15%）计算样本大小
3. **上下限控制**：设置最小样本量（默认500）和最大样本量（默认3000）
4. **平衡代表性**：确保每个年份的数据都有足够的代表性，同时避免某一年份的数据过度主导分析结果

这种动态抽样方法相比固定样本大小更加合理，能够更好地反映不同年份数据量的差异，同时保持计算效率。

## 文本预处理

该工具对英文财报文本进行了以下预处理：

1. 转换为小写
2. 去除非字母字符
3. 去除停用词（使用NLTK英文停用词列表）
4. 去除常见财报中的无意义词汇（如"company", "year", "report"等）
5. 去除短词（长度小于3的词）

## 自定义分析

您可以通过修改代码来自定义分析：

```python
# 自定义抽样参数
visualizer = FinancialReportVisualizer()
visualizer.run_all_visualizations(
    max_sample_size=5000,     # 每年最大样本数量
    sample_ratio=0.2,         # 抽样比例（占总数据量的百分比）
    min_sample_size=800       # 每年最小样本数量
)

# 单独运行特定分析
visualizer = FinancialReportVisualizer()
visualizer.load_data_samples(max_sample_size=3000, sample_ratio=0.15)
visualizer.visualize_word_frequency(top_n=50)  # 显示前50个高频词
visualizer.visualize_yearly_word_trends(keywords=['research', 'innovation', 'technology'])  # 自定义关键词
```

## 输出文件及其含义

程序运行后，将在`visualizations`目录下生成以下文件：

### 1. `data_distribution.png`：数据分布可视化
这个图表包含四个子图，展示了财报数据的基本分布特征：
- **各年份数据量分布**：显示每年收集的财报样本数量，帮助了解数据集在时间上的分布是否均衡。
- **文本长度分布**：展示所有财报文本长度的分布情况，包括平均长度和中位数长度，帮助了解财报文本的典型长度范围。
- **各年份平均文本长度**：显示每年财报的平均文本长度，可以观察财报详细程度是否随时间变化。
- **各年份文本长度箱线图**：通过箱线图展示每年文本长度的分布情况，包括中位数、四分位数和异常值，帮助识别财报长度的年度变化趋势和波动情况。

### 2. `word_frequency.png`：词频分析
展示财报中出现频率最高的词汇（已去除停用词），以水平条形图的形式呈现。这有助于识别财报中最常讨论的主题和关键概念，反映了财务报告的核心关注点。

### 3. `wordcloud.png`：词云图
将高频词以词云的形式直观展示，词的大小与其出现频率成正比。词云提供了财报文本内容的视觉摘要，让用户能够快速把握财报的主要内容和关键词。

### 4. `keyword_trends.png`：关键词趋势分析
追踪关键财务词汇（如"profit"、"revenue"、"growth"等）在不同年份中的使用频率变化。这个图表可以揭示财务报告关注点的历史变化，例如在经济衰退期可能会看到"risk"和"challenge"等词汇的使用频率增加。

### 5. `sentiment_analysis.png`：情感分析
包含两个子图：
- **情感分数分布箱线图**：展示每年财报的情感分数分布，正值表示积极情感，负值表示消极情感。
- **平均情感分数趋势**：显示每年财报的平均情感分数变化趋势。
这个分析可以揭示财报语调的变化，例如在经济繁荣时期可能会有更多积极的表述，而在经济下行时期则可能更加谨慎或消极。

### 6. `topic_distribution.png`：主题分布分析
使用TF-IDF和SVD降维技术将财报文本映射到二维空间，每个点代表一份财报，不同颜色代表不同年份。这个可视化可以帮助识别不同年份财报主题的聚类情况和变化趋势，例如某些年份的财报可能在主题上更加相似或分散。

### 7. `lexical_diversity.png`：词汇多样性分析
包含两个子图：
- **词汇多样性分布箱线图**：展示每年财报的词汇多样性分布（不同词汇数/总词汇数）。
- **平均词汇多样性趋势**：显示每年财报的平均词汇多样性变化趋势。
词汇多样性是衡量文本复杂性和丰富度的指标，较高的词汇多样性可能表明财报内容更加详细和复杂，而较低的词汇多样性可能表明财报使用了更多的标准化语言。

## 注意事项

- 程序使用动态抽样策略，根据每年数据文件大小自动调整样本量
- 首次运行时会自动下载NLTK资源（punkt和stopwords）
- 对于大型数据集，主题分布分析可能需要较长时间
- 文件大小估计和行数估计是近似值，可能与实际情况有所偏差

## 解读建议

在解读这些可视化结果时，建议关注以下几点：

1. **时间趋势**：观察各指标随时间的变化趋势，可能反映了财报写作风格、监管要求或经济环境的变化。

2. **异常点**：寻找数据中的异常值或突变点，这些可能与重大经济事件（如金融危机）或监管变化相关。

3. **相关性分析**：比较不同指标之间的关系，例如情感分数与特定关键词使用频率之间是否存在相关性。

4. **行业对比**：如果数据包含不同行业的财报，可以比较不同行业之间的差异，了解行业特定的报告特征。

## BERT回归模型

本项目使用BERT预训练模型对财务报告文本数据进行微调，构建一个回归模型，以预测`logvol+12`（未来12个月的对数交易量）。

### 模型架构

模型架构如下：
1. 使用BERT预训练模型对文本进行编码，获取文本表示
2. 将BERT的文本表示与`logvol-12`（过去12个月的对数交易量）特征连接
3. 通过多层感知机（MLP）输出`logvol+12`的预测值
4. 使用均方误差（MSE）作为损失函数进行微调

### 数据集

- 训练集：1996-1999年的财务报告数据
- 测试集：2000年的财务报告数据

### 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行BERT回归模型：
```bash
python bert.py --epochs 5 --batch_size 16 --learning_rate 2e-5
```

3. 如果已经训练过模型，下次运行时会自动加载已有模型，跳过训练阶段：
```bash
python bert.py  # 自动加载已有模型
```

4. 如果想强制重新训练模型，可以使用`--force_train`参数：
```bash
python bert.py --force_train
```

可选参数：
- `--data_dir`：数据目录，默认为`data`
- `--bert_model`：BERT预训练模型名称，默认为`bert-base-uncased`
- `--max_length`：最大序列长度，默认为512
- `--batch_size`：批次大小，默认为16
- `--epochs`：训练轮数，默认为5
- `--learning_rate`：学习率，默认为2e-5
- `--val_size`：验证集比例，默认为0.1
- `--seed`：随机种子，默认为42
- `--force_train`：强制重新训练模型，即使本地已存在模型
- `--model_path`：模型保存/加载路径，默认为`best_bert_regressor.pt`

### 结果评估

模型训练完成后，会生成以下结果：
1. 保存最佳模型为`best_bert_regressor.pt`
2. 生成多种可视化图表用于评估模型性能：
   - `bert_regression_results.png`：包含训练损失曲线、预测值与真实值散点图、残差图和误差分布直方图
   - `prediction_comparison.png`：真实值与预测值的时间序列对比图
   - `qq_plot.png`：Q-Q图，用于检查误差的正态性
   - `pred_vs_residuals.png`：预测值与残差关系图
3. 输出测试集上的评估指标：MSE、RMSE和R²

### 可视化图表解释

1. **训练和验证损失曲线**：展示模型在训练过程中的学习情况，帮助识别过拟合或欠拟合问题。

2. **预测值与真实值散点图**：直观展示预测值与真实值的关系，理想情况下点应该分布在对角线附近。

3. **残差图**：显示预测误差（残差）与真实值的关系，用于检查模型是否存在系统性偏差。理想情况下，残差应随机分布在零线附近。

4. **误差分布直方图**：展示预测误差的分布情况，理想情况下应呈正态分布，中心在零附近。

5. **真实值与预测值对比图**：以时间序列形式展示真实值和预测值的对比，直观显示模型的预测准确性。

6. **Q-Q图**：用于检验误差是否服从正态分布，如果点大致落在直线上，表明误差接近正态分布。

7. **预测值与残差关系图**：检查预测值与残差之间是否存在模式，理想情况下应随机分布，无明显模式。

## BERT+XGBoost模型

本项目还实现了一个结合BERT和XGBoost的混合模型，用于预测`logvol+12`（未来12个月的对数交易量）。该模型利用BERT提取文本特征，然后使用XGBoost进行回归预测。

### 模型架构

模型架构如下：
1. 使用预训练的BERT模型提取文本特征（移除最后的MLP层）
2. 将BERT提取的特征与`logvol-12`（过去12个月的对数交易量）特征连接
3. 使用XGBoost回归模型预测`logvol+12`

### 工作流程

1. **BERT预训练**：首先运行`bert.py`对BERT模型进行预训练，使其适应财务文本数据
2. **特征提取**：使用预训练的BERT模型提取文本特征
3. **XGBoost回归**：将BERT特征与`logvol-12`特征结合，训练XGBoost模型

### 使用方法

1. 首先训练BERT模型（如果尚未训练）：
```bash
python bert.py
```

2. 运行BERT+XGBoost混合模型：
```bash
python train.py
```

3. 如果已经提取过BERT特征和训练过XGBoost模型，下次运行时会自动加载：
```bash
python train.py  # 自动加载已有特征和模型
```

4. 如果想强制重新提取特征或重新训练模型：
```bash
python train.py --force_extract  # 强制重新提取BERT特征
python train.py --force_train    # 强制重新训练XGBoost模型
python train.py --force_extract --force_train  # 两者都强制重新执行
```

可选参数：
- `--data_dir`：数据目录，默认为`data`
- `--bert_model`：BERT预训练模型名称，默认为`bert-base-uncased`
- `--bert_model_path`：BERT模型路径，默认为`best_bert_regressor.pt`
- `--xgboost_model_path`：XGBoost模型保存/加载路径，默认为`bert_xgboost_model.pkl`
- `--max_length`：最大序列长度，默认为512
- `--batch_size`：批次大小，默认为16
- `--num_rounds`：XGBoost训练轮数，默认为100
- `--val_size`：验证集比例，默认为0.2
- `--seed`：随机种子，默认为42
- `--force_train`：强制重新训练XGBoost模型
- `--force_extract`：强制重新提取BERT特征

### 结果评估

模型训练完成后，会生成以下结果：
1. 保存XGBoost模型为`bert_xgboost_model.pkl`
2. 生成多种可视化图表用于评估模型性能：
   - `bert_xgboost_regression_results.png`：包含预测值与真实值散点图、残差图和误差分布直方图
   - `bert_xgboost_prediction_comparison.png`：真实值与预测值的时间序列对比图
   - `bert_xgboost_qq_plot.png`：Q-Q图，用于检查误差的正态性
   - `bert_xgboost_feature_importance.png`：XGBoost特征重要性图
3. 输出测试集上的评估指标：MSE、RMSE和R²
4. 如果存在纯XGBoost模型（不使用BERT特征），还会提供两个模型的性能比较

### 纯XGBoost模型

为了比较BERT特征的贡献，项目还提供了一个仅使用`logvol-12`作为特征的纯XGBoost模型：

```bash
python xgboost_model.py
```

这个模型不使用文本特征，仅依赖于过去12个月的对数交易量来预测未来12个月的对数交易量。通过比较纯XGBoost模型和BERT+XGBoost混合模型的性能，可以评估文本特征对预测的贡献。

## 安装指南

为了确保所有依赖项正确安装，特别是XGBoost库，我们提供了一个安装脚本：

```bash
python setup.py
```

这个脚本会自动安装所有必要的依赖，并验证XGBoost是否正确安装。如果发现XGBoost安装不完整，脚本会尝试重新安装正确的版本。

### 手动安装

如果您想手动安装依赖，请按照以下步骤操作：

1. 安装基本依赖：
```bash
pip install -r requirements.txt
```

2. 确保XGBoost正确安装：
```bash
pip install xgboost==1.7.6 --force-reinstall
```

### 常见问题解决

1. **AttributeError: module 'xgboost' has no attribute 'DMatrix'**
   - 这个错误通常是由于XGBoost安装不完整导致的
   - 解决方法：运行`python setup.py`或手动重新安装XGBoost

2. **CUDA相关错误**
   - 如果您在使用GPU时遇到CUDA错误，可以尝试在CPU上运行
   - 在`train.py`中，设备选择会自动检测GPU可用性

3. **内存不足错误**
   - 处理大型数据集时可能会遇到内存不足的问题
   - 解决方法：减小批次大小（使用`--batch_size`参数）或减小最大序列长度（使用`--max_length`参数） 