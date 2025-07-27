from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from PIL import Image
import os

# 忽略警告信息
warnings.filterwarnings('ignore')
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号

# 1. 初始化Spark会话
spark = SparkSession.builder \
    .appName("SST-2 Sentiment Analysis") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# 2. 加载本地数据集
# train_path = "file:///home/hadoop/data/glue/SST-2/train.tsv"
# dev_path = "file:///home/hadoop/data/glue/SST-2/dev.tsv"

train_path = "./glue/SST-2/train.tsv"
dev_path = "./glue/SST-2/dev.tsv"

# 读取数据集
train_df = spark.read.csv(train_path, sep="\t", header=True, inferSchema=True)
dev_df = spark.read.csv(dev_path, sep="\t", header=True, inferSchema=True)

print(f"训练集大小: {train_df.count()}, 验证集大小: {dev_df.count()}")

# 3. 数据预处理
train_df = train_df.dropna()
dev_df = dev_df.dropna()

# 检查数据结构
print("训练集示例:")
train_df.show(5, truncate=50)
print("\n验证集示例:")
dev_df.show(5, truncate=50)

# 4. 构建文本处理Pipeline
tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=3000)
idf = IDF(inputCol="raw_features", outputCol="features")

# 5. 创建逻辑回归模型
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# 构建Pipeline模型
pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lr])

# 6. 训练模型
model = pipeline.fit(train_df)

# 7. 在验证集上进行预测
predictions = model.transform(dev_df)

# 8. 定义函数处理向量列
def extract_probability(vec):
    return float(vec[1])

# 注册UDF
extract_prob_udf = udf(extract_probability, FloatType())

# 添加概率列
predictions = predictions.withColumn("positive_prob", extract_prob_udf(col("probability")))

# 9. 模型评估
# 二元评估器
binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = binary_evaluator.evaluate(predictions)


# 多类评估器
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

print(f"\n模型评估结果:")
print(f" AUC = {auc:.4f}")
print(f" 准确率 = {accuracy:.4f}")
print(f" F1分数 = {f1:.4f}")

# 10. 使用NumPy数组直接绘图
# 收集数据为NumPy数组
labels = np.array(predictions.select("label").rdd.flatMap(lambda x: x).collect())
preds = np.array(predictions.select("prediction").rdd.flatMap(lambda x: x).collect())
probs = np.array(predictions.select("positive_prob").rdd.flatMap(lambda x: x).collect())

# 11. 结果可视化 - 分别保存每个图表
# 创建目录保存临时图像
os.makedirs("./temp_plots", exist_ok=True)

# A. 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(labels, preds)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = np.arange(len(cm))
plt.xticks(tick_marks, ['Negative (0)', 'Positive (1)'], rotation=45)
plt.yticks(tick_marks, ['Negative (0)', 'Positive (1)'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')

# 添加数值标注
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig("./temp_plots/confusion_matrix.png", dpi=150)
plt.close()

# B. ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = roc_auc_score(labels, probs)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('接收者操作特征(ROC)曲线')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("./temp_plots/roc_curve.png", dpi=150)
plt.close()

# C. 特征重要性
plt.figure(figsize=(10, 8))
# 获取特征重要性
vocab = model.stages[2].vocabulary  # CountVectorizer的词汇表
coefficients = model.stages[-1].coefficients.toArray()  # 逻辑回归系数

# 直接排序特征重要性
sorted_indices = np.argsort(coefficients)
top_n = 20

# 获取最重要的正面和负面特征
top_pos_indices = sorted_indices[-top_n:][::-1]  # 从大到小排序
top_neg_indices = sorted_indices[:top_n]         # 从小到大排序

# 获取对应的单词和系数
top_pos_words = [vocab[i] for i in top_pos_indices]
top_pos_coefs = [coefficients[i] for i in top_pos_indices]

top_neg_words = [vocab[i] for i in top_neg_indices]
top_neg_coefs = [coefficients[i] for i in top_neg_indices]

# 绘制最重要的20个特征
plt.barh(top_pos_words, top_pos_coefs, color='green', alpha=0.7, label='正面')
plt.barh(top_neg_words, top_neg_coefs, color='red', alpha=0.7, label='负面')
plt.xlabel('特征重要性')
plt.title(f'Top {top_n} 重要特征')
plt.legend()
plt.tight_layout()
plt.savefig("./temp_plots/feature_importance.png", dpi=150)
plt.close()

# D. 预测概率分布
plt.figure(figsize=(8, 6))
for label in [0, 1]:
    mask = (labels == label)
    if np.any(mask):  # 确保有数据点
        data = probs[mask]
        # 绘制直方图
        plt.hist(data, bins=30, alpha=0.5, label=f'真实标签: {label}', density=True)

plt.xlabel('预测概率 (正类概率)')
plt.ylabel('密度')
plt.title('预测概率分布')
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("./temp_plots/probability_distribution.png", dpi=150)
plt.close()

# 12. 合并图像
# 创建一个新图像来组合所有图表
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# 加载并添加每个图表
images = [
    Image.open("./temp_plots/confusion_matrix.png"),
    Image.open("./temp_plots/roc_curve.png"),
    Image.open("./temp_plots/feature_importance.png"),
    Image.open("./temp_plots/probability_distribution.png")
]

# 添加图像到子图
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.axis('off')  # 关闭坐标轴

# 保存组合图像
plt.tight_layout()
plt.savefig("sst2_analysis_results.png", dpi=300)
plt.close()

# 清理临时文件
for file in os.listdir("./temp_plots"):
    os.remove(os.path.join("./temp_plots", file))
os.rmdir("./temp_plots")

print("\n可视化结果已保存为 'sst2_analysis_results.png'")

# 13. 保存预测结果
print("\n验证集预测结果示例:")
predictions.select("sentence", "prediction", "positive_prob").show(5, truncate=50)

# 保存完整预测结果
predictions.select("sentence", "label", "prediction", "positive_prob") \
    .write.mode("overwrite") \
    .csv("./predictions", header=True)

# 14. 保存模型
model.write().overwrite().save("./sst2_sentiment_model")
print("\n模型已保存到 'sst2_sentiment_model' 目录")

# 15. 停止Spark会话
spark.stop()