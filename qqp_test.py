from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, RegexTokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, udf, expr, when
from pyspark.sql.types import DoubleType, IntegerType
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号

# 初始化Spark Session
spark = SparkSession.builder \
    .appName("QQP_Text_Classification") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()


# 1. 数据加载和预处理
def load_data(file_path):
    df = spark.read \
        .option("sep", "\t") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("quote", '"') \
        .option("escape", '"') \
        .csv(file_path)

    # 数据清洗
    df = df.select(
        col("id").cast("int"),
        col("qid1").cast("string"),
        col("qid2").cast("string"),
        col("question1").cast("string"),
        col("question2").cast("string"),
        col("is_duplicate").cast("float")
    ).na.drop(subset=["question1", "question2", "is_duplicate"])

    # 合并问题文本作为特征
    return df.withColumn("combined", expr("concat(question1, ' [SEP] ', question2)"))



train_path = "./glue/QQP/train.tsv"
dev_path = "./glue/QQP/dev.tsv"

train_df = load_data(train_path)
dev_df = load_data(dev_path)

print(f"训练集大小: {train_df.count()}")
print(f"验证集大小: {dev_df.count()}")

# 检查数据结构
print("训练集示例:")
train_df.show(5, truncate=50)
print("\n验证集示例:")
dev_df.show(5, truncate=50)

# 2. 特征工程
tokenizer = RegexTokenizer(
    inputCol="combined",
    outputCol="words",
    pattern="\\W",
    toLowercase=True
)

stopwords_remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

hashing_tf = HashingTF(
    inputCol="filtered_words",
    outputCol="raw_features",
    numFeatures=2 ** 16
)

idf = IDF(
    inputCol="raw_features",
    outputCol="features",
    minDocFreq=5
)

label_indexer = StringIndexer(
    inputCol="is_duplicate",
    outputCol="label"
)

# 3. 建立分类模型
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.5
)

pipeline = Pipeline(stages=[
    tokenizer,
    stopwords_remover,
    hashing_tf,
    idf,
    label_indexer,
    lr
])

# 4. 模型训练
model = pipeline.fit(train_df)

# 5. 模型评估
predictions = model.transform(dev_df)

# 指标计算
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    metricName="accuracy"
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    metricName="f1"
)

auc = evaluator_auc.evaluate(predictions)
acc = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print(f"\n评估结果:")
print(f" AUC: {auc:.4f}")
print(f" 准确率: {acc:.4f}")
print(f" F1分数: {f1:.4f}")


# 6. 结果可视化
def plot_confusion_matrix(predictions):
    # 转换为Pandas DataFrame
    pd_pred = predictions.select("label", "prediction").toPandas()

    # 绘制混淆矩阵
    cm = confusion_matrix(pd_pred["label"], pd_pred["prediction"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["非重复问题", "重复问题"])
    disp.plot(cmap="Blues")
    plt.title("混淆矩阵")
    plt.savefig("./qqp/visual/confusion_matrix.png")
    plt.show()


def plot_roc_curve(predictions):
    # 提取预测概率和标签
    predictions.selectExpr("combined as combined_question", "prediction","probability as positive_prob").show(10)
    pd_roc = predictions.select("label", "probability").toPandas()
    pd_roc["positive_prob"] = pd_roc["probability"].apply(lambda x: list(x)[1])

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(pd_roc["label"], pd_roc["positive_prob"])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig("./qqp/visual/roc_curve.png")
    plt.show()


def plot_class_distribution(df, title):
    class_dist = df.groupBy("is_duplicate").count().toPandas()
    plt.figure(figsize=(8, 6))
    plt.bar(class_dist["is_duplicate"].astype(str), class_dist["count"], color=['skyblue', 'salmon'])
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.title(title)
    plt.savefig(f"./qqp/visual/{title}.png".replace(" ", "_"))
    plt.show()


# 可视化结果
plot_confusion_matrix(predictions)
plot_roc_curve(predictions)
plot_class_distribution(train_df, "训练集类别分布")
plot_class_distribution(dev_df, "测试集类别分布")

# 停止Spark会话
spark.stop()