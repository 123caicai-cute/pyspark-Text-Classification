from pyspark import keyword_only
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline, Transformer
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.metrics import roc_curve, auc

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


# 1. 自定义转换器处理空列表问题
class EmptyListHandler(Transformer, HasInputCol, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable):
    """自定义转换器处理分词后的空列表问题"""

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(EmptyListHandler, self).__init__()
        self._setDefault(inputCol="words", outputCol="safe_words")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        def replace_empty(words):
            """如果分词后为空数组，替换为['unknown']"""
            if not words or len(words) == 0:
                return ["unknown"]
            return words

        replace_empty_udf = F.udf(replace_empty, ArrayType(StringType()))
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        return dataset.withColumn(output_col, replace_empty_udf(F.col(input_col)))


# 初始化Spark会话
spark = SparkSession.builder \
    .appName("QNLI-Text-Classification") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# 设置文件路径
base_path = "./glue/QNLI"
train_path = os.path.join(base_path, "train.tsv")
valid_path = os.path.join(base_path, "dev.tsv")

# 1. 数据加载
train_df = spark.read.csv(train_path, sep="\t", header=True, inferSchema=True)
valid_df = spark.read.csv(valid_path, sep="\t", header=True, inferSchema=True)

# 打印数据模式

# 检查数据量
print(f"\n训练集大小: {train_df.count():,}")
print(f"验证集大小: {valid_df.count():,}")
# 检查数据结构
print("训练集示例:")
train_df.show(5, truncate=50)
print("\n验证集示例:")
valid_df.show(5, truncate=50)

# 2. 数据预处理 - 修复空值和格式问题
# GLUE QNLI格式处理
def clean_label(value):
    """将标签转换为数值（0或1）"""
    if value == "not_entailment":
        return 1
    elif value == "entailment":
        return 0
    else:
        try:
            return int(value) if value is not None else None
        except:
            return None


# 注册UDF
clean_label_udf = F.udf(clean_label, "int")

# 选择列并应用清洗
train_df = train_df.selectExpr("question as text", "sentence as context", "label") \
    .withColumn("label", clean_label_udf("label")) \
    .filter(F.col("label").isNotNull())

valid_df = valid_df.selectExpr("question as text", "sentence as context", "label") \
    .withColumn("label", clean_label_udf("label")) \
    .filter(F.col("label").isNotNull())

# 合并问题和上下文作为输入文本
train_df = train_df.withColumn("input_text", F.concat_ws(" ", F.trim("text"), F.trim("context")))
valid_df = valid_df.withColumn("input_text", F.concat_ws(" ", F.trim("text"), F.trim("context")))

# 过滤空文本
train_df = train_df.filter(F.length("input_text") > 0)
valid_df = valid_df.filter(F.length("input_text") > 0)

# 3. 数据探索 - 检查文本长度分布
print("\n文本长度统计:")
train_df.select(F.length("input_text").alias("length")).describe().show()
valid_df.select(F.length("input_text").alias("length")).describe().show()

# 4. 数据预处理管道 - 添加更多健壮性
tokenizer = RegexTokenizer(inputCol="input_text", outputCol="words", pattern="\\W", minTokenLength=2)
empty_list_handler = EmptyListHandler(inputCol="words", outputCol="safe_words")
stopwords = StopWordsRemover(inputCol="safe_words", outputCol="filtered_words")

# Word2Vec配置
word2vec = Word2Vec(
    vectorSize=100,
    minCount=10,  # 增加最小词频
    inputCol="filtered_words",
    outputCol="features"
)

# 5. 模型训练（逻辑回归）
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,  # 增加迭代次数
    regParam=0.05,
    elasticNetParam=0.8
)

# 6. 构建管道
pipeline = Pipeline(stages=[
    tokenizer,
    empty_list_handler,  # 添加自定义转换器处理空列表
    stopwords,
    word2vec,
    lr
])

print("\n开始训练模型...")
try:
    # 训练模型
    model = pipeline.fit(train_df)
    print("模型训练完成！")

    # 7. 预测与评估
    predictions = model.transform(valid_df)

    # 创建评估器 - 使用三个不同的评估器分别计算不同指标
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC"
    )

    # 计算各项指标
    accuracy = acc_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)
    auc_score = auc_evaluator.evaluate(predictions)

    print("\n验证集评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1分数 (F1): {f1_score:.4f}")
    print(f"AUC分数: {auc_score:.4f}")

    # 8. 结果可视化 - 只保留混淆矩阵和ROC曲线
    # 混淆矩阵
    conf_matrix = predictions.groupBy("label", "prediction").count().collect()

    # 提取混淆矩阵数据
    conf_dict = {(row.label, row.prediction): row['count'] for row in conf_matrix}
    labels = sorted([0, 1])  # QNLI是二分类问题

    # 创建矩阵
    matrix = np.zeros((len(labels), len(labels)))
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = conf_dict.get((true_label, int(pred_label)), 0)

    # 计算总数用于标准化
    total = np.sum(matrix)

    # ROC曲线数据准备
    # 收集预测概率和真实标签
    predictions.selectExpr("text", "prediction", "probability as positive_prob").show()
    roc_data = predictions.select("label", "probability").collect()
    y_true = np.array([row.label for row in roc_data])
    # 获取正类（label=1）的概率
    y_score = np.array([float(row.probability[1]) for row in roc_data])

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # 创建可视化图表
    plt.figure(figsize=(15, 6))

    # 1. 混淆矩阵
    plt.subplot(1, 2, 1)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    class_labels = ["蕴含", "不蕴含"]
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)

    # 添加百分比和绝对值
    matrix_percent = matrix.astype('float') / matrix.sum() * 100
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{int(matrix[i, j])}\n({matrix_percent[i, j]:.1f}%)",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black",
                     fontsize=10)

    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)

    # 2. ROC曲线
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
    plt.title('ROC曲线', fontsize=14)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('qnli_results.png', dpi=300)
    plt.show()
    print(f"可视化结果已保存为: qnli_results.png")

    # 9. 保存模型
    try:
        model_path = "qnli_classification_model"
        model.save(model_path)
        print(f"\n模型已成功保存到: {model_path}")
    except Exception as e:
        print(f"\n保存模型时出错: {str(e)}")
        print("尝试保存为PySpark本地模型格式...")
        try:
            model.write().overwrite().save(model_path)
            print(f"模型成功保存到: {model_path}")
        except Exception as e2:
            print(f"保存模型时仍然出错: {str(e2)}")

except Exception as e:
    print(f"\n模型训练过程中出现错误: {str(e)}")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print(f"错误类型: {exc_type}")
    print(f"错误信息: {exc_value}")

    # 创建详细的错误日志
    with open("error_log.txt", "w") as f:
        f.write(f"Error during model training:\n")
        f.write(f"Type: {exc_type}\n")
        f.write(f"Message: {exc_value}\n")
        f.write(f"Traceback:\n")
        import traceback

        traceback.print_tb(exc_traceback, file=f)

    print("详细的错误日志已保存到 error_log.txt")

finally:
    # 停止Spark会话
    spark.stop()
    print("\nSpark会话已停止")