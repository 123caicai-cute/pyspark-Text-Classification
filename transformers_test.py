from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, monotonically_increasing_id, when
from pyspark.sql.types import StringType, FloatType, StructType, StructField, IntegerType
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import mlflow
import time
import os
import json
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score
import seaborn as sns
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

os.environ['http_proxy'] = 'http://127.0.0.1:10809'
os.environ['https_proxy'] = 'http://127.0.0.1:10809'

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("GLUE_Text_Classification") \
    .config("spark.driver.memory", "32g") \
    .config("spark.executor.memory", "32g") \
    .config("spark.executor.cores", "6") \
    .config("spark.driver.maxResultSize", "32g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# 记录开始时间
start_time = time.time()

# 设置 MLflow 跟踪
mlflow.set_experiment("GLUE Text Classification")
mlflow.start_run()
mlflow.log_param("project", "GLUE文本分类")


# 1. 数据集加载函数
def load_glue_dataset(task_name, data_type="train", base_path="./glue"):
    """加载TSV格式的GLUE数据集"""
    # 任务路径映射
    task_paths = {
        "sst2": "SST-2",
        "qqp": "QQP",
        "qnli": "QNLI"
    }

    # 文件路径
    file_path = os.path.join(base_path, task_paths[task_name], f"{data_type}.tsv")
    print(f"加载 {task_name.upper()} 数据集: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        available_files = os.listdir(os.path.dirname(file_path))
        raise FileNotFoundError(f"文件不存在。可用文件: {available_files}")

    # 读取TSV文件
    df = spark.read.option("sep", "\t").option("header", True).csv(file_path)

    # 列名映射 - 修复QNLI的标签列名
    column_mapping = {
        "sst2": {
            "sentence": "text",
            "label": "target"
        },
        "qqp": {
            "question1": "text1",
            "question2": "text2",
            "is_duplicate": "target"
        },
        "qnli": {
            "question": "text1",
            "sentence": "text2",
            "label": "target"  # 修复标签列名拼写错误
        }
    }

    # 应用列名映射
    for orig_col, new_col in column_mapping[task_name].items():
        if orig_col in df.columns:
            df = df.withColumnRenamed(orig_col, new_col)

    # 选择必要列
    if task_name == "sst2":
        df = df.select("text", "target")
    else:
        df = df.select("text1", "text2", "target")

    # 转换目标列为整数
    df = df.withColumn("target", col("target").cast("int"))

    print(f"{task_name.upper()}数据集加载完成，样本数: {df.count()}")
    return df


# 2. 大语言模型分类器
class GLUEClassifier:
    """GLUE任务大语言模型分类器"""

    model_mapping = {
        "sst2": "hf-internal-testing/tiny-random-distilbert",
        "qqp": "hf-internal-testing/tiny-random-roberta",
        "qnli": "hf-internal-testing/tiny-random-bert"
    }


    label_mapping = {
        "sst2": {"NEGATIVE": 0, "POSITIVE": 1},
        "qqp": {"not_duplicate": 0, "duplicate": 1},
        "qnli": {"entailment": 0, "not_entailment": 1}
    }

    def __init__(self, task_name):
        self.task_name = task_name
        self.classifier = None
        self.setup()

    def setup(self):
        """初始化模型"""
        device = -1  # 强制使用CPU
        self.classifier = pipeline(
            "text-classification",
            model=self.model_mapping[self.task_name],
            device=device
        )
        return self

    def predict(self, text1, text2=None):
        """执行预测"""
        try:
            if self.task_name == "sst2":
                result = self.classifier(text1)[0]
                return result['label'], result['score']
            else:
                text = f"{text1} [SEP] {text2}"
                result = self.classifier(text)[0]
                return result['label'], result['score']
        finally:
            # 每次预测后释放内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def map_label(self, label):
        """标签映射"""
        return self.label_mapping[self.task_name].get(label, -1)


# 3. 使用pandas UDF进行分布式预测
def create_glue_predictor(task_name):
    """创建分布式预测函数"""
    classifier = GLUEClassifier(task_name)

    # 定义返回类型 - 包含目标列
    return_schema = StructType([
        StructField("target", IntegerType()),
        StructField("predicted_label", StringType())
    ])

    @pandas_udf(return_schema, PandasUDFType.GROUPED_MAP)
    def predict_batch(pdf):
        results = []
        for i, row in pdf.iterrows():
            try:
                if task_name == "sst2":
                    label, score = classifier.predict(row["text"])
                else:
                    label, score = classifier.predict(row["text1"], row["text2"])

                # 包含原始目标值
                results.append({
                    "target": row["target"],
                    "predicted_label": label
                })

                # 每处理100个样本释放一次内存
                if i % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"预测出错: {str(e)}")
                results.append({
                    "target": row["target"],
                    "predicted_label": "ERROR",
                })
        return pd.DataFrame(results)

    return predict_batch


# 4. 执行预测和评估
def run_glue_classification(df, task_name):
    """执行大语言模型分类并评估结果"""
    print(f"\n===== 开始 {task_name.upper()} 分类任务 =====")

    # 限制样本数量（调试用）
    df = df.limit(100)  # 只处理前100个样本

    # 创建预测器
    predict_udf = create_glue_predictor(task_name)

    # 添加唯一ID用于分组 - 使用单调递增ID避免数据倾斜
    df = df.withColumn("id", monotonically_increasing_id())

    # 添加分组列 - 基于ID分组
    df = df.withColumn("group", (col("id") % 20).cast("int"))

    # 执行预测
    results_df = df.groupby("group").apply(predict_udf)

    # 标签映射
    classifier = GLUEClassifier(task_name)
    label_mapping = classifier.label_mapping[task_name]

    # 创建表达式映射
    case_expr = None
    for k, v in label_mapping.items():
        condition = (col("predicted_label") == k)
        if case_expr is None:
            case_expr = when(condition, v)
        else:
            case_expr = case_expr.when(condition, v)

    # 添加默认值并转换为 double 类型
    case_expr = case_expr.otherwise(-1).cast("double")

    # 应用映射
    results_df = results_df.withColumn("predicted_label_num", case_expr)

    # 将 confidence 列也转换为 double 类型
    results_df = results_df.withColumn("confidence", col("confidence").cast("double"))

    # 收集结果到驱动程序
    results_pd = results_df.select("target", "predicted_label_num", "confidence").toPandas()

    # 在本地计算评估指标
    print("\n评估指标:")
    # 准确率
    accuracy = accuracy_score(results_pd['target'], results_pd['predicted_label_num'])
    print(f" - 准确率: {accuracy:.4f}")

    # AUC
    if len(results_pd['target'].unique()) > 1:
        auc_score = roc_auc_score(results_pd['target'], results_pd['confidence'])
        print(f" - AUC: {auc_score:.4f}")
    else:
        auc_score = 0.0
        print(" - 目标值只有一个类别，无法计算AUC")

    # F1分数
    f1 = f1_score(results_pd['target'], results_pd['predicted_label_num'], average='weighted')
    print(f" - F1分数: {f1:.4f}")

    # 记录到MLflow
    mlflow.log_metric(f"{task_name}_accuracy", accuracy)
    mlflow.log_metric(f"{task_name}_auc", auc_score)
    mlflow.log_metric(f"{task_name}_f1", f1)

    return results_pd, accuracy, auc_score


# 5. 可视化函数
def visualize_results(results_pd, task_name, accuracy, auc_score):
    """可视化分类结果"""
    print(f"\n生成 {task_name.upper()} 可视化结果...")

    # 1. 混淆矩阵
    plt.figure(figsize=(10, 8))
    confusion = pd.crosstab(
        results_pd['target'],
        results_pd['predicted_label_num'],
        rownames=['实际值'],
        colnames=['预测值']
    )
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{task_name.upper()} 混淆矩阵\n准确率: {accuracy:.4f}")
    plt.savefig(f"{task_name}_confusion.png")
    mlflow.log_artifact(f"{task_name}_confusion.png")
    plt.close()

    # 2. ROC曲线（只有在有多个类别时）
    if len(results_pd['target'].unique()) > 1 and len(results_pd['predicted_label_num'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(results_pd['target'], results_pd['confidence'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{task_name.upper()} ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig(f"{task_name}_roc.png")
    mlflow.log_artifact(f"{task_name}_roc.png")
    plt.close()

    print(f"可视化结果已保存并记录到MLflow")


# 6. 性能报告函数
def generate_report(results):
    """生成性能报告"""
    report_data = []
    for task, (df, acc, auc_score) in results.items():
        report_data.append({
            "任务": task.upper(),
            "样本数": len(df),
            "准确率": f"{acc:.4f}",
            "AUC": f"{auc_score:.4f}" if auc_score > 0 else "N/A"
        })

    report_df = pd.DataFrame(report_data)

    # 保存报告
    report_path = "performance_report"
    os.makedirs(report_path, exist_ok=True)
    report_df.to_json(os.path.join(report_path, "report.json"), orient="records")
    mlflow.log_artifact(report_path)

    # 可视化比较
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x="任务", y="准确率", data=report_df)
    plt.title("任务准确率对比")

    plt.subplot(1, 2, 2)
    # 只有有AUC值的任务才显示
    if "AUC" in report_df.columns and not report_df["AUC"].isnull().all():
        sns.barplot(x="任务", y="AUC", data=report_df)
        plt.title("任务AUC对比")

    plt.tight_layout()
    plt.savefig("task_comparison.png")
    mlflow.log_artifact("task_comparison.png")
    plt.close()

    return report_df


# 7. 主函数
def main():
    # 配置参数
    DATA_PATH = "./glue"
    TASKS = ["sst2"]  # 先只处理一个任务

    # 加载数据集
    task_dfs = {}
    for task in TASKS:
        try:
            task_dfs[task] = load_glue_dataset(task, "dev", DATA_PATH)
        except Exception as e:
            print(f"加载 {task.upper()} 数据集失败: {str(e)}")
            continue

    # 执行分类任务
    results = {}
    for task in TASKS:
        if task not in task_dfs:
            continue

        df = task_dfs[task]
        try:
            results_df, accuracy, auc_score = run_glue_classification(df, task)
            results[task] = (results_df, accuracy, auc_score)

            # 可视化
            visualize_results(results_df, task, accuracy, auc_score)

            # 保存结果
            output_path = f"{task}_results"
            os.makedirs(output_path, exist_ok=True)
            results_df.to_parquet(os.path.join(output_path, "results.parquet"))
            mlflow.log_artifact(output_path)
            print(f"{task.upper()}结果已保存到: {output_path}")
        except Exception as e:
            print(f"处理 {task.upper()} 任务时出错: {str(e)}")
            # 记录错误到MLflow
            mlflow.log_text(f"{task.upper()} error: {str(e)}", f"{task}_error.log")

    # 生成性能报告
    if results:
        report_df = generate_report(results)
        print("\n性能报告:")
        print(report_df)
    else:
        print("没有任务成功完成，无法生成报告")

    # 性能统计
    elapsed_time = time.time() - start_time
    print(f"\n=== 任务完成 ===")
    print(f"总执行时间: {elapsed_time:.2f}秒")

    # 结束MLflow运行
    mlflow.end_run()

    # 停止Spark会话
    spark.stop()
    print("Spark会话已停止")


if __name__ == "__main__":
    main()