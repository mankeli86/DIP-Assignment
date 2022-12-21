"""The assignment for Data-Intensive Programming 2022"""

from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import matplotlib.pyplot as plt


class Assignment:
    spark: SparkSession = SparkSession.builder \
        .appName("assignment") \
        .config("spark.driver.host", "localhost") \
        .master("local") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.shuffle.partitions", "5")

    # Schema for dataD2
    schema1: StructType = StructType([
        StructField("a", FloatType(), True),
        StructField("b", FloatType(), True),
        StructField("LABEL", StringType(), True)
    ])

    # Schema for dataD3
    schema2: StructType = StructType([
        StructField("a", FloatType(), True),
        StructField("b", FloatType(), True),
        StructField("c", FloatType(), True),
        StructField("LABEL", StringType(), True)
    ])

    @staticmethod
    def filterErroneousValues(df: DataFrame) -> DataFrame:
        colCount : int = len(df.columns)
        if colCount == 3:
            filtered2D: DataFrame = df.filter((functions.col("a").cast("float").isNotNull())
                                              & (functions.col("b").cast("float").isNotNull()))\
                .filter("LABEL = 'Ok' or LABEL = 'Fatal'")
            return filtered2D
        else:
            filtered3D: DataFrame = df.filter((functions.col("a").cast("float").isNotNull())
                                              & (functions.col("b").cast("float").isNotNull())
                                              & (functions.col("c").cast("float").isNotNull())) \
                .filter("LABEL = 'Ok' or LABEL = 'Fatal'")
            return filtered3D

    # the data frame to be used in tasks 1 and 4
    dataD2: DataFrame = filterErroneousValues(spark.read.options(header=True).schema(schema1)
                                              .csv("../data/dataD2.csv").cache())

    # the data frame to be used in task 2
    dataD3: DataFrame = filterErroneousValues(spark.read.options(header=True).schema(schema2)
                                              .csv("../data/dataD3.csv").cache())

    # the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
    dataD2WithLabels: DataFrame = StringIndexer(inputCol='LABEL', outputCol='LABEL_NUMERIC')\
        .fit(dataD2).transform(dataD2).cache()

    # Calculate cluster means for two-dimensional data
    @staticmethod
    def task1(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        va: VectorAssembler = VectorAssembler(inputCols=["a", "b"], outputCol="features")
        scaler: StandardScaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                                withStd=True, withMean=False)
        kmeans: KMeans = KMeans(featuresCol="scaledFeatures", k=k, seed=1)
        pipeline: Pipeline = Pipeline(stages=[va, scaler, kmeans])
        model: PipelineModel = pipeline.fit(df)
        clusters: List[Tuple[float, float]] = list(tuple([tuple(center * model.stages[1].std)
                                                          for center in model.stages[2].clusterCenters()]))
        return clusters

    # Calculate cluster means for three-dimensional data
    @staticmethod
    def task2(df: DataFrame, k: int) -> List[Tuple[float, float, float]]:
        va: VectorAssembler = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")
        scaler: StandardScaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                                withStd=True, withMean=False)
        kmeans: KMeans = KMeans(featuresCol="scaledFeatures", k=k, seed=1)
        pipeline: Pipeline = Pipeline(stages=[va, scaler, kmeans])
        model: PipelineModel = pipeline.fit(df)
        clusters: List[Tuple[float, float, float]] = list(tuple([tuple(center * model.stages[1].std)
                                                                 for center in model.stages[2].clusterCenters()]))
        return clusters

    # Calculate two cluster means that have largest count of Fatal data points
    @staticmethod
    def task3(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        va: VectorAssembler = VectorAssembler(inputCols=["a", "b", "LABEL_NUMERIC"], outputCol="features")
        scaler: StandardScaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                                withStd=True, withMean=False)
        kmeans: KMeans = KMeans(featuresCol="scaledFeatures", k=k, seed=1)
        pipeline: Pipeline = Pipeline(stages=[va, scaler, kmeans])
        model: PipelineModel = pipeline.fit(df)

        clusters: List[Tuple[float, float, float]] = list(tuple([tuple(center * model.stages[1].std)
                                                                 for center in model.stages[2].clusterCenters()]))
        clustersTop2Fatal: List[Tuple[float, float]] = \
            [t[:2] for t in sorted(clusters, key=lambda x: x[2], reverse=True)[0:2]]
        return clustersTop2Fatal

    # Calculate silhouette scores for dataframes with K in [low, high]
    @staticmethod
    def task4(df: DataFrame, low: int, high: int) -> List[Tuple[int, float]]:
        scores: List[Tuple[int, float]] = []
        va: VectorAssembler = VectorAssembler(inputCols=["a", "b"], outputCol="features")
        scaler: StandardScaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                                withStd=True, withMean=False)
        evaluator: ClusteringEvaluator = ClusteringEvaluator(featuresCol="scaledFeatures",
                                                             metricName='silhouette',
                                                             distanceMeasure='squaredEuclidean')
        # calculate different k values
        for k in range(low, high + 1):
            kmeans: KMeans = KMeans(featuresCol="scaledFeatures", k=k, seed=1)
            pipeline: Pipeline = Pipeline(stages=[va, scaler, kmeans])
            dfWithPredictions: DataFrame = pipeline.fit(df).transform(df)
            scores.append((k, evaluator.evaluate(dfWithPredictions)))

        # visualize scores
        plt.xlabel("K")
        plt.ylabel("Silhouette score")
        plt.scatter(*zip(*scores))
        plt.show(block=False)
        plt.pause(7)
        plt.close()
        return scores

