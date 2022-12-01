"""The assignment for Data-Intensive Programming 2022"""

from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StringIndexerModel


class Assignment:
    spark: SparkSession = SparkSession.builder \
        .appName("ex5") \
        .config("spark.driver.host", "localhost") \
        .master("local") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.shuffle.partitions", "5")

    # the data frame to be used in tasks 1 and 4
    dataD2: DataFrame = spark.read.options(inferSchema=True, header=True) \
        .csv("file:/C:/Users/Tuomas/Documents/COMP.CS.320/tuomas/python/data/dataD2.csv").cache()

    # the data frame to be used in task 2
    dataD3: DataFrame = spark.read.options(inferSchema=True, header=True) \
        .csv("file:/C:/Users/Tuomas/Documents/COMP.CS.320/tuomas/python/data/dataD3.csv").cache()

    # the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
    dataD2WithLabels: DataFrame = None  # REPLACE with actual implementation

    @staticmethod
    def task1(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        va: VectorAssembler = VectorAssembler(inputCols=['a', 'b'], outputCol='features')
        assembledDF: DataFrame = va.transform(df)
        kmeans: KMeans = KMeans(featuresCol='features', k=k)
        model: KMeansModel = kmeans.fit(assembledDF)
        clusters: List[Tuple[float, float]] = list(tuple([tuple(center) for center in model.clusterCenters()]))
        return clusters

    @staticmethod
    def task2(df: DataFrame, k: int) -> List[Tuple[float, float, float]]:
        va: VectorAssembler = VectorAssembler(inputCols=['a', 'b', 'c'], outputCol='features')
        assembledDF: DataFrame = va.transform(df)
        kmeans: KMeans = KMeans(featuresCol='features', k=k)
        model: KMeansModel = kmeans.fit(assembledDF)
        clusters: List[Tuple[float, float, float]] = list(tuple([tuple(center) for center in model.clusterCenters()]))
        return clusters

    print(task1(dataD2, 3))
    print(task2(dataD3, 3))

    @staticmethod
    def task3(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        indexerModel: StringIndexerModel = StringIndexer(inputCol='LABEL', outputCol='LABEL_NUMERIC').fit(df)
        indexedDF: DataFrame = indexerModel.transform(df)
        va: VectorAssembler = VectorAssembler(inputCols=['a', 'b', 'LABEL_NUMERIC'], outputCol='features')
        assembledDF: DataFrame = va.transform(indexedDF)
        kmeans: KMeans = KMeans(featuresCol='features', k=k)
        model: KMeansModel = kmeans.fit(assembledDF)
        ##transformed = model.transform(assembledDF)
        transformed = model.transform(assembledDF).groupBy("prediction").agg(
            functions.sum(functions.col("LABEL_NUMERIC")),
            functions.avg(functions.col("a")),
            functions.avg(functions.col('b'))) \
            .sort(functions.col("sum(LABEL_NUMERIC)").desc()).select("*")
        ##print(model.summary)
        for i in transformed.collect():
            print(i)

        clusters: List[Tuple[float, float, float]] = list(tuple([tuple(center) for center in model.clusterCenters()]))
        clusters = sorted(clusters, key=lambda x: x[2], reverse=True)[0:2]
        clustersTop2Fatal: List[Tuple[float, float]] = [t[:2] for t in clusters]

        print(clustersTop2Fatal)
        print(clusters)

        return clusters2

    task3(dataD2, 4)

    # Parameter low is the lowest k and high is the highest one.
    @staticmethod
    def task4(df: DataFrame, low: int, high: int) -> List[Tuple[int, float]]:
        pass  # REPLACE with actual implementation
