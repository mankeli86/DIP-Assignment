package assignment22

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession


class Assignment {

  val spark: SparkSession = ???

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = ???

  // the data frame to be used in task 2
  val dataD3: DataFrame = ???

  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val dataD2WithLabels: DataFrame = dataD2  // REPLACE with actual implementation



  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    Array.empty  // REPLACE with actual implementation
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    Array.empty  // REPLACE with actual implementation
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    Array.empty  // REPLACE with actual implementation
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    Array.empty  // REPLACE with actual implementation
  }

}
