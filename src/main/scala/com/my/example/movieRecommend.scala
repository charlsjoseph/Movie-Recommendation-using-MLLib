package com.my.example

import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object movieRecommend {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

//    if (args.length != 2) {
//      println("Usage: /path/to/spark/bin/spark-submit --driver-memory 2g --class MovieLensALS " +
//        "target/scala-*/movielens-als-ssembly-*.jar movieLensHomeDir personalRatingsFile")
//      sys.exit(1)
//    }

    
    
    // set up environment

    val conf = new SparkConf().setMaster("local")
      .setAppName("MovieLensALS")
    val sc = new SparkContext(conf)

    // load personal ratings

    val myRatings = loadRatings("/home/cloudera/Downloads/ml-1m/myRating.dat")
    val myRatingsRDD = sc.parallelize(myRatings, 1)

    // load ratings and movie titles

    val movieLensHomeDir = "hdfs://quickstart.cloudera:8020/user/cloudera/movieRatingDataset" ;


    val ratings = sc.textFile("hdfs://quickstart.cloudera:8020/user/cloudera/movieRatingDataset/ratings.dat").map { line =>
      val fields = line.split("::")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }
    

    val movies = sc.textFile("hdfs://quickstart.cloudera:8020/user/cloudera/movieRatingDataset/movies.dat").map { line =>
      val fields = line.split("::")
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect().toMap

    val numRatings = ratings.count()
    val numUsers = ratings.map(_._2.user).distinct().count()
    val numMovies = ratings.map(_._2.product).distinct().count()

    println("Got " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

        ratings.take(10).foreach(println);
  
      
    // split ratings into train (60%), validation (20%), and test (20%) based on the 
    // last digit of the timestamp, add myRatings to train, and cache them
    println("Coming here.")

    val numPartitions = 4
    val training = ratings.filter(x => x._1 < 6)
      .values
      .union(myRatingsRDD)
      .repartition(numPartitions)
      .cache()
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .repartition(numPartitions)
      .cache()
    val test = ratings.filter(x => x._1 >= 8).values.cache()

    val numTraining = training.count()
    
    
    val numValidation = validation.count()
    
        validation.take(10).foreach(println);

    val numTest = test.count()
    
            test.take(10).foreach(println);


    println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest)

    // train models and evaluate them on the validation set

    val ranks = List(12, 20)
    val lambdas = List(0.1, 10.0)
    val numIters = List(8, 10)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRmse(model, validation, numValidation)
      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = " 
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    // evaluate the best model on the test set

    val testRmse = computeRmse(bestModel.get, test, numTest)

    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")

    // create a naive baseline and compare it with the best model

    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRmse = 
      math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)
    val improvement = (baselineRmse - testRmse) / baselineRmse * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    // make personalized recommendations
    
        println(" the myRatedMovieIds : ")
    myRatings.foreach(println);

    val sortedMyrating = myRatings.sortBy(- _.rating ).take(5);
    
    val myRatedMovieIds = sortedMyrating.map(_.product).toSet
    
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    
    
    println(" the myRatedMovieIds : ")

    myRatedMovieIds.take(5).foreach(println)
    
        println(" the candidates : ")

    candidates.take(5).foreach(println)
    
    val recommendations = bestModel.get
      .predict(candidates.map((1, _)))
      .collect()
      .sortBy(-_.rating)
      .take(10)
      
    var i = 1
    println("Movies recommended for you:")
    
    recommendations.foreach { r =>
      println("%2d".format(i) + ": " + movies(r.product) + " : " + r.rating)
      i += 1
    }

    // clean up
    sc.stop()
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
//    println("printing the prediction for test data : ");
//    
//    predictions.foreach ( println);
    
    
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

  /** Load ratings from file. */
  def loadRatings(path: String): Seq[Rating] = {
    val lines = Source.fromFile(path).getLines()
    val ratings = lines.map { line =>
      val fields = line.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }.filter(_.rating > 0.0)
    if (ratings.isEmpty) {
      sys.error("No ratings provided.")
    } else {
      ratings.toSeq
    }
  }
  
}