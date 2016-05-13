/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.cloudera.datascience.montecarlorisk

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.{KryoSerializer, KryoRegistrator}
import com.esotericsoftware.kryo.Kryo
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.distribution.MultivariateNormalDistribution

import scala.io.Source
import java.io.PrintWriter

case class Stock(weights: Array[Double], min: Double = 0,
  max: Double = Double.MaxValue)

class MyRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[Stock])
  }
}

object MonteCarloRisk {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Monte Carlo Risk")
    sparkConf.set("spark.serializer", classOf[KryoSerializer].getName)
    sparkConf.set("spark.kryo.registrator", classOf[MyRegistrator].getName)
    val sc = new SparkContext(sparkConf)
    val stocks = readStocks(args(0))
    val attempts = args(1).toInt
    val concurrency = args(2).toInt
    val means = getMeans(args(3))
    val covars = getCovars(args(4))
    val seed = if (args.length > 5) args(5).toLong else System.currentTimeMillis()
    val broadcastStocks = sc.broadcast(stocks)

    val seeds = (seed until seed + concurrency)
    val seeR = sc.parallelize(seeds, concurrency)

    val attemptsR = seeR.flatMap(getVs(_, attempts / concurrency,
      broadcastStocks.value, means, covars))

    attemptsR.cache()
    val result = attemptsR.takeOrdered(math.max(attempts / 20, 1)).last
    println("VaR: " + result)
    val dom = Range.Double(20.0, 60.0, .2).toArray
    val dens = KernelDensity.estimate(attemptsR, 0.25, dom)
    val pw = new PrintWriter("dens.csv")
    for (point <- dom.zip(dens)) { pw.println(point._1 + "," + point._2) }
    pw.close()
  }

  def getVs(seed: Long, attempts: Int, stocks: Seq[Stock],
      means: Array[Double], covars: Array[Array[Double]]): Seq[Double] = {
    val rand = new MersenneTwister(seed)
    val multivariateNormal = new MultivariateNormalDistribution(rand, means,
      covars)

    val getVs = new Array[Double](attempts)
    for (i <- 0 until attempts) {
      val attempt = multivariateNormal.sample()
      getVs(i) = getV(attempt, stocks)
    }
    getVs
  }

  def getV(attempt: Array[Double], stocks: Seq[Stock]): Double = {
    var totalValue = 0.0
    for (stock <- stocks) {
      totalValue += stockTV(stock, attempt)
    }
    totalValue
  }

  def stockTV(stock: Stock, attempt: Array[Double]): Double = {
    var stockTV = 0.0
    var i = 0
    while (i < attempt.length) {
      stockTV += attempt(i) * stock.weights(i)
      i += 1
    }
    Math.min(Math.max(stockTV, stock.min), stock.max)
  }

  def readStocks(file: String): Array[Stock] = {
    val src = Source.fromFile(file)
    val stocks = src.getLines().map(_.split(",")).map(
      x => new Stock(x.slice(2, x.length).map(_.toDouble), x(0).toDouble, x(1).toDouble))
    stocks.toArray
  }

  def getMeans(file: String): Array[Double] = {
    val src = Source.fromFile(file)
    val means = src.getLines().map(_.toDouble)
    means.toArray
  }

  def getCovars(file: String): Array[Array[Double]] = {
    val src = Source.fromFile(file)
    val covs = src.getLines().map(_.split(",")).map(_.map(_.toDouble))
    covs.toArray
  }
}
