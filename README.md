tl2699 kl2855
# Estimating_VaR_based_on_Apache_Spark
This is a simple Spark program to calculate Value at Risk using Monte Carlo simulation.

How to run:
spark-submit --class com.cloudera.datascience.montecarlorisk.MonteCarloRisk --master local target/montecarlo-risk-0.0.1-SNAPSHOT.jar <instrument file> <trials> <parallelism> <mean file> <covariance file>

