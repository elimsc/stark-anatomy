from pyspark import SparkContext, SparkConf


conf = SparkConf().setAppName("test").setMaster("spark://zhdeMacBook-Pro:7077")
sc = SparkContext(conf=conf)

rdd = sc.textFile("trace.csv")

l = rdd.collect()

print(l)
