from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import cv2
import numpy as np
import os
import pandas as pd
import re
import pyspark.sql.functions as F
# $example on$
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import DCT
from pyspark.ml import Pipeline
from ClusteredBands import *


spark = SparkSession.builder.appName("LandSatExample") \
    .config("spark.driver.memory", "50g") \
    .config("spark.driver.maxResultSize", "50g") \
    .config("executor-memory", "60g") \
    .getOrCreate()

# (1) load remote sensing data into Spark pipeline...
# struct ['origin', 'height', 'width', 'nChannels', 'mode', 'data']
df = spark.read.format("image") \
    .option("pathGlobFilter", "*.TIF")\
    .option("recursiveFileLookup", "true")\
    .load("/Users/thiyaj1/Downloads/midterm2/Test3")

df.show()

df2 = df.selectExpr(
    "split_part(image.origin, '/', -2) as Scene",
    "split_part(split_part(image.origin, '_', -1), '.', 1) as Band",
    "split_part(image.origin, '//', 2) as path")
df2.show()

df3 = df2.groupBy("Scene").pivot("Band").agg(first("path"))
df3.show()

#data_collect = df2.collect()
band_list = [];
#currentScene = "";

# select only id and company
for rows in df3.select("Scene", "B4", "B5", "B6").collect():
    print(rows[0], rows[1], rows[2])
    band_list.append(rows[1])
    band_list.append(rows[2])
    band_list.append(rows[3])

#for row in data_collect:
#    print(row["path"])
 #   if currentScene == row["Scene"]:
  #      band_list.append(row["path"])
   # elif (currentScene != ""):
        # process the one Scene contents
    #    print('process the one Scene contents of ', currentScene)
    #processed_list = [ x for x in band_list if re.match(r'.*_B[4-6].TIF$', x)] # get all the band files
    #for i in processed_list:
    #    print(i)
    clustered_models = ClusteredBands(band_list, rows[0])
    clustered_models.set_raster_stack()
    ranges = np.arange(6,7, 1)
    plt.gca().set_prop_cycle(None)
    clustered_models.build_models(ranges)
    band_list = []

