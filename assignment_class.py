#--------------------------------------------------------------------
#  Author:        Ahad Beykaei
#  Written:       23/02/2018
#  Last updated:  24/02/2018
#
#  Language:      Python 3.6
#  Execution:     Python Spark 2.0
#--------------------------------------------------------------------

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
import json
import pygeohash as pgh
from math import sin, cos, sqrt, atan2, radians

spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# load all files data
file = spark.sparkContext.textFile("file:///Users/iAhad/PycharmProjects/data_science2/Company_test/"
                                   "location-data-sample/*.gz")
# loading one file for test
#file = spark.sparkContext.textFile("file:///Users/iAhad/PycharmProjects/data_science2/Company_test/"
#                                   "location-data-sample/2017-04-01-ip-10-130-80-170-rsyslog-ip-10-130-80-231-ginger-json-log.log-1491004981")

class Mapper():
    file = None
    def __init__(self):
        pass

    # loading input data
    def load_data(self, dir_file):
        self.file = spark.sparkContext.textFile(dir_file)
        print('File loaded successfully...')

    # mapper for dataframe which contains Row object
    @staticmethod
    def mapper_df(line):
        obj = json.loads(line)
        pgh.encode(float(obj['lat']), float(obj['lng']))
        return Row(IDFA=obj['idfa'], lat=obj['lat'], lng=obj['lng'], geohash1=obj['geohash'],
                   geohash2=pgh.encode(float(obj['lat']), float(obj['lng']), precision=8))
    # mapper for rdd
    @staticmethod
    def mapper_rdd(line):
        obj = json.loads(line)
        pgh.encode(float(obj['lat']), float(obj['lng']))
        return (obj['idfa'], obj['lat'], obj['lng'], obj['geohash'], obj['event_time'],
                   pgh.encode(float(obj['lat']), float(obj['lng']), precision=7))

    # Creating rdd/df for file loaded
    def create_events(self, rdd_Type):
        if rdd_Type == 'df':
            # create RDD contains Row object
            # create DataFrame from RDD and cache it
            events = self.file.map(self.mapper_df)
            return spark.createDataFrame(events).cache()
        elif rdd_Type == 'rdd':
            # Create RDD
            return self.file.map(self.mapper_rdd)

# Loading all files
all_file = "file:///Users/iAhad/PycharmProjects/data_science2/Company_test/location-data-sample/*.gz"
test_file = "file:///Users/iAhad/PycharmProjects/data_science2/Company_test/location-data-sample/2017-04-01-ip-10-130-80-170-rsyslog-ip-10-130-80-231-ginger-json-log.log-1491004981"
database = Mapper()

# loading data and set
database.load_data(test_file)

###########################################################################
##
## Calculate location event statistics
##
###########################################################################
# Dataframe from file
user_df = database.create_events('df')

def location_events_stats():
    df_agg = user_df.groupBy('IDFA', 'geohash1').count()
    df_agg_stats = df_agg.groupby('IDFA').agg(min('count').alias('min_count'),
                                              max('count').alias('max_count'),
                                              avg('count').alias('avg_count'),
                                              stddev('count').alias('std_count'))

    return df_agg_stats

# Calculate location event stats and save as parquet format file
location_events_stats = location_events_stats()
location_events_stats.write.parquet("output/location_events_stats.parquet")

###########################################################################
##
## Generate clusters of events in different geohash areas
##
###########################################################################
# rdd from uploaded file
user_rdd = database.create_events('rdd')

geohash_cluster = user_rdd.map(lambda ev: (ev[5], (1, str(ev[1]) + ' ' + str(ev[2])))).\
    reduceByKey(lambda e1,e2: [e1[0]+e2[0], e1[1] + '|' + e2[1]]).filter(lambda x: x[1][0] >=2)

def calculate_distance(lat1, lon1, lat2, lon2):
    '''
    Calculate the distance between two points with (lat , lon) known
    '''
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # distance between two point
    return R * c

def calculate_avg_distance_geohash(event_list):
    point_list = event_list[1][1].split('|')
    sumDis = 0
    n = 0
    for i in range(0, len(point_list)-1):
        for j in range(i+1, len(point_list)):
            n+=1
            sumDis += calculate_distance(float(point_list[i].split()[0]), float(point_list[i].split()[1]), float(point_list[j].split()[0]), float(point_list[j].split()[1]))
    return (event_list[0], event_list[1][0], sumDis/n)

geohash_cluster_distance = spark.createDataFrame(geohash_cluster.map(calculate_avg_distance_geohash), schema=['geohash2', 'count', 'avg_distance'])
geohash_cluster_distance.write.parquet("output/geohash_cluster_distance.parquet")

###########################################################################
##
## Define users behaviour (IDFA) in different temporal-spetial coordinates
## Users with more than 1 events are considered
##
###########################################################################
# rdd of IDFAs and aggregate their locations and timestamp for further data analysis
IDFAs_rdd = user_rdd.map(lambda ev: (ev[0], (1, str(ev[1]) + ' ' + str(ev[2]), str(ev[4]) ))).\
    reduceByKey(lambda e1,e2: [e1[0]+e2[0], e1[1] + '|' + e2[1], e1[2] + '|' + e2[2]]).filter(lambda x: x[1][0] >=2)

def generate_IDFA_behaviour(event_list):
    point_list = event_list[1][1].split('|')
    time_list = event_list[1][2].split('|')

    return (event_list[0], event_list[1][0], point_list, time_list)


IDFAs_behaviour = spark.createDataFrame(IDFAs_rdd.map(generate_IDFA_behaviour), schema=['IDFA', 'num_points', 'points', 'times'])
IDFAs_behaviour.write.parquet("output/IDFAs_behaviour.parquet")