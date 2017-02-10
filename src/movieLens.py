# coding: utf-8
import sys
from os.path import join

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS

reload(sys)
sys.setdefaultencoding("utf-8")


def train_model(training, num_iterations=10, rank=1, lambda_=0.01):
    return ALS.train(training.rdd, iterations=num_iterations, rank=rank, lambda_=lambda_, seed=0)


def tuning_model(training, test):
    testing = test.select(["userId", "movieId"]).rdd

    min_mse = 1e6
    best_rank = 10
    best_lambda = 1.0
    num_iterations = 10
    for rank in range(1, 5):
        for lambda_f in range(1, 5):
            lambda_ = lambda_f * 0.01
            asl = train_model(training, num_iterations, rank, lambda_)
            predictions = asl.predictAll(testing).toDF(["userId", "movieId", "p_rating"])
            rates_and_preds = test \
                .join(predictions,[test.userId == predictions.userId, test.movieId == predictions.movieId]) \
                .drop(test.userId).drop(test.movieId)

            MSE = rates_and_preds.rdd.map(lambda r: (r.rating - r.p_rating) ** 2) \
                      .reduce(lambda x, y: x + y)/rates_and_preds.count()
            print "rank = %s, lambda = %s, Mean Squared Error = %s" % (rank, lambda_, MSE)

            if MSE < min_mse:
                min_mse = MSE
                best_rank = rank
                best_lambda = lambda_

    print "*" * 80
    print "The best params are: rank = %s, lambda = %s, Mean Squared Error = %s" % \
          (best_rank, best_lambda, min_mse)
    print "*" * 80
    return train_model(training, num_iterations, best_rank, best_lambda)


def recommend_for_all(model, movies, result_path, file_out_flag=False):
    def parse_recommendations(line):
        res = []
        for item in line[1]:
            movie_name = b_movies.value[item[1]]
            res.append((movie_name, float(item[2])))
        return Row(user=line[0], recommendations=res)

    b_movies = spark.sparkContext.broadcast(dict((int(l[0]), l[1]) for l in movies.collect()))
    products_for_all_users = model.recommendProductsForUsers(10).map(parse_recommendations).toDF()
    recommendation_result = products_for_all_users.repartition(1).orderBy(products_for_all_users.user).rdd \
        .map(lambda l: Row(str(l.user) + "," + str(list((r[0], r[1]) for r in l.recommendations))))\
        .toDF().repartition(1)
    if file_out_flag:  # 输出到文件
        recommendation_result.write.text(result_path)


def recommend_for_users(model, movies, user_ids):
    def recommend_for_one_user(model, user_id):
        b_movies = spark.sparkContext.broadcast(dict((int(l[0]), l[1]) for l in movies.collect()))
        recommendations = model.recommendProducts(user_id, 10)
        result = []
        for item in recommendations:
            result.append((b_movies.value[item.product], item.rating))
        print "Recommendations for user ", user_id, ": ", result

    if isinstance(user_ids, list):
        for user in user_ids:
            recommend_for_one_user(model, user)
    elif isinstance(user_ids, int):
        recommend_for_one_user(model, user_ids)


def main(spark, data_path, result_path):
    movies = spark.read.csv(join(data_path, "movies.csv")).rdd.map(
        lambda l: Row(int(l[0]), l[1], l[2])).toDF(["movieId", "title", "genres"])
    ratings = spark.read.csv(join(data_path, "ratings.csv")).rdd.map(
        lambda l: Row(int(l[0]), int(l[1]), float(l[2]))).toDF(["userId", "movieId", "rating"])

    all_users = ratings.select("userId").distinct()
    all_movies = ratings.select("movieId").distinct()
    print "Got %d ratings from %d users on %d movies." \
          % (ratings.count(), all_users.count(), all_movies.count())

    training, test = ratings.randomSplit([0.8, 0.2], 1)

    # 是否使用默认参数，为 False 时从一系列参数中选取最优参数
    default_flag = True
    if default_flag:
        # 使用默认参数建立模型
        model = train_model(training)
    else:
        # 选择最优参数建立模型
        training.cache()
        model = tuning_model(training, test)
        training.unpersist()

    # 为以下所有用户作出推荐并输出到终端
    user_ids = list(u.userId for u in all_users.take(3))
    print "userIds: ", user_ids
    recommend_for_users(model, movies, user_ids)

    # 为所有的用户作出推荐并输出到文件
    file_out_flag = True
    recommend_for_all(model, movies, result_path, file_out_flag)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        data_path = sys.argv[1]  # 数据目录
        result_path = sys.argv[2]  # 保存结果目录
    else:
        print "用法：$SPARK_HOME/bin/spark-submit --master spark://master:7077 " \
              "movieLens.py /data_path/ /result/path"
        sys.exit()

    spark = SparkSession \
        .builder \
        .appName("MovieLens Recommendation") \
        .getOrCreate()

    main(spark, data_path, result_path)

    spark.stop()
