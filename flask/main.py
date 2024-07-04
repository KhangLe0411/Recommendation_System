import numpy as np
import requests
import bs4 as bs
from flask import Flask, request, render_template
import json
# from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from tmdbv3api import TMDb
from tmdbv3api import Movie
from minio import Minio
import urllib.request, os, joblib, io
tmdb = TMDb()
tmdb.api_key = 'e9e9d8da18ae29fc430845952232787c'

app = Flask(__name__, template_folder='/app/frontend')
# global df_merge, df_list
df_merge = None
df_list = None
flag1 = False
flag2 = False


def get_model():
    client = Minio(
        endpoint="minio:9000",
        access_key="minio_username",
        secret_key="minio_password",
        secure=False
    )
    clf = joblib.load(io.BytesIO(client.get_object(
        "lakehouse", "/model/model_sentiment.pkl").read()))
    return clf

def init_spark_session():
    try:
        spark = (
            SparkSession.builder.master("spark://spark-master:7077")
            .appName("Spark IO Manager")
            .config("spark.jars", "/usr/local/spark/jars/hadoop-aws-3.3.2.jar,/usr/local/spark/jars/hadoop-common-3.3.2.jar,/usr/local/spark/jars/aws-java-sdk-1.12.367.jar,/usr/local/spark/jars/s3-2.18.41.jar,/usr/local/spark/jars/aws-java-sdk-bundle-1.11.1026.jar,/usr/local/spark/jars/iceberg-spark-runtime-3.3_2.12-1.5.2.jar")
            .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
            .config("spark.sql.catalog.spark_catalog.type", "hive")
            .config(f"spark.sql.catalog.hive_prod", "org.apache.iceberg.spark.SparkCatalog")
            .config(f"spark.sql.catalog.hive_prod.type", "hive")
            .config(f"spark.sql.catalog.hive_prod.uri", "thrift://hive-metastore:9083")
            .config("spark.sql.warehouse.dir", f"s3a://lakehouse/")
            .config("spark.hadoop.fs.s3a.endpoint", f"http://minio:9000")
            .config("spark.hadoop.fs.s3a.access.key", f"minio_username")
            .config("spark.hadoop.fs.s3a.secret.key", f"minio_password")
            .config(
                "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
            )
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.connection.ssl.enabled", "false")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.rpc.message.maxSize", "800")
            .enableHiveSupport()
            .getOrCreate()
        )
        return spark
    except Exception as e:
        raise e


def ListOfGenres(genre_json):
    genre_json = json.loads(genre_json.replace("'", '"'))
    if genre_json:
        genres = []
        genre_str = ", "
        for i in range(0, len(genre_json)):
            genres.append(genre_json[i]['name'])
        return genre_str.join(genres)


def date_convert(s):
    MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    date_str = s.strftime("%Y-%m-%d")
    y = date_str[:4]
    m = int(date_str[5:-3])
    d = date_str[8:]
    month_name = MONTHS[m-1]

    result = month_name + ' ' + d + ' ' + y
    return result


def CosineSim(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    denominator = np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2))
    return float(numerator / denominator)


def get_titles():
    try:
        global df_merge
        global n_df
        global flag1
        if not flag1:
            spark = init_spark_session()
            df_merge = spark.read.format("iceberg").load(
                "hive_prod.silver.merge_df")
            n_df = df_merge.toPandas()
            flag1 = True
        return list(n_df['title'])
    except Exception as e:
        raise e


def get_vecs():
    try:
        global df_list, all_vecs, flag2
        if not flag2:
            spark = init_spark_session()
            df_list = spark.read.format("iceberg").load(
                "hive_prod.gold.transform_model")
            # rows_rdd = spark.read.format("iceberg").load("hive_prod.gold.transform_model").rdd.map(lambda row: Row(id=row.id, vecs=Vectors.dense([float(x) for x in row.vecs.strip('[]').split(',')])))
            # new_df = spark.createDataFrame(rows_rdd)
            # df_list = spark.createDataFrame(spark.read.format("iceberg").load("hive_prod.gold.transform_model").rdd.map(
            #     lambda row: Row(id=row.id, vecs=Vectors.dense([float(x) for x in row.vecs.strip('[]').split(',')])))).collect()
            # all_vecs = [(row['id'], row['vecs']) for row in df_list]
            data_original = df_list.collect()
            all_vecs = [(row.id, Vectors.dense(row.vecs))
                        for row in data_original]
            flag2 = True
        return all_vecs
    except Exception as e:
        raise e


def recommendation(m_title, sim_mov_limit=10):
    global df_merge
    schema = StructType([
        StructField("movies_id", StringType(), True),
        StructField("score", IntegerType(), True),
        StructField("input_movies_id", StringType(), True)
    ])

    # print(df_merge.filter(col("title") == m_title).count())
    # print(df_merge.count())
    # return 'Sorry! The movie you searched is not in our database. Please check the spelling or try with some other movies'

    if not df_merge.filter(col("title") == m_title).count() > 0:
        return ('Sorry! The movie you searched is not in our database. Please check the spelling or try with some other movies')
    else:
        spark = init_spark_session()
        sc = spark.sparkContext
        similar_movies_df = spark.createDataFrame([], schema)
        all_movies_vecs = get_vecs()
        m_id = df_merge.filter(col("title") == m_title).select(
            col('id')).collect()[0][0]
        input_vec = [(r[1]) for r in all_movies_vecs if r[0] == m_id][0]
        similar_movies_rdd = sc.parallelize(
            [(i[0], float(CosineSim(input_vec, i[1]))) for i in all_movies_vecs], numSlices=5)
        similar_movies_df = spark.createDataFrame(similar_movies_rdd) \
            .withColumnRenamed('_1', 'movies_id') \
            .withColumnRenamed('_2', 'score') \
            .orderBy("score", ascending=False) \
            .na.drop()
        similar_movies_df = similar_movies_df.filter(
            col("movies_id") != m_id).limit(sim_mov_limit)
        similar_movies_df = similar_movies_df.withColumn(
            'input_movies_id', lit(m_id))
        # spark.stop()
        return similar_movies_df
    # return 'a'


def getMovieDetails(in_mov):
    global df_merge
    vote_counts = df_merge.filter(
        col("vote_count").isNotNull()).select(col("vote_count"))
    vote_averages = df_merge.filter(
        col("vote_average").isNotNull()).select(col("vote_average"))
    C = vote_averages.select(mean("vote_average")).collect()[0][0]
    quantiles = vote_counts.approxQuantile("vote_count", [0.7], 0.001)
    m = quantiles[0]
    qualified = df_merge.filter((col("vote_count") >= m) & col(
        "vote_count").isNotNull() & col("vote_average").isNotNull())
    qualified = qualified.withColumn("vote_count", col("vote_count").cast("int")) \
        .withColumn("vote_average", col("vote_average").cast("int"))
    weighted_rating_udf = udf(lambda v, R: (
        v / (v + m) * R) + (m / (m + v) * C), FloatType())
    qualified = qualified.withColumn("weighted_rating", weighted_rating_udf(
        col("vote_count"), col("vote_average")))
    qualified = qualified.orderBy(col("weighted_rating").desc())

    if type(in_mov) == type('string'):
        return "f"
    a = in_mov.alias("a")
    b = qualified.alias("b")

    raw = a.join(b, col("a.movies_id") == col("b.id"), 'inner') \
        .orderBy("score", ascending=False) \
        .select([col('a.' + c) for c in a.columns] + [col('b.title'), col('b.genres_convert'), col('b.keyword_convert'), col("b.director"), col("b.cast_names"), col("b.weighted_rating")])
    # return raw.select("title").rdd.flatMap(lambda x: x).collect()
    return raw.select("title", "genres_convert", "director", "cast_names", "score", "weighted_rating")


@app.route("/")
def home():
    suggestions = get_titles()
    return render_template('home.html', suggestions=suggestions)


@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = getMovieDetails(recommendation(movie))
    title_list = r.select("title").rdd.flatMap(lambda x: x).collect()
    genre_list = r.select("genres_convert").rdd.flatMap(lambda x: x).collect()
    director_list = r.select("director").rdd.flatMap(lambda x: x).collect()
    cast_list = r.select("cast_names").rdd.flatMap(lambda x: x).collect()
    score_list = r.select("score").rdd.flatMap(lambda x: x).collect()
    suggestions = get_titles()
    if type(r) == type('string'):
        return render_template('recommend.html', r=r, t='s', suggestions=suggestions)
    else:
        # tmdb_movie = Movie()
        raw_df = df_merge.filter(col("title") == movie).select(
            col('id'), col('imdb_id'), col('vote_count'), col(
                'vote_average'), col('release_date'),
            col('status'), col('time'), col('genres'), col('overview'), col('popularity'))
        m_id = raw_df.select(col('id')).collect()[0][0]
        imdb_id = raw_df.select(col('imdb_id')).collect()[0][0]

        response = requests.get(
            'https://api.themoviedb.org/3/movie/{}?api_key={}'.format(m_id, tmdb.api_key))
        data_json = response.json()
        poster = data_json['poster_path']
        img_path = 'https://image.tmdb.org/t/p/original{}'.format(poster)
        # genres = [ListOfGenres(row[0]) for row in raw_df.select(col('genres')).collect()[0][0]]
        genres = ListOfGenres(
            [row[0] for row in raw_df.select(col('genres')).collect()][0])
        vote_count = "{:,}".format(raw_df.select(
            col('vote_count')).collect()[0][0])
        rd = date_convert(raw_df.select(col('release_date')).collect()[0][0])
        status = raw_df.select(col('status')).collect()[0][0]
        runtime = raw_df.select(col('time')).collect()[0][0]
        overview = raw_df.select(col('overview')).collect()[0][0]
        vote_average = raw_df.select(col('vote_average')).collect()[0][0]
        popularity = raw_df.select(col('popularity')).collect()[0][0]

        posters = []
        ids = []
        for movie_title in title_list:
            # tmdb_movie = Movie()
            movie_id = int(df_merge.filter(col("title") == movie_title).select(col('id')).collect()[0][0])
            # print('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id, tmdb.api_key))
            # response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id, tmdb.api_key))
            # dt_json = response.json()
            # print(dt_json)
            # p = dt_json["poster_path"]
            # img_path = 'https://image.tmdb.org/t/p/original{}'.format(p)
            # poster.append(img_path)
            # list_result = tmdb_movie.search(movie_title[0])
            # movie_id = list_result[0].id
            response = requests.get(
                'https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id, tmdb.api_key))
            data_json = response.json()
            posters.append('https://image.tmdb.org/t/p/original{}'.format(data_json['poster_path']))
            ids.append(movie_id)
        
        movie_cards = {posters[i]: [title_list[i], genre_list[i], director_list[i], cast_list[i], score_list[i]] for i in range(r.count())}
        # movie_cards = {posters[i]: r[i] for i in range(len(r))}

        # web scraping to get user reviews from IMDB site
        sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
        soup = bs.BeautifulSoup(sauce, 'lxml')
        soup_result = soup.find_all("div", {"class": "text show-more__control"})

        reviews_list = []  # list of reviews
        reviews_status = []  # list of comments (good or bad)
        clf = get_model()
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # passing the review to our model
                pred = clf.predict([reviews.string])
                reviews_status.append('Good' if pred[0] == 'positive' else 'Bad')
        
        # combining reviews and comments into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

        return render_template('recommend.html', genres=genres, img_path=img_path, movie=movie,
                               vote_count=vote_count, release_date=rd, status=status, runtime=runtime,
                               overview=overview, vote_average=vote_average, t='l', reviews=movie_reviews,
                               popularity=popularity, cards=movie_cards, suggestions=suggestions)
    # return render_template('recommend.html', t='l')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
