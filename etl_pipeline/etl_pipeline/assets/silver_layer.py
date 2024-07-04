import os
import joblib
from minio import Minio
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as Pl
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StopWordsRemover, VectorAssembler
from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from operator import add
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from dagster import asset, AssetIn, Output
import pandas as pd
import numpy as np

from pyspark.sql import DataFrame

from ..resources.spark_io_manager import init_spark_session
from pyspark.sql.functions import *
from pyspark.sql.types import *
import ast
import re
import nltk
nltk.download('stopwords')
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

REPLACE_BY_SPACE_RE = re.compile('[/(){}—[]|@,;‘?|।!-॥–’-]')
COMPUTE_KIND = "PySpark"
LAYER = "silver"


@asset(
    description="Cleaning keywords table",
    ins={
        "bronze_keywords": AssetIn(
            key_prefix=["bronze", "keywords"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "keywords"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_cleaned_keywords(context, bronze_keywords: pd.DataFrame) -> Output[DataFrame]:
    # Load keywords table from bronze layer in MinIO, into a Spark dataframe, then clean data
    context.log.debug("Start cleaning keywords table")
    with init_spark_session() as spark:
        # Convert bronze_trade from polars DataFrame to Spark DataFrame
        pandas_df = bronze_keywords
        keywords_schema = StructType([
            StructField("id", StringType(), nullable=True),
            StructField("keywords", StringType(), nullable=True)
        ])
        context.log.debug(
            f"Converted to pandas DataFrame with shape: {pandas_df.shape}")
        rdd = spark.createDataFrame(pandas_df, schema=keywords_schema).rdd
        convert_df = rdd.map(lambda row: row.asDict()) \
            .map(lambda row: row.update({'keyword_convert': ' '.join([item['name'].replace(" ", "") for item in ast.literal_eval(row['keywords'])])}) or row) \
            .toDF()
        context.log.info("Got Spark DataFrame")
        # spark.sql("CREATE SCHEMA IF NOT EXISTS hive_prod.silver")

        return Output(
            value=convert_df,
            metadata={
                "table": "silver_cleaned_keywords",
                "row_count": convert_df.count(),
                "column_count": len(convert_df.columns),
                "columns": convert_df.columns,
            },
        )


@asset(
    description="Cleaning ratings table",
    ins={
        "bronze_ratings": AssetIn(
            key_prefix=["bronze", "ratings"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "ratings"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_cleaned_ratings(context, bronze_ratings: pd.DataFrame) -> Output[DataFrame]:
    """
        Load ratings table from bronze layer in MinIO, into a Spark dataframe, then clean data
    """

    context.log.debug("Start cleaning ratings table")

    with init_spark_session() as spark:
        # Convert bronze_trade from polars DataFrame to Spark DataFrame

        pandas_df = bronze_ratings
        ratings_schema = StructType([
            StructField("userId", IntegerType(), nullable=True),
            StructField("movieId", IntegerType(), nullable=True),
            StructField("rating", FloatType(), nullable=True),
            StructField("timestamp", IntegerType(), nullable=True)
        ])

        context.log.debug(
            f"Converted to pandas DataFrame with shape: {pandas_df.shape}"
        )

        # rdd = spark.createDataFrame(pandas_df, schema=ratings_schema).rdd
        # convert_df = rdd.map(lambda row: row.asDict()) \
        #     .map(lambda row: row.update({'keyword_convert': ' '.join([item['name'].replace(" ", "") for item in ast.literal_eval(row['keywords'])])}) or row) \
        #     .toDF()
        rdd = spark.sparkContext.parallelize(
            pandas_df.to_records(index=False), numSlices=1000)
        rdd_tuples = rdd.map(lambda x: (
            int(x[0]), int(x[1]), float(x[2]), int(x[3])))
        convert_df = spark.createDataFrame(rdd_tuples, schema=ratings_schema)

        context.log.info("Got Spark DataFrame")

        return Output(
            value=convert_df,
            metadata={
                "table": "silver_cleaned_ratings",
                "row_count": convert_df.count(),
                "column_count": len(convert_df.columns),
                "columns": convert_df.columns,
            },
        )


@asset(
    description="Cleaning movies table",
    ins={
        "bronze_movies": AssetIn(
            key_prefix=["bronze", "movies"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "movies"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_cleaned_movies(context, bronze_movies: pd.DataFrame) -> Output[DataFrame]:
    """
        Load movies table from bronze layer in MinIO, into a Spark dataframe, then clean data
    """

    context.log.debug("Start cleaning movies table")

    with init_spark_session() as spark:
        # Convert bronze_trade from polars DataFrame to Spark DataFrame
        pandas_df = bronze_movies
        context.log.debug(
            f"Converted to pandas DataFrame with shape: {pandas_df.shape}"
        )

        movies_schema = StructType([
            StructField("adult", BooleanType(), nullable=True),
            StructField("belongs_to_collection", StringType(), nullable=True),
            StructField("budget", IntegerType(), nullable=True),
            StructField("genres", StringType(), nullable=True),
            StructField("homepage", StringType(), nullable=True),
            StructField("id", StringType(), nullable=True),
            StructField("imdb_id", StringType(), nullable=True),
            StructField("original_language", StringType(), nullable=True),
            StructField("original_title", StringType(), nullable=True),
            StructField("overview", StringType(), nullable=True),
            StructField("popularity", DoubleType(), nullable=True),
            StructField("poster_path", StringType(), nullable=True),
            StructField("production_companies", StringType(), nullable=True),
            StructField("production_countries", StringType(), nullable=True),
            StructField("release_date", DateType(), nullable=True),
            StructField("revenue", StringType(), nullable=True),
            StructField("runtime", IntegerType(), nullable=True),
            StructField("spoken_languages", StringType(), nullable=True),
            StructField("status", StringType(), nullable=True),
            StructField("tagline", StringType(), nullable=True),
            StructField("title", StringType(), nullable=True),
            StructField("video", BooleanType(), nullable=True),
            StructField("vote_average", FloatType(), nullable=True),
            StructField("vote_count", IntegerType(), nullable=True)
        ])

        values_to_delete = ['/zV8bHuSL6WXoD6FWogP9j4x80bL.jpg',
                            '/ff9qCepilowshEtG2GYWwzt2bs4.jpg', '/zaSf5OG7V8X8gqFvly88zDdRm46.jpg']
        pandas_df = pandas_df[~pandas_df['budget'].isin(values_to_delete)]
        pandas_df = pandas_df.fillna(value=0)
        pandas_df['adult'] = pandas_df['adult'].astype(bool)
        pandas_df['budget'] = pandas_df['budget'].astype('int')
        pandas_df['genres'] = pandas_df['genres'].astype('str')
        pandas_df['homepage'] = pandas_df['homepage'].astype('str')
        pandas_df['id'] = pandas_df['id'].astype('str')
        pandas_df['imdb_id'] = pandas_df['imdb_id'].astype('string')
        pandas_df['original_language'] = pandas_df['original_language'].astype(
            'string')
        pandas_df['original_title'] = pandas_df['original_title'].astype(
            'string')
        pandas_df['overview'] = pandas_df['overview'].astype(
            'string')
        pandas_df['popularity'] = pandas_df['popularity'].astype(
            'float')
        pandas_df['poster_path'] = pandas_df['poster_path'].astype('string')
        pandas_df['production_companies'] = pandas_df['production_companies'].astype(
            'str')
        pandas_df['production_countries'] = pandas_df['production_countries'].astype(
            'str')
        pandas_df['release_date'] = pd.to_datetime(
            pandas_df['release_date'], infer_datetime_format=True)
        pandas_df['spoken_languages'] = pandas_df['spoken_languages'].astype(
            'str')
        pandas_df['status'] = pandas_df['status'].astype(
            'string')
        pandas_df['tagline'] = pandas_df['tagline'].astype(
            'string')
        pandas_df['title'] = pandas_df['title'].astype(
            "string")
        pandas_df['video'] = pandas_df['video'].astype(bool)
        pandas_df['revenue'] = pandas_df['revenue'].astype(
            'string')
        pandas_df['runtime'] = pandas_df['runtime'].astype(
            'int')
        pandas_df['vote_count'] = pandas_df['vote_count'].astype(
            'int')

        context.log.info("Got Spark DataFrame")

        rdd = spark.createDataFrame(pandas_df, schema=movies_schema).rdd

        convert_df = rdd.map(lambda row: row.asDict()) \
            .map(lambda row: row.update({'time': f"{row['runtime'] // 60} hours {row['runtime'] % 60} minutes"}) or row) \
            .map(lambda row: row.update({'genres_convert': ' '.join([item['name'].replace(" ", "") for item in ast.literal_eval(row['genres'])])}) or row) \
            .toDF()
        convert_df = convert_df.drop('adult', 'belongs_to_collection', 'homepage',
                                     'video', 'spoken_languages', 'production_countries', 'production_companies', 'runtime')

        return Output(
            value=convert_df,
            metadata={
                "table": "silver_cleaned_movies",
                "row_count": convert_df.count(),
                "column_count": len(convert_df.columns),
                "columns": convert_df.columns,
            },
        )


@asset(
    description="Cleaning credits table",
    ins={
        "bronze_credits": AssetIn(key_prefix=["bronze", "credits"]),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "credits"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_cleaned_credits(context, bronze_credits: pd.DataFrame) -> Output[DataFrame]:
    context.log.debug("Start cleaning credits table")
    with init_spark_session() as spark:
        # Convert bronze_trade from pandas DataFrame to Spark DataFrame
        pandas_df = bronze_credits
        credits_schema = StructType([
            StructField("cast", StringType(), nullable=True),
            StructField("crew", StringType(), nullable=True),
            StructField("id", StringType(), nullable=True)
        ])
        rdd = spark.createDataFrame(pandas_df, schema=credits_schema).rdd
        convert_df = rdd.map(lambda row: row.asDict()) \
            .map(lambda row: row.update({'director': ''.join([item['name'].replace(" ", "") for item in ast.literal_eval(row['crew']) if item['job'] == 'Director'])}) or row) \
            .map(lambda row: row.update({'cast_names': ' '.join([item['name'].replace(" ", "") for item in ast.literal_eval(row['cast'])[:3]])}) or row) \
            .toDF()
        convert_df = convert_df.select('id', 'cast_names', 'director')
        return Output(
            value=convert_df,
            metadata={
                "table": "silver_cleaned_credits",
                "row_count": convert_df.count(),
                "column_count": len(convert_df.columns),
                "columns": convert_df.columns,
            },
        )


@asset(
    description="Merge all table",
    ins={
        "silver_cleaned_keywords": AssetIn(
            key_prefix=["silver", "keywords"]
        ),
        "silver_cleaned_movies": AssetIn(
            key_prefix=["silver", "movies"]
        ),
        "silver_cleaned_credits": AssetIn(
            key_prefix=["silver", "credits"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "final"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_merge_df(context, silver_cleaned_keywords: DataFrame, silver_cleaned_movies: DataFrame, silver_cleaned_credits: DataFrame) -> Output[DataFrame]:
    context.log.debug("Start cleaning credits table")
    final_df = silver_cleaned_movies \
        .join(silver_cleaned_keywords, on=['id'], how='inner') \
        .join(silver_cleaned_credits, on=['id'], how='inner') \
        .withColumn('comb', concat_ws(" ", col('keyword_convert'), col('cast_names'), col('director'), col('genres_convert')))

    return Output(
        value=final_df,
        metadata={
            "table": "silver_merge_df",
            "row_count": final_df.count(),
            "column_count": len(final_df.columns),
            "columns": final_df.columns,
        },
    )


@asset(
    description="Start training model ...",
    ins={
        "silver_merge_df": AssetIn(key_prefix=["silver", "final"]),
    },
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_fit_model(context, silver_merge_df: DataFrame):
    context.log.debug("Start training model ...")
    spark_df = silver_merge_df.select('id', 'comb')
    regexTokenizer = RegexTokenizer(
        gaps=False, pattern='\w+', inputCol='comb', outputCol='token')
    stopWordsRemover = StopWordsRemover(
        inputCol='token', outputCol='nostopwrd')
    # countVectorizer = CountVectorizer(inputCol="nostopwrd", outputCol="rawFeature")
    # iDF = IDF(inputCol="rawFeature", outputCol="idf_vec")
    word2Vec = Word2Vec(vectorSize=150, minCount=3,
                        inputCol='nostopwrd', outputCol='word_vec', seed=123)
    # vectorAssembler = VectorAssembler(inputCols=['idf_vec', 'word_vec'], outputCol='comb_vec')
    pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, word2Vec])
    pipeline_model = pipeline.fit(spark_df)
    pipeline_model.write().overwrite().save(
        "s3a://lakehouse/model/" + 'pipeline_model')


# sentiment

nltk.download('stopwords')
stop = stopwords.words('english')
porter = PorterStemmer()


# def preprocessor(text):
#     """ Return a cleaned version of text
#     """
#     # Remove HTML markup
#     text = re.sub('<[^>]*>', '', text)
#     # Save emoticons for later appending
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     # Remove any non-word character and append the emoticons,
#     # removing the nose character for standarization. Convert to lower case
#     text = (re.sub('[\W]+', ' ', text.lower()) +
#             ' ' + ' '.join(emoticons).replace('-', ''))

#     return text


# def tokenizer(text):
#     return text.split()


# def tokenizer_porter(text):
#     return [porter.stem(word) for word in text.split()]


@asset(
    description="Cleaning review table",
    ins={
        "bronze_review": AssetIn(
            key_prefix=["bronze", "review"]
        ),
    },
    io_manager_key="spark_io_manager",
    key_prefix=[LAYER, "review"],
    metadata={"mode": "overwrite"},
    compute_kind=COMPUTE_KIND,
    group_name=LAYER
)
def silver_cleaned_review(context, bronze_review: pd.DataFrame) -> Output[DataFrame]:
    context.log.debug("Start cleaning review table")

    with init_spark_session() as spark:
        # Convert bronze_trade from pandas DataFrame to Spark DataFrame

        pandas_df = bronze_review

        review_schema = StructType([
            StructField("review", StringType(), nullable=True),
            StructField("sentiment", StringType(), nullable=True)
        ])

        # rdd = spark.createDataFrame(pandas_df, schema=review_schema).rdd

        # convert_df = rdd.map(lambda row: row.asDict()) \
        #     .map(lambda row: row.update({'review_clean': BeautifulSoup(row['review'], "html.parser").get_text()}) or row) \
        #     .map(lambda row: row.update({'review_clean': row['review'].lower()}) or row) \
        #     .toDF()
        rdd = spark.sparkContext.parallelize(
            pandas_df.to_records(index=False), numSlices=50)

        rdd_convert = rdd \
            .map(lambda row: {'review': BeautifulSoup(row['review'], "html.parser").get_text(), 'sentiment': row['sentiment']}) \
            .map(lambda row: {'review': row['review'].lower(), 'sentiment': row['sentiment']})

        convert_df = spark.createDataFrame(rdd_convert, schema=review_schema)

        return Output(
            value=convert_df,
            metadata={
                "table": "silver_cleaned_review",
                "row_count": convert_df.count(),
                "column_count": len(convert_df.columns),
                "columns": convert_df.columns,
            },
        )


@asset(
    description="Start training model ...",
    ins={"silver_cleaned_review": AssetIn(key_prefix=["silver", "review"]), },
    compute_kind="Python",
    group_name=LAYER
)
def silver_sentiment_model(context, silver_cleaned_review: DataFrame):
    context.log.debug("Start training model ...")
    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )
    pandas_df = silver_cleaned_review.select('review', 'sentiment').toPandas()
    X = pandas_df['review']
    y = pandas_df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    tfidf = TfidfVectorizer(
        strip_accents=None, lowercase=False, preprocessor=None)
    param_grid = [{'vect__ngram_range': [(1, 1), (1, 2)],
                   'vect__stop_words': [stop, None],
                   'vect__use_idf': [True, False],
                   'clf__penalty': [None, 'l2']
                   }]
    lr_tfidf = Pl(
        [('vect', tfidf), ('clf', LogisticRegression(random_state=0, max_iter=500))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy',
                               cv=5, verbose=1, n_jobs=None)
    gs_lr_tfidf.fit(X_train, y_train)
    context.log.debug('Best parameter set: ' + str(gs_lr_tfidf.best_params_))
    context.log.debug('Best accuracy: %.3f' % gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    context.log.debug('Accuracy in test: %.3f' % clf.score(X_test, y_test))
    source_file = "model_sentiment.pkl"
    destination_file = "/model/model_sentiment.pkl"
    joblib.dump(clf, source_file)
    # Tải file model lên MinIO
    client.fput_object("lakehouse", destination_file, source_file)
