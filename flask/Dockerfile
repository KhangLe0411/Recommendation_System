FROM python:3.9.13

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk

RUN curl -O https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz \
    && tar zxvf spark-3.3.2-bin-hadoop3.tgz \
    && rm -rf spark-3.3.2-bin-hadoop3.tgz \
    && mv spark-3.3.2-bin-hadoop3/ /usr/local/ \
    && rm -rf /usr/local/spark \
    && rm -rf /usr/local/spark-3.3.0-bin-hadoop3 \
    && ln -s /usr/local/spark-3.3.2-bin-hadoop3 /usr/local/spark

RUN curl -O https://repo1.maven.org/maven2/software/amazon/awssdk/s3/2.18.41/s3-2.18.41.jar \
    && curl -O https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.12.367/aws-java-sdk-1.12.367.jar \
    && curl -O https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar \
    && curl -O https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar \
    && curl -O https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.2/hadoop-common-3.3.2.jar\
    # && curl -O https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark3-runtime/0.13.2/iceberg-spark3-runtime-0.13.2.jar \
    && curl -O https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.3_2.12/1.5.2/iceberg-spark-runtime-3.3_2.12-1.5.2.jar \
    && mv s3-2.18.41.jar /usr/local/spark/jars \
    && mv aws-java-sdk-1.12.367.jar /usr/local/spark/jars \
    && mv aws-java-sdk-bundle-1.11.1026.jar /usr/local/spark/jars \
    && mv hadoop-aws-3.3.2.jar /usr/local/spark/jars \
    && mv hadoop-common-3.3.2.jar /usr/local/spark/jars \
    # && mv iceberg-spark3-runtime-0.13.2.jar /usr/local/spark/jars
    && mv iceberg-spark-runtime-3.3_2.12-1.5.2.jar /usr/local/spark/jars

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]