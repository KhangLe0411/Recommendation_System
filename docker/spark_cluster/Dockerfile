FROM docker.io/bitnami/spark:3.3.2

USER root

# Install prerequisites
RUN apt-get update && apt-get install -y curl

RUN curl -O https://repo1.maven.org/maven2/software/amazon/awssdk/s3/2.18.41/s3-2.18.41.jar \
    && curl -O https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.12.367/aws-java-sdk-1.12.367.jar \
    && curl -O https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar \
    && curl -O https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.2/hadoop-common-3.3.2.jar \
    # && curl -O https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark3-runtime/0.13.2/iceberg-spark3-runtime-0.13.2.jar \
    && curl -O https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.3_2.12/1.5.2/iceberg-spark-runtime-3.3_2.12-1.5.2.jar \
    && mv s3-2.18.41.jar /opt/bitnami/spark/jars \
    && mv aws-java-sdk-1.12.367.jar /opt/bitnami/spark/jars \
    && mv aws-java-sdk-bundle-1.11.1026.jar /opt/bitnami/spark/jars \
    && mv hadoop-common-3.3.2.jar /opt/bitnami/spark/jars \
    # && mv iceberg-spark3-runtime-0.13.2.jar /opt/bitnami/spark/jars
    && mv iceberg-spark-runtime-3.3_2.12-1.5.2.jar /opt/bitnami/spark/jars

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install numpy
RUN pip3 install beautifulsoup4