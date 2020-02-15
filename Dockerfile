#FROM jupyter/scipy-notebook:latest
#FROM jupyter/pyspark-notebook:latest
FROM jupyter/all-spark-notebook:latest
#FROM jupyter/pyspark-notebook:ad3574d3c5c7
# change UID & GID
#COPY entrypoint.sh /usr/local/bin/entrypoint.sh
#RUN chmod +x /usr/local/bin/entrypoint.sh
#ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

#USER root
#RUN usermod -u 1002 jovyan

# package install
USER root
#RUN apt update && \
#    apt install -y --no-install-recommends \
#    build-essential\
#    swig\
#    vim \
#    graphviz\
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential\
    curl\
    gnupg\
    swig\
    vim \
    graphviz

# scala

## Install sbt
#RUN \
#  curl -L -o sbt-$SBT_VERSION.deb https://dl.bintray.com/sbt/debian/sbt-$SBT_VERSION.deb && \
#  dpkg -i sbt-$SBT_VERSION.deb && \
#  rm sbt-$SBT_VERSION.deb && \
#  apt-get update && \
#  apt-get install sbt && \
#  sbt sbtVersion

RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list && \
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add && \
    apt-get update && \
    apt-get install -y \
    scala\
    openjdk-8-jdk\
    maven\
    junit\
    junit4\
    sbt


RUN chown -R $NB_UID /home/jovyan/.conda

USER $NB_UID
RUN conda update -y conda && \
#	conda install -y \
    conda install -c conda-forge --yes \
        'nodejs'\
        'ipympl'\
        'ipywidgets'\
        'gdal==2.4.*'\
        'pandas==1.*'\
        'statsmodels'\
        'pydotplus'\
        'memory_profiler'\
        'ipyparallel'\
        'snappy'\
        'fastparquet'\
        'geopy'\
        'geopandas'\
        'shapely'\
        'osmnx'\
        'folium'\
        'cartopy'\
        'selenium'\
        'geckodriver'\
        'firefox'\
        'holoviews'\
        'python-igraph'\
        'tensorflow=2.*'\
        'tensorflow-datasets'\
        'somoclu'\
        'watermark'\
#	'keras'\
        'chainer'\
        'd3'\
        'pygraphviz'\
        'swifter'\
        'imbalanced-learn'\
        'xgboost'\
        'lightgbm'\
        'django'\
        'flask'\
        'sqlite'\
        'blaze'\
        'pyprind'\
        'nltk'\
        'wtforms'\
        'opencv' \
        'autopep8' \
        'flake8' \
        'jupyter_contrib_nbextensions' \
        'jupyterlab-nvdashboard' \
        'jupyterlab_code_formatter' && \
        conda clean --all -f --yes && \
        fix-permissions $CONDA_DIR


# pyarrow is BROKEN... TEMPORARY treatment
#RUN pip uninstall pyarrow --yes && \
#RUN pip uninstall pyarrow --yes
#RUN pip install -U pip
#	install --upgrade setuptools && \
#RUN pip install --no-cache-dir pyarrow  graphframes
#RUN pip install --no-cache-dir pyarrow==0.14.1  graphframes
#RUN pip install graphframes

USER jovyan
#RUN echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64" >> ~/.bashrc
#RUN echo "export SPARK_HOME=/usr/local/spark" >> ~/.bashrc
#RUN echo "export PATH=\$PATH:$SPARK_HOME/bin" >> ~/.bashrc
#RUN echo "export PYSPARK_PYTHON=/opt/conda/bin/python" >> ~/.bashrc
#RUN echo "export PYSPARK_DRIVER_PYTHON=jupyter" >> ~/.bashrc
#RUN echo "export PYSPARK_DRIVER_PYTHON_OPTS='notebook' pyspark" >> ~/.bashrc

#RUN source ~/.bashrc
#RUN /usr/local/spark/bin/spark-shell --packages graphframes:graphframes:0.7.0-spark2.4-s_2.11
RUN /usr/local/spark/bin/pyspark --packages graphframes:graphframes:0.7.0-spark2.4-s_2.11

#RUN pip install --user mca py_d3 somoclu smopy
RUN pip install --user \
       	         mca \
                 py_d3 \
                 smopy \
                 graphframes 


# FONT
RUN mkdir ~/.fonts \
    && chown jovyan ~/.fonts \
    && chmod 755 ~/.fonts
RUN wget https://ipafont.ipa.go.jp/IPAexfont/ipaexg00401.zip \
    && unzip ipaexg00401.zip \
    && mv ipaexg00401 -t ~/.fonts/ \
    && rm ipaexg00401.zip \
    && rm -rf ~/.cache/* \
    && fc-cache -fv


# jupyter lab
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install jupyter-matplotlib
RUN jupyter labextension install @lckr/jupyterlab_variableinspector
RUN jupyter labextension install @jupyterlab/toc
#RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter
#RUN jupyter serverextension enable --py jupyterlab_code_formatter
#RUN jupyter labextension install jupyterlab_vim
RUN jupyter labextension install jupyterlab-nvdashboard
#RUN jupyter labextension install jupyterlab-flake8
#RUN jupyter contrib nbextension install --user

USER jovyan

ENV JAVA_HOME="/usr/lib/jvm/java-1.8.0-openjdk-amd64" \
    SPARK_HOME="/usr/local/spark" \
    PATH="$PATH:$SPARK_HOME/bin" \
    PYSPARK_PYTHON="/opt/conda/bin/python" \
    PYTHONPATH="$(ls -a ${SPARK_HOME}/python/lib/py4j-*-src.zip):${SPARK_HOME}/python:$PYTHONPATH"

