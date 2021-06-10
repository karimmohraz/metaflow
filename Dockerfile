FROM demisto/sklearn:1.0.0.16411

RUN pip install --upgrade pip

# Metaflow libraries
RUN pip install awscli==1.18.204 click==7.1.2 requests==2.25.1 boto3==1.16.44 -qqq

# non-root mode
ENV HOME=/tmp
WORKDIR $HOME
