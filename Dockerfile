FROM python:3.x

LABEL description="Image with Kubernetes" \
      maintainer="<Infrastructure & Operations> io@yifeng2019.ru" \
      source="https://github.com/yifeng2019/pytorch_mnist"

ADD . /opt/pytorch_mnist/

RUN apk --no-cache add git ca-certificates bash openssl gcc libc-dev libffi-dev openssl-dev make \
    && cd /opt/pytorch_mnist/
    && python setup.py install \
    && apk del gcc libc-dev libffi-dev openssl-dev

ENV PATH="/opt/pytorch_mnist/:${PATH}"

WORKDIR /tmp/