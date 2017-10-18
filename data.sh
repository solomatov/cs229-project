#!/bin/bash

DATA_DIR=./data
CIFAR_DIR=$DATA_DIR/cifar10
CIFAR_URL=https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

if [ ! -d $DATA_DIR ]; then
    echo 'creating data dir'
    mkdir $DATA_DIR
fi

if [ ! -d $CIFAR_DIR ]; then
    echo 'creating cifar10 dir'
    mkdir $CIFAR_DIR
    CIFAR_FILE=$CIFAR_DIR/cifar.tar.gz
    wget $CIFAR_URL -O $CIFAR_FILE
    tar xvf $CIFAR_FILE -C $CIFAR_DIR --strip-components=1
fi

