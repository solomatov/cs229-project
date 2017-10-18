#!/bin/bash

VENV_DIR=./env

if [ ! -d $VENV_DIR ]; then
    echo 'creating venv...'
    python3 -m venv $VENV_DIR
    echo 'done'
else
    echo 'venv have been already created'
fi

source $VENV_DIR/bin/activate
echo 'venv activated'

echo 'updating requirements'
pip3 install -r requirements.txt
echo 'done'