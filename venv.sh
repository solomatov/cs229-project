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

echo 'installing stanalone requirements...'
pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl
echo 'done'

echo 'updating requirements...'
pip3 install -r requirements.txt
echo 'done'