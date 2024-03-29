#!/bin/bash
set -e

echo "Starting SSH ..."
service ssh start

#python /code/app.py runserver 0.0.0.0:8000
python /code/app.py

