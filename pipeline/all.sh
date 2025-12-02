#!/bin/bash


echo "Starting Program..."

echo "Starting data process."
pipeline/data-process.sh

echo "Starting simulate season."
pipeline/simulate-season.sh

echo "Starting simulate simple."
pipeline/simulate-simple.sh

echo "Starting predict season."
pipeline/predict-season.sh

echo "Starting predict simple."
pipeline/predict-simple.sh

echo "Program ended"