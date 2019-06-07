#!/bin/bash

FILES=$(ls samples/*.txt)

for file in $FILES; do
  bin/parse $file
done;
