#!/bin/bash

searchdir=`ls ./editted/`
for entry in $searchdir
do
  filename=${entry%.*}
  python3 plot.py -i "./editted/${entry}" -o "./image_editted/${filename}"
done
