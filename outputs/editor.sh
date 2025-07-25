#!/bin/bash
searchdir=`ls ./prober/`
for entry in $searchdir
do
  filename=${entry%.*}
  # head -n 1 "./prober/${entry}" > "./editted/${entry}"
  head -n 50000 "./prober/${entry}" > "./editted/${entry}"
  # tail -n 20000 "./prober/${entry}" >> "./editted/${entry}"
done
  rm ./editted/*renew*
  rm ./editted/test_power.csv

