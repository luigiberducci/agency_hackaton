#!/bin/bash

for file in $(ls */*/models/rl_model*zip)
do
	steps=$(basename $file | cut -d . -f 1 | cut -d _ -f 3)
	if [ $((steps%100000)) -ne 0 ]
	then
		ls $file
		# for safety reason I commented the next line out
		# if you are really sure, uncomment it and run
		# rm $file	
	fi
done
