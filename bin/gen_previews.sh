#!/bin/bash
for f in out/*.svg
do
	bn="$(basename $f)"
	if [[ "$bn" == square* ]]; then
        continue
    fi
	convert $f out/preview/${bn%.*}.png
done
