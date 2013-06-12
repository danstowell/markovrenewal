#!/bin/bash

# TODO put this directly into the datadir, for publishing the dataset (change $datadir to match)

datadir=~/birdsong/xenocanto_chiffchaff
mp3dir="$datadir/mp3"
wavdir="$datadir/wav"

mkdir -p "$wavdir"

for f in "$mp3dir"/XC*.mp3; do
	item=`basename $f .mp3`
	lame --decode $f - | sox - -c 1 -r 44100 -t wav - highpass 1500 gain -n > "$wavdir/$item.wav"
done

