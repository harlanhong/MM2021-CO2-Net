#!/bin/bash
for t in {0..1}
do 
  for q in {0..3}
  do
    let e=$t*4+$q
    let j=$q
    python extract_I3D.py --index $e --split_len 37 --gpu $j --test --machine data2 --origin &
  done
  wait
done
