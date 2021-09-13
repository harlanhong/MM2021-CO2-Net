#!/bin/bash
#for t in {0..3}
##do
for i in {0..7}
do
  let e=$i
  let j=$i
  let f=$e
  python extract_I3D.py --index $e --split_len 98758 --gpu $j  --machine data2 --train SGA &
  #python extract_I3D.py --index $f --split_len 34 --gpu $j  --machine data2 --origin &
  done
  wait
#done
for q in {0..3}
do
  let e=$q
  let j=$q+4
  python extract_I3D.py --index $e --split_len 17341 --gpu $j --test  --machine data2 --train SGA &
done
wait
#for t in {0..1}
#do 
#  for q in {0..3}
#  do
#    let e=$t*4+$q
#    let j=$q+4
#    python extract_I3D.py --index $e --split_len 37 --gpu $j --test --machine data2 --origin &
#  done
#  wait
#done
python extract_I3D.py --merge --machine data2

