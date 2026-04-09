#!/bin/tcsh
setenv HOME /home/operator
sleep 1
date
cd $HOME/tuotanto/RAE
set IRISDIR = IRISdata


if($2 == "POH") then
   ./extproduct_hail_noPOHlim $1 ZERO_level.pgm $HOME/prod_in $2 $3
endif

if($2 == "LHI") then
    ./extproduct_hail_noPOHlim $1 M20_level.pgm $HOME/prod_in $2 $3
endif

if($2 == "POHLARGE") then
#   set GT = `echo $3 | cut -c-10`   
#   gzip -dc levels/"$GT".pgm.gz > zerotmp.pgm
   pnmcut -left 106 -top 115 -right 865 -bottom 1340 ZERO_level.pgm > cut.pgm
   pnmscale -xsize 1628 -ysize 2625 cut.pgm > zerolarge.pgm
   ./extproduct_hail_noPOHlim $1 zerolarge.pgm $HOME/prod_in "$2" $3
endif

if($2 == "LHILARGE") then
#   set GT = `echo $3 | cut -c-10`   
#   gzip -dc levels/"$GT".pgm.gz > zerotmp.pgm
   pnmcut -left 106 -top 115 -right 865 -bottom 1340 M20_level.pgm > M20cut.pgm
   pnmscale -xsize 1628 -ysize 2625 M20cut.pgm > M20large.pgm
   ./extproduct_hail_noPOHlim $1 M20large.pgm $HOME/prod_in "$2" $3
endif

if($2 == "POHSEUTU") then
   set GT = `echo $3 | cut -c-10`   
   gzip -dc levels/"$GT".pgm.gz > zerotmp.pgm
   pnmcut -left 106 -top 115 -right 865 -bottom 1340 zerotmp.pgm > cut.pgm
   pnmscale -xsize 1628 -ysize 2625 cut.pgm > zerolarge.pgm
   pnmcut -left 715 -top 436 -right 912 -bottom 2109 zerolarge.pgm > seututmp.pgm
   pnmscale -xsize 1000 -ysize 800 seututmp.pgm > zeroseutu.pgm
    ./extproduct_hail_noPOHlim $1 zeroseutu.pgm $HOME/prod_in "$2"
endif


sleep 1
 rm -f $1
exit
