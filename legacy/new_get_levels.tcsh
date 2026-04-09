#!/bin/tcsh                           
setenv TZ UTC
date
# exit
             

set LEVELDIR = /usr/fmi/tuotanto/data/RAE/levels
cd $HOME/tuotanto/RAE/modeldata
set DATE = `date +%Y%m%d%H`00
set YY = `echo $DATE | cut -c-4`
set MM = `echo $DATE | cut -c5-6`
set DD = `echo $DATE | cut -c7-8`

foreach LEVEL (ZERO M20 925)
  if("$LEVEL" == "ZERO") then
#    set MODELFILE = hirlam_zerolevel_stere_hohti.txt
    set MODELFILE = meps_zerolevel_stere_radar.txt
    set ARCDIR = height_T0
  endif

  if("$LEVEL" == "M20") then
#    set MODELFILE = hirlam_M20_level_stere_hohti.txt
    set MODELFILE = meps_M20_level_stere_radar.txt
    set ARCDIR = height_T-20
  endif

  if("$LEVEL" == "925") then
    set MODELFILE = meps_925_Z_T_radar.fqd
  endif

  set MODELPATH = /mnt/meru/data/prod/radman/mallidata/$MODELFILE

  cmp -s $MODELFILE $MODELPATH
  if($? != 0) then
     cp $MODELPATH .
     date
     if("$LEVEL" == "925") then     
        set FQDFILE = "$MODELPATH:h"/"$DATE"_meps_"$LEVEL"_ZT.fqd
        cp $MODELPATH $FQDFILE
        # ssh radman@radprod cp $FQDFILE /arch/radar/storage/$YY/$MM/$DD/fmi/model/levels/P925_fqd/ &
        find $MODELPATH:h -name '*_hirlam_925_ZT.fqd' -mtime +30 -exec rm -f {} \;
        exit
     else
        echo "Got new $LEVEL fqd"
        ../.extract_Tlevels_fromtext $MODELFILE 2 $LEVEL
     endif
  endif

  cp "$DATE"_"$LEVEL"_level.pgm ../"$LEVEL"_level.pgm
  set DATGZ = $LEVELDIR/"$DATE"_"$LEVEL"_levels.dat.gz
  gzip -c "$DATE"_"$LEVEL"_levels.dat > $DATGZ
  set PGMGZ = $LEVELDIR/"$DATE"_"$LEVEL"_level.pgm.gz
  gzip -c "$DATE"_"$LEVEL"_level.pgm > $PGMGZ
  scp $PGMGZ $DATGZ radman@radprod:/arch/radar/storage/$YY/$MM/$DD/fmi/model/levels/$ARCDIR/ &
end



$HOME/tuotanto/pipebin/loopscp "$DATE"_ZERO_heights_per_radar.lis radman@radprod:fmi/prod/run/fmi/radar/iris/vpr/zerolevels/hirlam_zerolevels.latest
$HOME/tuotanto/pipebin/loopssh radman@radprod "cd fmi/prod/run/fmi/radar/iris/vpr/zerolevels/ ; cat hirlam_zerolevels.latest >> hirlam_zerolevels.list" &
$HOME/tuotanto/pipebin/loopscp "$DATE"_ZERO_heights_per_radar.lis fmi@io.elmo:/fmi/data/raw/virpo/RaPu/nollarajat/radar_zeroheights.txt &


# tutkakohtaiset nollarajankorkeudet tutkien haettavaksi
foreach RAD ("KOR" "VIH" "ANJ" "KES" "KAN" "PET" "KUO" "VIM" "NUR" "UTA" "LUO" "KAU")
   cp "$DATE"_"$RAD"_ZEROHEIGHT.txt ../"$RAD"_ZEROHEIGHT.txt
   set SITE = `echo $RAD | tr '[:upper:]' '[:lower:]'`
   scp -B  ../"$RAD"_ZEROHEIGHT.txt radarop@"$SITE"rvp.fmi.fi:/tmp/
   scp ../"$RAD"_ZEROHEIGHT.txt radarop@iris-b.fmi.fi:/tmp
end
find . -name '*.*' -mtime +1 -exec rm {} \;

exit
