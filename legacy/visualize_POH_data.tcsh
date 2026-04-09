#!/bin/tcsh
setenv HOME /home/operator
set TIMESTAMP = $2
set YY = `echo $TIMESTAMP | cut -c-4`
set mm = `echo $TIMESTAMP | cut -c5-6`
set dd = `echo $TIMESTAMP | cut -c7-8`
set YMD = "$YY"/"$mm"/"$dd"
cd $HOME/tuotanto/RAE
set PRODDIR = /usr/fmi/tuotanto/data
set DATADIR = $PRODDIR/RAE
set ARCDIR = /mnt/meru/data/prod/radman/raevarasto
set STORDIR = /arch/radar/storage/$YMD/fmi/radar/iris/POH

setenv LATEST_IRIS  /mnt/meru/data/prod/radman/latest/fmi/radar/iris
setenv ROADMASK maps/suomi_500_tiet.pgm
setenv WATERMASK maps/suomi_500_vedet.pgm
setenv NAMEMASK maps/suomi_500_nimet.pgm
setenv COLORFILE maps/POHcolors.txt
setenv COLORMAPFILE maps/RAEcolors.ppm
setenv POH_MAX 1.5

rawtopgm -headerskip 640 1628 2625 $1 > POHLARGE.tmp
pnmflip -tb POHLARGE.tmp > POHLARGE.pgm
./.RAE_map POHLARGE.pgm POHKARTTA.ppm
ppmtogif POHKARTTA.ppm >  POHLARGE.gif
# $HOME/tuotanto/pipebin/loopscp POHLARGE.gif  radman@radprod:fmi/data/latest/fmi/radar/iris/POH/"$TIMESTAMP"_fmi.radar.iris.POH_FIN_POHLARGE.gif &
cp POHLARGE.gif $LATEST_IRIS/POH/"$TIMESTAMP"_fmi.radar.iris.POH_FIN_POHLARGE.gif &

setenv COLORMAPFILE maps/HHIlegend.ppm
setenv POH_MAX 1.5
./.RAE_map_index POHLARGE.pgm HHIKARTTA.ppm HHIDATA.ppm
 ./.generate_latlonbox_flip HHIDATA.ppm HHIGEO.ppm
convert -flip -transparent black HHIGEO.ppm HHI.png
cp HHI.png modeldata/"$TIMESTAMP"_HHI.png
# cp HHI.png $ARCDIR//"$TIMESTAMP"_HHI.png

# HHI index tuned by Zero isotherm height (Jari Tuovinen)
pamflip -tb zerolarge.pgm > zeroflip.pgm
./.RAE_map_modindex POHLARGE.pgm THIKARTTA.ppm THIDATA.ppm zeroflip.pgm
pnmpaste $COLORMAPFILE 1000 2580 THIKARTTA.ppm > THIFINAL.ppm
ppmtogif THIFINAL.ppm >  THILARGE.gif
# cp THILARGE.gif  $LATEST_IRIS/THI/"$TIMESTAMP"_fmi.radar.iris.THI_FIN_THILARGE.gif &

./generate_animkml.tcsh modeldata HHI http://img.fmi.fi/rapu/kml 12 > HHI_anim.kml 
$HOME/tuotanto/pipebin/loopscp HHI.png fmi@io.elmo:/fmi/data/raw/virpo/RaPu/kml_icons/
$HOME/tuotanto/pipebin/loopssh fmi@io.elmo "cd /fmi/data/raw/virpo/RaPu/kml_icons/ ; cp HHI.png "$TIMESTAMP"_HHI.png"
$HOME/tuotanto/pipebin/loopscp HHI_anim.kml fmi@io.elmo:/fmi/data/raw/virpo/RaPu/kml_icons/


cp HHI.png latest/
set M = 0
foreach HHIPNG (`ls -1 modeldata/*_HHI.png | tail -7`)
  set MIN = $M
  if($M < 10) set MIN = 0"$M"
  cp $HHIPNG latest/HHI_"$MIN".png
  set M = `expr $M + 5`
end

scp latest/HHI*.png fmi@io.elmo:/fmi/dev/www/radar.fmi.fi/products/ &
pnmpaste $COLORMAPFILE 1000 2580 HHIKARTTA.ppm > HHIFINAL.ppm
ppmtogif HHIFINAL.ppm >  HHILARGE.gif
set STOREPGM = $PRODDIR/WARN_store/daily/"$TIMESTAMP"_"$3".pgm.gz
gzip -c POHLARGE.pgm > $STOREPGM
cp $STOREPGM $LATEST_IRIS/POH/ &
# scp $STOREGPGM radman@radprod.fmi.fi:"$STORDIR" &
./POH_osat.tcsh "$TIMESTAMP"

# HHI smartmetiin
set HDIR = /mnt/meru/data/prod/radman/Hail
cp $1 $HDIR
echo "$1 $2 $3"  >> HHI.log
ssh operator@winky tuotanto/Geoserver/cartesian_geohandler.tcsh $HDIR/"$1:t" $TIMESTAMP HHI FIN >>& HHI.log & 

set LATESTGIF = $LATEST_IRIS/HHI/"$TIMESTAMP"_fmi.radar.iris.HHI_FIN_HHILARGE.gif
cp HHILARGE.gif $LATESTGIF
# ssh radman@ravake.fmi.fi Hailmax/gen_single_hailindex.tcsh $LATESTGIF >& latestHHIgif.log &
ssh radman@esteri.fmi.fi Hailmax/gen_single_hailindex.tcsh $LATESTGIF >& latestHHIgif_esteri.log &

set LATESTGIF = $LATEST_IRIS/THI/"$TIMESTAMP"_fmi.radar.iris.THI_FIN_THILARGE.gif
cp THILARGE.gif $LATESTGIF
scp POHLARGE.pgm radman@esteri.fmi.fi:Hailmax/
# ssh radman@ravake.fmi.fi Hailmax/gen_single_hailindex.tcsh $LATESTGIF >& latestTHIgif.log &
ssh radman@esteri.fmi.fi Hailmax/gen_single_hailindex.tcsh $LATESTGIF >& latestTHIgif_esteri.log &

rm $1
ssh fmi@io.elmo "find /fmi/data/raw/virpo/RaPu/kml_icons/ -name '*_?HI.png' -mmin +80 -exec rm {} \;"
exit

