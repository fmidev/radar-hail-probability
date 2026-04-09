#!/bin/tcsh
setenv HOME /home/operator
set TIMESTAMP = $2
cd $HOME/tuotanto/RAE
set PRODDIR = /usr/fmi/tuotanto/data
set DATADIR = $PRODDIR/RAE
set ARCDIR = /mnt/meru/data/prod/radman/raevarasto
setenv LATEST_IRIS  /mnt/meru/data/prod/radman/latest/fmi/radar/iris
setenv ROADMASK maps/suomi_500_tiet.pgm
setenv WATERMASK maps/suomi_500_vedet.pgm
setenv NAMEMASK maps/suomi_500_nimet.pgm
setenv COLORFILE maps/POHcolors.txt
setenv COLORMAPFILE maps/LHIcolors_new.ppm

rawtopgm -headerskip 640 1628 2625 $1 > LHILARGE.tmp
pnmflip -tb LHILARGE.tmp > LHILARGE.pgm
./.LHI_map LHILARGE.pgm LHIKARTTA.ppm LHIDATA.ppm

./.generate_latlonbox_flip LHIDATA.ppm LHIGEO.ppm
convert -flip -transparent black LHIGEO.ppm LHI.png
cp LHI.png modeldata/"$TIMESTAMP"_LHI.png
# cp LHI.png $ARCDIR/"$TIMESTAMP"_LHI.png

./generate_animkml.tcsh modeldata LHI http://img.fmi.fi/rapu/kml 12 > LHI_anim.kml
$HOME/tuotanto/pipebin/loopscp LHI.png fmi@io.elmo:/fmi/data/raw/virpo/RaPu/kml_icons/
$HOME/tuotanto/pipebin/loopssh fmi@io.elmo "cd /fmi/data/raw/virpo/RaPu/kml_icons/ ; cp LHI.png "$TIMESTAMP"_LHI.png"
$HOME/tuotanto/pipebin/loopscp LHI_anim.kml fmi@io.elmo:/fmi/data/raw/virpo/RaPu/kml_icons/

# $HOME/tuotanto/pipebin/loopscp LHI.png fmi@io.elmo:/fmi/dev/www/radar.fmi.fi/products/ &


cp LHI.png latest/
set M = 0
foreach LHIPNG (`ls -1 modeldata/*_LHI.png | tail -7`)
  set MIN = $M
  if($M < 10) set MIN = 0"$M"
  cp $LHIPNG latest/LHI_"$MIN".png
  set M = `expr $M + 5`
end

scp latest/LHI*.png fmi@io.elmo:/fmi/dev/www/radar.fmi.fi/products/ &

pnmpaste maps/LHIlegend_new.ppm 1000 2580 LHIKARTTA.ppm > LHIFINAL.ppm
ppmtogif LHIFINAL.ppm >  LHILARGE.gif
# scp LHILARGE.gif  www@ukko:/var/www/radar.fmi.fi/Testituotteet/IRIS/LHILARGE_"$TIMESTAMP".gif &
set STOREPGM = $PRODDIR/WARN_store/daily/"$TIMESTAMP"_"$3".pgm.gz
gzip -c LHILARGE.pgm > $STOREPGM
cp $STOREPGM $LATEST_IRIS/LHI/ &
./LHI_osat.tcsh "$TIMESTAMP"

# LHI smartmetiin
set HDIR = /mnt/meru/data/prod/radman/Hail
cp $1 $HDIR
echo "$1 $2 $3"  >> LHI.log
ssh operator@winky tuotanto/Geoserver/cartesian_geohandler.tcsh $HDIR/"$1:t" $TIMESTAMP LHI FIN >>& LHI.log &

set LATESTGIF = $LATEST_IRIS/LHI/"$TIMESTAMP"_fmi.radar.iris.LHI_FIN_LHILARGE.gif
cp LHILARGE.gif $LATESTGIF

# ssh radman@ravake.fmi.fi Hailmax/gen_single_hailindex.tcsh $LATESTGIF >& latestLHIgif.log & 
ssh radman@esteri.fmi.fi Hailmax/gen_single_hailindex.tcsh $LATESTGIF >& latestLHIgif_esteri.log & 


rm -f $1
exit
