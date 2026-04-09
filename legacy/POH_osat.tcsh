#!/bin/tcsh
# set LATEST_IRIS = "/mnt/meru/data/prod/radman/latest/fmi/radar/iris"
set WIDTH = 900
set HEIGHT = 800
foreach AREA ("NC" "NW" "NE" "CW" "CC" "CE" "SW" "SE" "SC")
  switch("$AREA")
    case "NC":
       set LEFT = 470
       set TOP = 0
    breaksw
    case "NW":
       set LEFT = 150
       set TOP = 700
    breaksw
    case "NE":
       set LEFT = 700
       set TOP = 700
    breaksw
    case "CW":
       set LEFT = 0
       set TOP = 1200
    breaksw
    case "CC":
       set LEFT = 400
       set TOP = 1200
    breaksw
    case "CE":
       set LEFT = 700
       set TOP = 1200
    breaksw
    case "SW":
       set LEFT = 0
       set TOP = 1700
    breaksw
    case "SE":
       set LEFT = 700
       set TOP = 1700
    breaksw
    case "SC":
       set LEFT = 350
       set TOP = 1700
    breaksw
  endsw

  pnmcut -left $LEFT -top $TOP -width $WIDTH -height $HEIGHT POHKARTTA.ppm > POH_"$AREA".ppm
  ppmtogif POH_"$AREA".ppm  > POH_"$AREA".gif
  pnmcut -left $LEFT -top $TOP -width $WIDTH -height $HEIGHT HHIKARTTA.ppm > HHI_"$AREA".ppm
  ppmtogif HHI_"$AREA".ppm  > HHI_"$AREA".gif
  pnmcut -left $LEFT -top $TOP -width $WIDTH -height $HEIGHT THIKARTTA.ppm > THI_"$AREA".ppm
  ppmtogif THI_"$AREA".ppm  > THI_"$AREA".gif
  cp POH_"$AREA".gif  $LATEST_IRIS/composite/POH/"$1"_fmi.radar.iris.composite.POH_FIN_POH-"$AREA".gif &   
  cp HHI_"$AREA".gif  $LATEST_IRIS/composite/HHI/"$1"_fmi.radar.iris.composite.HHI_FIN_HHI-"$AREA".gif &   
  cp THI_"$AREA".gif  $LATEST_IRIS/composite/THI/"$1"_fmi.radar.iris.composite.THI_FIN_THI-"$AREA".gif &   
end

exit
