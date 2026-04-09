#!/bin/tcsh
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

  pnmcut -left $LEFT -top $TOP -width $WIDTH -height $HEIGHT LHIFINAL.ppm > LHI_"$AREA".ppm
  ppmtogif LHI_"$AREA".ppm  > LHI_"$AREA".gif
  $HOME/tuotanto/pipebin/loopscp LHI_"$AREA".gif  radman@radprod:fmi/data/latest/fmi/radar/iris/composite/LHI/"$1"_fmi.radar.iris.composite.LHI_FIN_LHI-"$AREA".gif &   
end

exit
