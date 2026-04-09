#/bin/tcsh
PRODCONF=$1

switch("$PRODCONF")

  case "CORECOMP":
     set REMOTE_RUN = 0
     set TEMPTOP = $TEMPPROD:t
     cp $TEMPPROD $PROD_HOME/RAE/
     pushd $PROD_HOME/RAE
     echo "BEGIN "$INGTIME >> POH.log
     if("$MINS" == "00") then
        ./generate_POH $TEMPTOP POH ZEROHEIGHT >>& POH.log &
     else
        ./generate_POH $TEMPTOP POH >>& POH.log &
     endif
     popd
  breaksw

  case "COREC50":
     set REMOTE_RUN = 0
     set TEMPTOP = $TEMPPROD:t
     cp $TEMPPROD $PROD_HOME/RAE/
     pushd $PROD_HOME/RAE
     echo "BEGIN "$INGTIME >> M20.log
     if("$MINS" == "00") then
        ./generate_POH $TEMPTOP LHI M20HEIGHT >>& M20.log &
     endif
     popd
  breaksw

  case "CORECOMP_L":
     set REMOTE_RUN = 0
     set TEMPTOP = $TEMPPROD:t
     cp $TEMPPROD $PROD_HOME/RAE/
     pushd $PROD_HOME/RAE
     echo "BEGIN "$INGTIME >> POHlarge.log
     set ISOTH = ""
     if("$MINS" == "00") set ISOTH = ZEROLARGE 
     ./generate_POH $TEMPTOP POHLARGE $ISOTH >>& POHlarge.log &
     cp $TEMPPROD /mnt/meru/data/prod/radman/latest/fmi/radar/iris/composite/TOPS/"$INGTIME"_fmi.radar.iris.composite.TOPS_FIN_45L.iris
     ssh operator@hagrid.fmi.fi tuotanto/focus/get_haildata.tcsh POH &
     popd
  breaksw

  # Suurten rakeiden indeksi
  case "COREC50_L":
     set REMOTE_RUN = 0
     set TEMPTOP = $TEMPPROD:t
     cp $TEMPPROD $PROD_HOME/RAE/
     pushd $PROD_HOME/RAE
     echo "BEGIN "$INGTIME >> LHIlarge.log
     set ISOTH = ""
     if("$MINS" == "00") set ISOTH = M20LARGE 
     ./generate_POH $TEMPTOP LHILARGE $ISOTH >>& LHIlarge.log &
     cp $TEMPPROD /mnt/meru/data/prod/radman/latest/fmi/radar/iris/composite/TOPS/"$INGTIME"_fmi.radar.iris.composite.TOPS_FIN_50L.iris
     ssh operator@hagrid.fmi.fi tuotanto/focus/get_haildata.tcsh LHI &
     popd
  breaksw

  case "POHLARGE":
    set REMOTE_RUN = 0
    set POHDATA = "$INGTIME"_"$PRODCONF".dat
    cp $TEMPPROD $PROD_HOME/RAE/$POHDATA
    pushd $PROD_HOME/RAE
    ./visualize_POH_data $POHDATA $INGTIME $PRODCONF
    popd
  breaksw

  case "LHILARGE":
    set REMOTE_RUN = 0
    set LHIDATA = "$INGTIME"_"$PRODCONF".dat
    cp $TEMPPROD $PROD_HOME/RAE/$LHIDATA
    pushd $PROD_HOME/RAE
    ./visualize_LHI_data $LHIDATA $INGTIME $PRODCONF
    popd
  breaksw

  case "ZEROHEIGHT":
     cp $TEMPPROD /mnt/meru/data/prod/radman/latest/fmi/radar/iris/composite/TOPS/"$INGTIME"_fmi.radar.iris.composite.TOPS_FIN_ZEROHEIGHT.iris
  breaksw

  case "ZEROLARGE":
     cp $TEMPPROD /mnt/meru/data/prod/radman/latest/fmi/radar/iris/composite/TOPS/"$INGTIME"_fmi.radar.iris.composite.TOPS_FIN_ZEROLARGE.iris
  breaksw

  case "M20HEIGHT":
     cp $TEMPPROD /mnt/meru/data/prod/radman/latest/fmi/radar/iris/composite/TOPS/"$INGTIME"_fmi.radar.iris.composite.TOPS_FIN_M20HEIGHT.iris
  breaksw

  case "M20LARGE":
     cp $TEMPPROD /mnt/meru/data/prod/radman/latest/fmi/radar/iris/composite/TOPS/"$INGTIME"_fmi.radar.iris.composite.TOPS_FIN_M20LARGE.iris
  breaksw

endsw