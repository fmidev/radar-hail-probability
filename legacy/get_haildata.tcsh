#!/bin/tcsh
cd ~/tuotanto/focus
set INDEX = $1
set SHARED = /mnt/meru/data/prod/radman/latest/fmi/radar/iris/composite/TOPS

if("$INDEX" == "POH") then
    set OLD_ZERO = `cat zerofile.old`
    set LAST_ZERO = `ls -1t $SHARED/*_ZEROLARGE.iris | head -1`
    if("$LAST_ZERO" != "$OLD_ZERO") then
       cp $LAST_ZERO ZERO.iris
       echo $LAST_ZERO > zerofile.old
    endif

    set LAST_TOP45 = `ls -1t $SHARED/*_45L.iris | head -1`
    echo $LAST_TOP45 >> log/TOP45.log
    cp $LAST_TOP45 TOP45.iris

    bin/irishails TOP45.iris ZERO.iris latest_HHI.iris HHI
    bin/irishails TOP45.iris ZERO.iris latest_THI.iris THI

    mv latest_HHI.iris latest_THI.iris ~/prod_in
endif

if("$INDEX" == "LHI") then
    set OLD_M20 = `cat m20file.old`
    set LAST_M20 = `ls -1t $SHARED/*_M20LARGE.iris | head -1`
    if("$LAST_M20" != "$OLD_M20") then
       cp $LAST_M20 M20.iris
       echo $LAST_M20 > m20file.old
    endif

    set LAST_TOP50 = `ls -1t $SHARED/*_50L.iris | head -1`
    echo $LAST_TOP50 >> log/TOP50.log
    cp $LAST_TOP50 TOP50.iris

    bin/irishails TOP50.iris M20.iris latest_LHI.iris LHI
    mv latest_LHI.iris ~/prod_in
endif



