##############################################################
# genplots.sh                                                #
#                                                            #
# Author      : Cristobal Navarro <crinavar@dcc.uchile.cl>   #
# Version     : 1.0                                          #
# Date        : September 2015                               #
# Discription : generates gnuplot files                      #
##############################################################

#!/bin/sh
echo -n "generating plots............."
gnuplot scripts/*
echo "ok"
