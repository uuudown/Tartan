reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 26
set output 'plots/quadm.eps'
set ylabel '<|M|^4>'
#set ylabel rotate right
set xlabel 'T'
set xtics font "Times, 20"
set ytics font "Times, 20"
set title '(PT) Ising 3D Random Field <M^4>'
set key box linestyle 1 lc rgb "black"
set key Left left samplen 0.1 reverse  
set key spacing 0.6 font "Times, 16" 
#set key width -3.0
set pointsize 0.8
set style line 1 lw 1.0 lc rgb "forest-green" lt 1
set style line 2 lw 1.0 lc rgb "tan1" lt 1

plot	'data/quadmagnetization.dat' using 1:2 notitle with linespoints pt 7 lt 3 lc rgb "black",\
		'data/quadmagnetization.dat' using 1:2:7 notitle linestyle 2 with errorbars
