reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 26
set output 'plots/binder.eps'
set ylabel 'Q'
set xlabel 'T'
set xtics 0.2 font "Times, 20"
set ytics 0.2 font "Times, 20"
set title '(PT) Ising 3D Random Field Binder Factor'
set key box linestyle 1 lc rgb "black"
set key Left left samplen 0.1 reverse  
set key spacing 0.6 font "Times, 16" 
set pointsize 0.8
set style line 1 lw 1.0 lc rgb "forest-green" lt 1
set style line 2 lw 1.0 lc rgb "tan1" lt 1

plot	'data/binder.dat' using 1:2 notitle	with linespoints pt 7 lt 3 lc rgb "black",\
		'data/binder.dat' using 1:2:3 notitle linestyle 1 with errorbars

#plot	'data/binder.dat' using 1:(0.5*(3.0-$2)) notitle	with linespoints pt 7 lt 3 lc rgb "black",\
#		'data/binder.dat' using 1:(0.5*(3.0-$2)):3 notitle linestyle 1 with errorbars
