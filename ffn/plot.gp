# Usar vírgula como separador do arquivo CSV
set datafile separator ","

set terminal gif
set output "mlpack-foo-1.gif"
set view 0, 0
splot "foo.csv"

# Salva uma animação gif
set terminal gif animate delay 5 loop 0 optimize
set output "mlpack-foo-2.gif"

set label "foo.csv" at screen 0.7, 0.9

n = 100
do for [i=1:n] {
   set view 60, i*360/n
   splot "foo.csv" notitle
}

set output

# vim: ft=gnuplot
