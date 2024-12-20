D=("citeseer" "cora" "pubmed" "fb15k237" "wn18" "mutag" "collab" "imdbbinary" "imdbmulti" "nci1" "proteins" "ptc" "redditbinary" "redditmulti5k")
# D=("mutag" "collab" "imdbbinary" "imdbmulti" "nci1" "proteins" "ptc" "redditbinary" "redditmulti5k")

for d in ${D[*]}
do
echo "# test fractal for $d"
python test_fractal.py --data $d
done