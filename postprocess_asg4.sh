infile=$1
outfile=$2
lang=$3

cat $infile | sed -r 's/(@@ )|(@@ ?$)//g' | perl /Users/wangqiaowen/atmt/moses_scripts/detruecase.perl | perl /Users/wangqiaowen/atmt/moses_scripts/detokenizer.perl -q -l $lang > $outfile
