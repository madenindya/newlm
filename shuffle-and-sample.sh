DATA_DIR=examples/data
RATIO=$1

echo "SAMPLING TO $RATIO PERCENT"
mkdir $DATA_DIR/text/en.$RATIO-percent
OUTFILE=$DATA_DIR/text/en.$RATIO-percent/text.txt

# compute the actual documents needed.
TOTAL_DOCS=5216027
N=$((RATIO*TOTAL_DOCS/100))

echo "sampling $N documents"

# cat the whole data, then merge documents into a single line
# then, shuffle and select N docs
# resplit the document into lines

cat $DATA_DIR/{books,wikipedia}/* \
	| head -n -1 \
	| awk '{gsub(/\n/, "~x~x~")} 1' RS= \
	| shuf -n $N --random-source=$DATA_DIR/wikipedia/wikipedia.txt-00000-of-00500 \
	| awk '{gsub(/~x~x~/, "\n")} 1' ORS="\n\n" > $OUTFILE

echo "done = $OUTFILE"

