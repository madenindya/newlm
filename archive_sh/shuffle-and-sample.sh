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
# filter by length
# then, shuffle and select N docs
# resplit the document into lines
# remove consequtive blank lines, since a document might be filtered completely.
# remove the last line of the output, since it is a trailing empty spaces.

cat $DATA_DIR/{books,wikipedia}/* \
	| gawk 'NF <= 128 {print $0}' \
	| gawk '{gsub(/\n/, "~x~x~")} 1' RS= \
	| shuf -n $N --random-source=$DATA_DIR/wikipedia/wikipedia.txt-00000-of-00500 \
	| gawk '{gsub(/~x~x~/, "\n")} 1' ORS="\n\n" \
	| cat -s \
	| head -n -1 \
	| iconv -c -t utf8 > $OUTFILE

echo "done = $OUTFILE"

