## Sort files based on 1st field with tab separator
for file in dpd_postpruned_sizes/*
do
    echo $file
    sort -k 1 -t $'\t' -g $file | cut -f2 -d$'\t' > $file.sort
done

mv dpd_postpruned_sizes/*.sort dpd_postpruned_sorted/

## Remove file extension
cd dpd_postpruned_sorted
for file in *
do
      cat $file >  ${file%.*.*}
done

rm *.sort

## Split the file

## Parameters for bag generation
NUM_BAGS=3
LFS_PER_BAG=100
# For panini1
SORTED_FP=/analytics/shivanshu/dpd/miml/dpd_postpruned_sorted/
# For HPC
# SORTED_FP=/scratch/cse/dual/cs5130298/dpd/miml/dpd_postpruned_sorted/

# lines per file
linesFile=$(($NUM_BAGS*$LFS_PER_BAG))
# adding the pos and neg case
totalLines=$((2*$linesFile))
# half lines if file has insufficient
halfLines=$(($linesFile/2))
# bags with insufficient num of lfs
insCnt=0

## Positive Bags
for file in $SORTED_FP/*
do
    echo $file
    cp $file .
    ## Find length of file
    numLines=$(cat $file | wc -l)

    if [[ $numLines -gt $totalLines ]]; then
        head -n $linesFile $file | sort -R | split -d -l $LFS_PER_BAG - $(basename $file)_p
    else
        insCnt=$(($insCnt+1))
        head -n $halfLines $file | sort -R | split -d -l $LFS_PER_BAG - $(basename $file)_p
    fi
    # head -n 50 $file > $(basename $file)_p
done


## Negative Bags
for file in $SORTED_FP/*
do
    echo $file
    ## Find length of file
    numLines=$(cat $file | wc -l)

    if [[ $numLines -gt $totalLines ]]; then
        tail -n $linesFile $file | sort -R | split -d -l $LFS_PER_BAG - $(basename $file)_n
    else
        tail -n $halfLines $file | sort -R | split -d -l $LFS_PER_BAG - $(basename $file)_n
    fi

    #head -n 100 $file | tail -n 50 > $(basename $file)_n
done

gzip *

echo "Number of files with insufficient number of logical forms = "$insCnt


