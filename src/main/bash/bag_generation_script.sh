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

## Positive Bags
cd /scratch/cse/dual/cs5130298/dpd/miml/dpd_bags_pos_neg/dpd_output
for file in /scratch/cse/dual/cs5130298/dpd/miml/dpd_postpruned_sorted/*
do
    echo $file
    #head -n 50 $file | sort -R | split -d -l 50 - $(basename $file)_p
    head -n 50 $file > $(basename $file)_p
done
gzip *_p*

## Negative Bags

for file in /scratch/cse/dual/cs5130298/dpd/miml/dpd_postpruned_sorted/*
do
    echo $file
    #tail -n 50 $file | sort -R | split -d -l 50 - $(basename $file)_n
    head -n 100 $file | tail -n 50 > $(basename $file)_n
done
gzip *_n*


