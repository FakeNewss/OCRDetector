declare -i dnase_enrichment=-200
while ((dnase_enrichment<1000))
do
  let dnase_enrichment+=200
  echo "____________________________"
  echo "DNase-$1:$dnase_enrichment"
  dnase_overlap_cnt=$(bedtools intersect -a ./$1/Dnase_seq.$1.bed -b ./$2 -wa | awk -v dnase_enrichment="$dnase_enrichment" '$5 >= dnase_enrichment' |sort | uniq | wc -l)
  dnase_all_cnt=$(less ./$1/Dnase_seq.$1.bed | awk -v dnase_enrichment="$dnase_enrichment" '$1 == 1 && $5 >= dnase_enrichment' | sort | uniq |wc -l)
  echo $dnase_overlap_cnt"/"$dnase_all_cnt
  echo "scale=3;$dnase_overlap_cnt/$dnase_all_cnt"|bc
  echo "____________________________"
done
