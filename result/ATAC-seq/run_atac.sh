declare -i atac_enrichment=-20
for atac_enrichment in 0 20 50 100 200 300 500
do
  echo "ATAC_$1:$atac_enrichment"
  atac_overlap_cnt=$(bedtools intersect -a ./$1/ATAC.$1.bed -b ./$2 -wa | awk -v atac_enrichment="$atac_enrichment" '$5 >= atac_enrichment' | sort | uniq | wc -l)
  atac_all_cnt=$(less ./$1/ATAC.$1.bed | awk -v atac_enrichment="$atac_enrichment" '$1 == 1 && $5 >= atac_enrichment' | sort | uniq |wc -l)
  echo $atac_overlap_cnt"/"$atac_all_cnt
  echo "scale=3;$atac_overlap_cnt/$atac_all_cnt"|bc
  echo "ATAC_$1:$atac_enrichment"
  echo "_______________________"
done
