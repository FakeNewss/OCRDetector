tissue=$1
less ${tissue}.txt | awk 'NR > 8 {print $1}' | cat | sort -u |while read a ; do  echo awk \'\$5==gene\'  gene=$a GRCh37.gene.bed; done|less|sh > ${tissue}.gene.bed
cat ${tissue}.gene.bed | awk '{if ($4 == "+") print $1"\t"$2 - 1000"\t"$2 + 1000"\t"$4"\t"$5"\t"$6; else print $1"\t"$3 - 1000"\t"$3 + 1000"\t"$4"\t"$5"\t"$6}' | bedtools sort -i |uniq > ${tissue}.gene.TSS.bed
bedtools intersect -b ../ndr_DHSAndTSS.chr10_e.txt -a ${tissue}.gene.TSS.bed -wa | sort -u | wc -l
wc -l ${tissue}.gene.TSS.bed
