# OCRDetector 
a novel bioinformatics pipeline named OCRDetector for detecting OCRs of the whole genome based on cfDNA.  

The original name is OCRDetector

email- chenliubin@stu.xjtu.edu.cn  

Usage:   python OCRDetectBycfDNA.py [-h usage] [-i input file] [-o OCRs output bed]

Example: python OCRDetectBycfDNA.py -i bamFileList.txt -o OCRs.bed

Options:

​	 -h: usage

​	-i: bam file list of cfDNA

​	-o: output file of detected OCRs

First, you need to replace the path in bamFileList.txt with your bam file path.
In addition, you can use bedtools to calculate the intersection of our OCRs and OCRs obtained from other data (ATAC-seq, Dnase-seq, TSS).
Instructions will be updated ...  
