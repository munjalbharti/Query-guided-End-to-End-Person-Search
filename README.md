
# Query-guided End-to-End-Person-Search

Paper Link: https://arxiv.org/abs/1905.01203

If you are referring this work please cite:

@inproceedings{munjal2019cvpr,
author = {Munjal, Bharti and Amin, Sikandar and Tombari, Federico and Galasso, Fabio},
title = {Query-guided End-to-End Person Search},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}

## Abstract
Person search has recently gained attention as the novel task of finding a person, provided as a cropped sample, from a gallery of non-cropped images, whereby several other people are also visible. We believe that i. person detection and re-identification should be pursued in a joint optimization framework and that ii. the person search should leverage the query image extensively (e.g. emphasizing unique query patterns). However, so far, no prior art realizes this. We introduce a novel query-guided end-to-end person search network (QEEPS) to address both aspects. We leverage a most recent joint detector and re-identification work, OIM. We extend this with i. a query-guided Siamese squeeze-and-excitation network (QSSE-Net) that uses global context from both the query and gallery images, ii. a query-guided region proposal network (QRPN) to produce query-relevant proposals, and iii. a query-guided similarity subnetwork (QSimNet), to learn a query-guided reidentification score. QEEPS is the first end-to-end queryguided detection and re-id network. On both the most recent CUHK-SYSU and PRW datasets, we outperform the previous state-of-the-art by a large margin.

![Query Guided Person Search](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search/blob/master/Network.JPG)

## Results on CUHK-SYSU [3]
 Method @Gallery100        | mAP           | top-1  |
| -------------            |:-------------:| -----: |
| Mask-G[2]                   | 83.0          |   83.7  |
| QEEPS                    | 88.9          |   89.1  |

## Results on PRW [4]
 Method                    | mAP           | top-1  |
| -------------            |:-------------:| -----: |
| Mask-G[2]                   | 32.6        |  72.1  |
| QEEPS                    | 37.1         |   76.7  |


## PRW-mini
The task of query dependent person search encompasses complexity of O(MN) during benchmarking (M queries, N gallery images). PRW dataset has 2,057 probes and 6,112 gallery images. This means, conditioning on the query requires jointly processing each [query-gallery] pair and the exhaustive evaluation of the product space, i.e. 2, 057 × 6, 112. We introduce the PRW-mini to reduce the evaluation time while maintaining the difficulty. PRW-mini tests 30 query images against the whole gallery. The 30 probes selected for PRW-mini are given in prwmini_query_info.txt 

## Results on PRW-mini [1]
 Method                    | mAP           | top-1  |
| -------------            |:-------------:| -----: |
| Mask-G[2]                   | 33.1        |  70.0  |
| QEEPS                    | 39.1         |   80.0  |



## PRW Evaluation 
We evaluate on PRW with the same evaluation script (prw_test.py) as adopted by Mask-G [2]. This evaluation is motivated from CUHK-SYSU https://github.com/ShuangLI59/person_search/tree/master/lib/datasets. Each probe image is compared to all gallery images except the probe image itself. We also provide here the script for PRW-mini evaluation (prwmini_test.py). If you are using this evaluation for PRW please cite:

[1] B. Munjal, S. Amin, F. Tombari, F. Galasso. Query-guided End-to-End Person Search. In The IEEE Conference on Computer Vision and     Pattern Recognition (CVPR), 2019 
   
[2] D. Chen, S. Zhang, W. Ouyang, J. Yang, and Y. Tai. Person search via a mask-guided two-stream cnn model. In The European Conference on Computer Vision (ECCV), 2018

[3] T. Xiao, S. Li, B. Wang, L. Lin, and X. Wang. Joint detection and identification feature learning for person search. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[4] L. Zheng, H. Zhang, S. Sun, M. Chandraker, Y. Yang, and Q. Tian. Person re-identification in the wild. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.


## RPN vs QRPN Proposals
Here, we compare the output region proposals of RPN and QRPN. Given a query-gallery image pair, in the first column we show the query images with the person bounding boxes (in yellow). In the second and third columns we illustrate the top 10 region proposals in the gallery image by RPN and QRPN, respectively.

QRPN utilizes the query person features to rank the relevant person proposals higher than the irrelevant ones. Therefore, in the following figure, we notice the output proposals of QRPN are concentrated largely around the target person. On the other hand, the output proposals of the RPN are distributed over different people and also the background, which results in false positives for the person search task.


[RPN Vs QRPN](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search/blob/master/QRPNExamples.jpg)

In the following examples, we highlight the effectiveness of QRPN in some challenging examples. For instance, in first and second rows, the query persons are wearing black clothes, which urges QRPN to select all people wearing black in the gallery images. Similarly, in the third row the QRPN focuses on two people wearing dark suits. This phenomenon can also be observed in the fourth row, where QRPN gives a few proposals on the persons wearing a similar shade of color as the target person. These results indicate the high importance of color features in the QRPN decisions. However, note that, in all these examples, the QRPN proposals are highly concentrated on the target person, unlike the RPN proposals.

![RPN Vs QRPN](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search/blob/master/RPNVsQRPNFig2.JPG)

In the following two examples, we see that RPN fails to capture the target persons in its top 10 proposals, due to scale and contrast challenges, respectively. However, QRPN is able retrieve multiple proposals for them, demonstrating the advantage of query-guided search.

![RPN Vs QRPN](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search/blob/master/RPNVsQRPNFig3.JPG)



