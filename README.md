
# Query-guided End-to-End-Person-Search
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

## Results
 Method @Gallery100        | mAP           | top-1  |
| -------------            |:-------------:| -----: |
| OIM                      | 75.5          | 78.7   |
| Mask-G                   | 83.0          |   83.7  |
| QEEPS                    | 88.9          |   89.1  |

## PRW-mini
The task of query dependent person search encompasses complexity of O(MN) during benchmarking (M queries, N gallery images). PRW dataset has 2,057 probes and 6,112 gallery images. This means, conditioning on the query requires jointly processing each [query-gallery] pair and the exhaustive evaluation of the product space, i.e. 2, 057 × 6, 112. We introduce the PRW-mini to reduce the evaluation time while maintaining the difficulty. PRW-mini tests 30 query images against the whole gallery. The 30 probes selected for PRW-mini are given in prwmini_query_info.txt 


## PRW Evaluation 
We evaluate on PRW and PRW-mini with the same script as adopted by Mask-G (prw_test.py and prwmini_test.py respectively). This evaluation is motivated from CUHK-SYSU  https://github.com/ShuangLI59/person_search/tree/master/lib/datasets. Each probe image is compared to whole gallery except the probe image itself.  If you are using this evaluation for PRW please cite:

@inproceedings{munjal2019cvpr,
author = {Munjal, Bharti and Amin, Sikandar and Tombari, Federico and Galasso, Fabio},
title = {Query-guided End-to-End Person Search},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}

@inproceedings{Chen_2018_ECCV,
author = {Chen, Di and Zhang, Shanshan and Ouyang, Wanli and Yang, Jian and Tai, Ying},
title = {Person Search via A Mask-guided Two-stream CNN Model},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2018}
}

@inproceedings{xiao2017joint,
  title={Joint detection and identification feature learning for person search},
  author={Xiao, Tong and Li, Shuang and Wang, Bochao and Lin, Liang and Wang, Xiaogang},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}

@inproceedings{zheng2016prw,
author = {Zheng, Liang and Zhang, Hengheng and Sun, Shaoyan and Chandraker, Manmohan and Yang, Yi and Tian, Qi},
title = {Person Re-Identification in the Wild},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2017}
}


## RPN vs QRPN Proposals
Here, we compare the output region proposals of RPN and QRPN. Given a query-gallery image pair, in the first column we show the query images with the person bounding boxes (in yellow). In the second and third columns we illustrate the top 10 region proposals in the gallery image by RPN and QRPN, respectively.

QRPN utilizes the query person features to rank the relevant person proposals higher than the irrelevant ones. Therefore, in the following figure, we notice the output proposals of QRPN are concentrated largely around the target person. On the other hand, the output proposals of the RPN are distributed over different people and also the background, which results in false positives for the person search task.


![RPN Vs QRPN](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search/blob/master/RPNVsQRPNFig1.JPG)

In the following examples, we highlight the effectiveness of QRPN in some challenging examples. For instance, in first and second rows, the query persons are wearing black clothes, which urges QRPN to select all people wearing black in the gallery images. Similarly, in the third row the QRPN focuses on two people wearing dark suits. This phenomenon can also be observed in the fourth row, where QRPN gives a few proposals on the persons wearing a similar shade of color as the target person. These results indicate the high importance of color features in the QRPN decisions. However, note that, in all these examples, the QRPN proposals are highly concentrated on the target person, unlike the RPN proposals.

![RPN Vs QRPN](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search/blob/master/RPNVsQRPNFig2.JPG)

In the following two examples, we see that RPN fails to capture the target persons in its top 10 proposals, due to scale and contrast challenges, respectively. However, QRPN is able retrieve multiple proposals for them, demonstrating the advantage of query-guided search.

![RPN Vs QRPN](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search/blob/master/RPNVsQRPNFig3.JPG)



