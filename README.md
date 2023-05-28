# Brief history of X-ray security imaging in Computer Vision

<p align="center">
  <img src="images/xray-history.png" />
</p>

*[List of datasets and papers (not exhaustive)]*

## :card_file_box: Dataset
:chart_with_upwards_trend: [2D: 15] [3D: 2]

|Name       | Type | Year | Class |Prohibited - Negative| Annotations| Views|Open Source | 
|-----------|------|------|-------------|-------------|------|-----|------|
|FSOD       |2D    | 2022 |20            |12,333 - 0 | bbox|1     |<span style="color:green;">✓</span> [[Link]](https://github.com/DIG-Beihang/XrayDetection)  |
|EDS       |2D    | 2022 |10            |14,219 - 0 | bbox|1     |<span style="color:green;">✓</span> [[Link]](https://github.com/DIG-Beihang/XrayDetection)  |
|Xray-PI       |2D    | 2022 |12            |2,409 - 0 | bbox, mask|1     |<span style="color:green;">✓</span> [[Link]](https://github.com/LPAIS/Xray-PI)  |
|PIXray       |2D    | 2022 |12            |5,046 - 0 | bbox, mask|1     |<span style="color:green;">✓</span> [[Link]](https://github.com/Mbwslib/DDoAS)  |
|CLCXray       |2D    | 2022 |12            |9,565 - 0 | bbox|1     |<span style="color:green;">✓</span> [[Link]](https://github.com/GreysonPhoenix/CLCXray)  |
|HiXray       |2D    | 2021 |8            |45,364 - 0 | bbox|1     |<span style="color:green;">✓</span> [[Link]](https://github.com/DIG-Beihang/XrayDetection)  |
|deei6       |2D    | 2021 |6            |7,022 - 0 | bbox, mask|2     |<span style="color:red;">✕</span> [[Link]](https://breckon.org/toby/publications/papers/bhowmik21energy.pdf)  |
|PIDray    |2D    | 2021 |12           |47,677 - 0  | bbox, mask |1     |<span style="color:green;">✓</span> [[Link]](https://github.com/bywang2018/security-dataset)       |
|AB       |2D    | 2021 |--            |417 - 6,608 | -- |2     |<span style="color:red;">✕</span> [[Link]](https://ieeexplore.ieee.org/document/9534034)  |
|dbf4       |2D    | 2020 |4            |10,112 - 0 | bbox, mask |4     |<span style="color:red;">✕</span> [[Link]](https://breckon.org/toby/publications/papers/isaac20multiview.pdf)  |
|OPIXray    |2D    | 2020 |5            |8,885  - 0 | bbox |1     |<span style="color:green;">✓</span>  [[Link]](https://github.com/OPIXray-author/OPIXray)           |
|SIXray     |2D    | 2019 |6            |8,929 - 1,050,0302 | bbox |1 |<span style="color:green;">✓</span> [[Link]](https://github.com/MeioJane/SIXray)           |
|COMPASS-XP     |2D    | 2019 |366            |1928 - 0 | -- |1 |<span style="color:green;">✓</span> [[Link]](https://zenodo.org/record/2654887#.YUtGVHVKikA)           |
|dbf6       |2D    | 2018 |6            |11,627 - 0 | bbox, mask |4    |<span style="color:red;">✕</span> [[Link]](https://breckon.org/toby/publications/papers/akcay18architectures.pdf)  |
|GDXray       |2D    | 2015 |5            |19,407 - 0 | bbox |1     |<span style="color:green;">✓</span> [[Link]](https://domingomery.ing.puc.cl/material/gdxray/)  |
|Dur_3D       |3D    | 2020 |5            |774 - 0 | bbox | --   |<span style="color:red;">✕</span> [[Link]](https://arxiv.org/abs/2008.01218)  |
|Flitton_3D       |3D    | 2015 |2        |810 - 2149 | bbox | --   |<span style="color:red;">✕</span> [[Link]](https://breckon.org/toby/publications/papers/flitton15codebooks.pdf)  |

---
## :scroll: Paper 
:chart_with_upwards_trend: [2D: 141] [3D: 38]


### 2023

#### 2D

- Optimization and Research of Suspicious Object Detection Algorithm in X-ray Image [[Link]](https://ieeexplore.ieee.org/document/10082660)
- Object Detection and X-Ray Security Imaging: A Survey [[Link]](https://ieeexplore.ieee.org/abstract/document/10120944)
- RWSC-Fusion: Region-Wise Style-Controlled Fusion Network for the Prohibited X-Ray Security Image Synthesis [[Link]](https://openaccess.thecvf.com/content/CVPR2023/html/Duan_RWSC-Fusion_Region-Wise_Style-Controlled_Fusion_Network_for_the_Prohibited_X-Ray_Security_CVPR_2023_paper.html)
- Transformers for Imbalanced Baggage Threat Recognition [[Link]](https://ieeexplore.ieee.org/document/9977427)
- CTA-FPN: Channel-Target Attention Feature Pyramid Network for Prohibited Object Detection in X-ray Images [[Link]](https://link.springer.com/article/10.1007/s11220-023-00416-7)
- Material-Aware Path Aggregation Network and Shape Decoupled SIoU for X-ray Contraband Detection [[Link]](https://www.mdpi.com/2079-9292/12/5/1179)
- Seeing Through the Data: A Statistical Evaluation of Prohibited Item Detection Benchmark Datasets for X-ray Security Screening [[Link]](https://breckon.org/toby/publications/papers/isaac23evaluation.pdf)
- X-Adv: Physical Adversarial Object Attacks against X-ray Prohibited Item Detection [[Link]](https://arxiv.org/abs/2302.09491)
- Cascaded structure tensor for robust baggage threat detection [[Link]](https://link.springer.com/article/10.1007/s00521-023-08296-4)
- Computer Vision on X-ray Data in Industrial Production and Security Applications: A Comprehensive Survey [[Link]](https://ieeexplore.ieee.org/document/10005308)

### 2022
#### 2D

- Learning-based Material Classiﬁcation in X-ray Security Images [[Link]](https://www.scitepress.org/Papers/2020/89517/pdf/index.html)
- Few-shot X-ray Prohibited Item Detection: A Benchmark and Weak-feature Enhancement Network [[Link]](https://dl.acm.org/doi/abs/10.1145/3503161.3548075)
- Balanced Affinity Loss for Highly Imbalanced Baggage Threat Contour-Driven Instance Segmentation [[Link]](https://ieeexplore.ieee.org/document/9897490)
- Joint Sub-component Level Segmentation and Classification for Anomaly Detection within Dual-Energy X-Ray Security Imagery [[Link]](https://arxiv.org/abs/2210.16453)
- Automatic Baggage Threat Detection Using Deep Attention Networks [[Link]](https://link.springer.com/chapter/10.1007/978-3-030-95070-5_11)
- A Multi-Task Semantic Segmentation Network for Threat Detection in X-Ray Security Images [[Link]](https://ieeexplore.ieee.org/document/9897736)
- Dualray: Dual-View X-ray Security Inspection Benchmark and Fusion Detection Framework [[Link]](https://link.springer.com/chapter/10.1007/978-3-031-18916-6_57#Fig6)
- MFA-net: Object detection for complex X-ray cargo and baggage security imagery [[Link]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0272961)
- Baggage Threat Recognition Using Deep Low-Rank Broad Learning Detector [[Link]](https://ieeexplore.ieee.org/document/9842976)
- Exploring Endogenous Shift for Cross-domain Detection: A Large-scale Benchmark and Perturbation Suppression Network [[Link]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tao_Exploring_Endogenous_Shift_for_Cross-Domain_Detection_A_Large-Scale_Benchmark_and_CVPR_2022_paper.pdf)
- Improved YOLOX detection algorithm for contraband in X-ray images [[Link]](https://opg.optica.org/ao/abstract.cfm?uri=ao-61-21-6297)
- LightRay: Lightweight network for prohibited items detection in X-ray images during security inspection [[Link]](https://www.sciencedirect.com/science/article/pii/S0045790622005110?via%3Dihub)
- Benefits of Decision Support Systems in Relation to Task Difficulty in Airport Security X-Ray Screening [[Link]](https://www.tandfonline.com/doi/full/10.1080/10447318.2022.2107775)
- Recent Advances in Baggage Threat Detection: A Comprehensive and Systematic Survey [[Link]](https://dl.acm.org/doi/10.1145/3549932)
- Automated Detection of Threat Materials in X -Ray Baggage Inspection System (XBIS) [[Link]](https://ieeexplore.ieee.org/document/9795120)
- Threat detection in x-ray baggage security imagery using convolutional neural networks [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12104/121040H/Threat-detection-in-x-ray-baggage-security-imagery-using-convolutional/10.1117/12.2622373.short?SSO=1)
- X-ray baggage screening and artificial intelligence (AI) [[Link]](https://op.europa.eu/en/publication-detail/-/publication/b6e77043-ede4-11ec-a534-01aa75ed71a1/language-en)
- Material-aware Cross-channel Interaction Attention (MCIA) for occluded prohibited item detection [[Link]](https://link.springer.com/article/10.1007/s00371-022-02498-y)
- A Novel Incremental Learning Driven Instance Segmentation Framework to Recognize Highly Cluttered Instances of the Contraband Items [[Link]](https://arxiv.org/abs/2201.02560)
- How Realistic Is Threat Image Projection for X-ray Baggage Screening? [[Link]](https://www.mdpi.com/1424-8220/22/6/2220)
- Cross-modal Image Synthesis within Dual-Energy X-ray Security Imagery [[Link]](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/papers/Isaac-Medina_Cross-Modal_Image_Synthesis_Within_Dual-Energy_X-Ray_Security_Imagery_CVPRW_2022_paper.pdf)
- Recursive CNN Model to Detect Anomaly Detection in X-Ray Security Image [[Link]](https://ieeexplore.ieee.org/abstract/document/9754033?casa_token=Z5JOZgYLA-cAAAAA:8mR-hC0nj2sRu23gI0uZwt0w4K_oHfKcXVnCk6PMWjzmv9YzGxmLGIjDWkdriyqegNf44JRPHZc)
- Towards More Efficient Security Inspection via Deep Learning: A Task-Driven X-ray Image Cropping Scheme [[Link]](https://www.mdpi.com/2072-666X/13/4/565)
- DMA-Net: Dual multi-instance attention network for X-ray image classification [[Link]](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12560)
- X-ray security check image recognition based on attention mechanism [[Link]](https://iopscience.iop.org/article/10.1088/1742-6596/2216/1/012104/meta)
- A Data Augmentation Method for Prohibited Item X-Ray Pseudocolor Images in X-Ray Security Inspection Based on Wasserstein Generative Adversarial Network and Spatial-and-Channel Attention Block [[Link]](https://www.hindawi.com/journals/cin/2022/8172466/)
- Enhanced threat detection in three dimensions: An image-matched comparison of computed tomography and dual-view X-ray baggage screening [[Link]](https://www.sciencedirect.com/science/article/pii/S0003687022001570)
- ETHSeg: An Amodel Instance Segmentation Network and a Real-world Dataset
for X-Ray Waste Inspection [[Link]](https://openaccess.thecvf.com/content/CVPR2022/papers/Qiu_ETHSeg_An_Amodel_Instance_Segmentation_Network_and_a_Real-World_Dataset_CVPR_2022_paper.pdf)
- Anomaly object detection in x-ray images with Gabor convolution and bigger discriminative RoI pooling [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12177/121770B/Anomaly-object-detection-in-x-ray-images-with-Gabor-convolution/10.1117/12.2625815.short)
- Intelligent Detection of Dangerous Goods in Security Inspection Based on Cascade Cross Stage YOLOv3 Model [[Link]](https://hrcak.srce.hr/en/275305)
- A Lightweight Dangerous Liquid Detection Method Based on Depthwise Separable Convolution for X-Ray Security Inspection [[Link]](https://www.hindawi.com/journals/cin/2022/5371350/)
- EAOD-Net: Effective anomaly object detection networks for X-ray images [[Link]](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12514)
- American National Standard for Evaluating the Image Quality of X-ray Computed Tomography (CT) Security-Screening Systems [[Link]](https://ieeexplore.ieee.org/document/9812579)
- Automated Segmentation of Prohibited Items in X-ray Baggage Images Using Dense De-overlap Attention Snake [[Link]](https://ieeexplore.ieee.org/document/9772992)
- Programmable Broad Learning System to Detect Concealed and Imbalanced Baggage Threats [[Link]](https://ieeexplore.ieee.org/document/9787420)
- Few-Shot Segmentation for Prohibited Items Inspection with Patch-based Self-Supervised Learning and Prototype Reverse Validation [[Link]](https://ieeexplore.ieee.org/document/9779459)
- Augmenting data with GANs for firearms detection in cargo x-ray images [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12104/1210406/Augmenting-data-with-GANs-for-firearms-detection-in-cargo-x/10.1117/12.2618887.short?SSO=1)
- Weight-guided dual-direction-fusion feature pyramid network for prohibited item detection in x-ray images [[Link]](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-31/issue-3/033032/Weight-guided-dual-direction-fusion-feature-pyramid-network-for-prohibited/10.1117/1.JEI.31.3.033032.short)
- Handling occlusion in prohibited item detection from X-ray images [[Link]](https://link.springer.com/article/10.1007/s00521-022-07578-7)
- Exploiting foreground and background separation for prohibited item detection in overlapping X-Ray images [[Link]](https://www.sciencedirect.com/science/article/pii/S0031320321004416?casa_token=YKXXjKmLbXUAAAAA:zISg01iB-CP49Ek1rlneL-HftSZnYiA69izOwObqUUM5WdDawLxiSdUePbXI0lq7KF72Wgphfw)
- PMix: a method to improve the classification of X-ray prohibited items based on probability mixing [[Link]](https://www.inderscienceonline.com/doi/pdf/10.1504/IJWMC.2022.123318)
- Detecting prohibited objects with physical size constraint from cluttered X-ray baggage images [[Link]](https://www.sciencedirect.com/science/article/pii/S0950705121010686?casa_token=Y9V5DLzSMw0AAAAA:xyzOIGXtGxAPRaORWuiXWa0E7u2ICS4B1wcZvotPTI9wXzPFEp3IzDuhXKPubRsWX3Mu8q-4AA)
- Synthetic threat injection using digital twin informed augmentation [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12104/1210407/Synthetic-threat-injection-using-digital-twin-informed-augmentation/10.1117/12.2618972.short)
- Target Detection by Target Simulation in X-ray Testing [[Link]](https://link.springer.com/article/10.1007/s10921-022-00851-8)
- Abnormal object detection in x-ray images with self-normalizing channel attention and efficient data augmentation [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12177/121770M/Abnormal-object-detection-in-x-ray-images-with-self-normalizing/10.1117/12.2625843.short)
- Raw data processing techniques for material classification of objects in dual energy X-ray baggage inspection systems [[Link]](https://www.sciencedirect.com/science/article/pii/S0969806X21001626?casa_token=oQnnKwLnYqIAAAAA:cVnclMgMt-24YQEl8dlErCzgMDZlvGC35CF6cddeil5qxxQ0ZhY05P74P0tcBN_M5I4iUk7dSg)

#### 3D
- CTIMS: Automated Defect Detection Framework Using Computed Tomography [[Link]](https://www.mdpi.com/2076-3417/12/4/2175)


### 2021
#### 2D
- Super-resolution network for x-ray security inspection [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12281/2616535/Super-resolution-network-for-x-ray-security-inspection/10.1117/12.2616535.short?SSO=1)
- Information-exchange Enhanced Feature Pyramid Network (IEFPN) for Detecting Prohibited Items in X-ray Security Images [[Link]](https://ieeexplore.ieee.org/document/9674494)
- Learning-Based Image Synthesis for Hazardous Object Detection in X-Ray Security Applications [[Link]](https://ieeexplore.ieee.org/document/9552004)
- X-ray Security Inspection Image Detection Algorithm Based on Improved YOLOv4 [[Link]](https://ieeexplore.ieee.org/document/9645636)
- A YOLOv5s-SE model for object detection in X-ray security images [[Link]](https://ieeexplore.ieee.org/document/9624606)
- Prohibited Items Detection in X-ray Images in YOLO Network [[Link]](https://ieeexplore.ieee.org/document/9594145)
- Automatic and Robust Object Detection in X-Ray Baggage Inspection Using Deep Convolutional Neural Networks [[Link]](https://ieeexplore.ieee.org/document/9209096)
- Classify and Localize Threat Items in X-Ray Imagery With Multiple Attention Mechanism and High-Resolution and High-Semantic Features [[Link]](https://ieeexplore.ieee.org/document/9513650)
- Raw Data Processing Using Modern Hardware for Inspection of Objects in X-Ray Baggage Inspection Systems [[Link]](https://ieeexplore.ieee.org/document/9417094)
- Temporal Fusion Based Mutli-scale Semantic Segmentation for Detecting Concealed Baggage Threats [[Link]](https://ieeexplore.ieee.org/document/9658932)
- Automatic Threat Detection Using Deep Neural Networks [[Link]](https://ieeexplore.ieee.org/document/9719550)
- Towards Real-world X-ray Security Inspection: A High-Quality Benchmark And Lateral Inhibition Module For Prohibited Items Detection [[Link]](https://ieeexplore.ieee.org/document/9710060)
- A Novel Incremental Learning Driven Instance Segmentation Framework to Recognize Highly Cluttered Instances of the Contraband Items [[Link]](https://arxiv.org/pdf/2201.02560.pdf)
- Deep Fusion Driven Semantic Segmentation for the Automatic Recognition of Concealed Contraband Items [[Link]](https://link.springer.com/chapter/10.1007%2F978-3-030-73689-7_53)
- Brittle Features May Help Anomaly Detection [[Link]](https://arxiv.org/abs/2104.10453)
- Deep Learning-Based X-Ray Baggage Hazardous Object Detection – An FPGA Implementation [[Link]](https://www.iieta.org/journals/ria/paper/10.18280/ria.350510)
- Operationalizing Convolutional Neural Network Architectures for Prohibited Object Detection in X-Ray Imagery [[Link]](https://arxiv.org/abs/2110.04906)
- Towards Automatic Threat Detection: A Survey of Advances of Deep Learning within X-ray Security Imaging  [[Link]](https://arxiv.org/abs/2001.01293)
- On the Impact of Using X-Ray Energy Response Imagery for Object Detection via Convolutional Neural Networks [[Link]](https://arxiv.org/abs/2108.12505)
- Towards Real-World Prohibited Item Detection: A Large-Scale X-ray Benchmark [[Link]](https://arxiv.org/pdf/2108.07020.pdf)
- PANDA: Perceptually Aware Neural Detection of Anomalies [[Link]](https://arxiv.org/abs/2104.13702)
- Tensor Pooling Driven Instance Segmentation Framework for Baggage Threat Recognition [[Link]](https://arxiv.org/abs/2108.09603)
- Unsupervised Anomaly Instance Segmentation for Baggage Threat Recognition [[Link]](https://arxiv.org/abs/2107.07333) 
- Symmetric Triangle Network for Object Detection Within X-ray Baggage Security Imagery [[Link]](https://ieeexplore.ieee.org/document/9533991)
- Anomaly Detection in X-ray Security Imaging: a Tensor-Based Learning Approach [[Link]](https://ieeexplore.ieee.org/document/9534034) 
- Automated Threat Objects Detection with Synthetic Data for Real-Time X-ray Baggage Inspection [[Link]](https://ieeexplore.ieee.org/document/9533928)
- Evaluating GAN-Based Image Augmentation for Threat Detection in Large-Scale Xray Security Images [[Link]](https://www.mdpi.com/2076-3417/11/1/36)
- An X-ray Image Enhancement Algorithm for Dangerous Goods in Airport Security Inspection [[Link]](https://ieeexplore.ieee.org/document/9407728/metrics#metrics)
- Baggage Threat Detection Under Extreme Class Imbalance [[Link]](https://ieeexplore.ieee.org/abstract/document/9787472)
- Detecting Overlapped Objects in X-Ray Security Imagery by a Label-Aware Mechanism [[Link]](https://ieeexplore.ieee.org/document/9722843)

#### 3D CT

- Contraband Materials Detection Within Volumetric 3D Computed Tomography Baggage Security Screening Imagery [[Link]](https://arxiv.org/abs/2012.11753)
- On the Evaluation of Semi-Supervised 2D Segmentation for Volumetric 3D Computed Tomography Baggage Security Screening [[Link]](https://breckon.org/toby/publications/papers/wang21segmentation.pdf)
- SliceNets — A Scalable Approach for Object Detection in 3D CT Scans [[Link]](https://ieeexplore.ieee.org/document/9423392)
- DEBISim: A simulation pipeline for dual energy CT-based baggage inspection systems [[Link]](https://content.iospress.com/articles/journal-of-x-ray-science-and-technology/xst200808)

### 2020
#### 2D

- Learning-based Material Classification in X-ray Security Images [[Link]](https://www.scitepress.org/Link.aspx?doi=10.5220/0008951702840291)
- Multi-label X-ray Imagery Classification via Bottom-up Attention and Meta Fusion [[Link]](https://openaccess.thecvf.com/content/ACCV2020/html/Hu_Multi-label_X-ray_Imagery_Classification_via_Bottom-up_Attention_and_Meta_Fusion_ACCV_2020_paper.html)
- Multi-view Object Detection Using Epipolar Constraints within Cluttered X-ray Security Imagery [[Link]](https://breckon.org/toby/publications/papers/isaac20multiview.pdf)
- Occluded Prohibited Items Detection: an X-ray Security Inspection Benchmark and De-occlusion Attention Module [[Link]](https://arxiv.org/abs/2004.08656)
- Trainable Structure Tensors for Autonomous Baggage Threat Detection Under Extreme Occlusion [[Link]](https://arxiv.org/abs/2009.13158)
- Cascaded Structure Tensor Framework for Robust Identification of Heavily Occluded Baggage Items from X-ray Scans [[Link]](https://arxiv.org/abs/2004.06780)
- Automatic Threat Detection in Baggage Security Imagery using Deep Learning Models [[Link]](https://ieeexplore.ieee.org/document/9342691)
- Automatic Threat Detection in Single, Stereo (Two) and Multi View X-Ray Images [[Link]](https://ieeexplore.ieee.org/document/9342253)
- Detecting Prohibited Items in X-Ray Images: a Contour Proposal Learning Approach [[Link]](https://ieeexplore.ieee.org/document/9190711)
- Background Adaptive Faster R-CNN for Semi-Supervised Convolutional Object Detection of Threats in X-Ray Images [[Link]](https://arxiv.org/abs/2010.01202)
- X-Ray Baggage Inspection With Computer Vision: A Survey [[Link]](https://ieeexplore.ieee.org/document/9162101)
- Data Augmentation of X-Ray Images in Baggage Inspection Based on Generative Adversarial Networks [[Link]](https://ieeexplore.ieee.org/document/9087880)

#### 3D CT
- Multi-Class 3D Object Detection Within Volumetric 3D Computed Tomography Baggage Security Screening Imagery [[Link]](https://arxiv.org/abs/2008.01218)
- On the Evaluation of Prohibited Item Classification and Detection in Volumetric 3D Computed Tomography Baggage Security Screening Imagery [[Link]](https://arxiv.org/abs/2003.12625)
- A Reference Architecture for Plausible Threat Image Projection (TIP) Within 3D X-ray Computed Tomography Volumes [[Link]](https://arxiv.org/abs/2001.05459)
- An Approach for Adaptive Automatic Threat Recognition Within 3D Computed Tomography Images for Baggage Security Screening [[Link]](https://arxiv.org/abs/1903.10604)

### 2019
#### 2D
- Evaluating the Transferability and Adversarial Discrimination of Convolutional Neural Networks for Threat Object Detection and Classification within X-Ray Security Imagery [[Link]](https://arxiv.org/abs/1911.08966)
- On the Impact of Object and Sub-Component Level Segmentation Strategies for Supervised Anomaly Detection within X-Ray Security Imagery [[Link]](https://arxiv.org/abs/1911.08216)
- Using Deep Neural Networks to Address the Evolving Challenges of Concealed Threat Detection within Complex Electronic Items [[Link]](https://breckon.org/toby/publications/papers/bhowmik19electronics.pdf)
- On the Use of Deep Learning for the Detection of Firearms in X-ray Baggage Security Imagery [[Link]](https://breckon.org/toby/publications/papers/gaus19firearms.pdf)
- The Good, the Bad and the Ugly: Evaluating Convolutional Neural Networks for Prohibited Item Detection Using Real and Synthetically Composite X-ray Imagery [[Link]](https://arxiv.org/abs/1909.11508)
- Evaluating a Dual Convolutional Neural Network Architecture for Object-wise Anomaly Detection in Cluttered X-ray Security Imagery [[Link]](https://breckon.org/toby/publications/papers/gaus19anomaly.pdf)
- Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection [[Link]](https://arxiv.org/abs/1901.08954)
- An evaluation of deep learning based object detection strategies for threat object detection in baggage security imagery [[Link]](https://www.sciencedirect.com/science/article/pii/S016786551930011X)
- Deep Convolutional Neural Network Based Object Detector for X-Ray Baggage Security Imagery [[Link]](https://ieeexplore.ieee.org/abstract/document/8995335)
- Automated firearms detection in cargo x-ray images using RetinaNet [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10999/109990P/Automated-firearms-detection-in-cargo-x-ray-images-using-RetinaNet/10.1117/12.2517817.full?SSO=1)
- Toward Automatic Threat Recognition for Airport X-ray Baggage Screening with Deep Convolutional Object Detection [[Link]](https://arxiv.org/abs/1912.06329)
- “Unexpected Item in the Bagging Area”: Anomaly Detection in X-Ray Security Images [[Link]](https://ieeexplore.ieee.org/document/8537982)
- Limits on transfer learning from photographic image data to X-ray threat detection [[Link]](https://www.semanticscholar.org/paper/Limits-on-transfer-learning-from-photographic-image-Caldwell-Griffin/7e7e445bbb757c4ec9165505925e48b0f94a92ad)
- Data Augmentation for X-Ray Prohibited Item Images Using Generative Adversarial Networks [[Link]](https://ieeexplore.ieee.org/document/8654640)
- Modified Adaptive Implicit Shape Model for Object Detection [[Link]](https://link.springer.com/chapter/10.1007/978-3-030-36802-9_17)
- Graph clustering and variational image segmentation for automated firearm detection in X-ray images [[Link]](https://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2018.5198)
- Semantic Segmentation for Prohibited Items in Baggage Inspection [[Link]](https://link.springer.com/chapter/10.1007/978-3-030-36189-1_41#:~:text=Semantic%20segmentation%20is%20a%20branch,open%20the%20baggage%20for%20inspection.)
- Application of Machine Learning Methods for Material Classification with Multi-energy X-Ray Transmission Images [[Link]](https://link.springer.com/chapter/10.1007/978-3-030-24274-9_17)
- Handgun Detection in Single-Spectrum Multiple X-ray Views Based on 3D Object Recognition [[Link]](https://link.springer.com/article/10.1007/s10921-019-0602-9)


#### 3D CT
- On the Relevance of Denoising and Artefact Reduction in 3D Segmentation and Classification within Complex Computed Tomography Imagery [[Link]](https://breckon.org/toby/publications/papers/mouton19relevance.pdf)

### 2018

#### 2D
- On Using Deep Convolutional Neural Network Architectures for Automated Object Detection and Classification within X-ray Baggage Security Imagery [[Link]](https://breckon.org/toby/publications/papers/akcay18architectures.pdf)
- GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training [[Link]](https://arxiv.org/abs/1805.06725)
- Multi-view X-ray R-CNN [[Link]](https://arxiv.org/abs/1810.02344)
- A GAN-Based Image Generation Method for X-Ray Security Prohibited Items [[Link]](https://link.springer.com/chapter/10.1007/978-3-030-03398-9_36)
- Prohibited Item Detection in Airport X-Ray Security Images via Attention Mechanism Based CNN [[Link]](https://link.springer.com/chapter/10.1007/978-3-030-03335-4_37)
- Convolutional Neural Networks for Automatic Threat Detection in Security X-Ray Images [[Link]](https://ieeexplore.ieee.org/document/8614074)
- Automatic threat recognition of prohibited items at aviation checkpoint with x-ray imaging: a deep learning approach [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10632/1063203/Automatic-threat-recognition-of-prohibited-items-at-aviation-checkpoint-with/10.1117/12.2309484.full?webSyncID=8531ab0d-3a6b-03c9-7c00-9b6bcd746b80&sessionGUID=d8e8abee-aed6-ae05-c117-aac5a9199362)


#### 3D CT
- Consensus relaxation on materials of interest for adaptive ATR in CT images of baggage [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10632/106320E/Consensus-relaxation-on-materials-of-interest-for-adaptive-ATR-in/10.1117/12.2309839.full)
- Adaptive Target Recognition: A Case Study Involving Airport Baggage Screening [[Link]](https://arxiv.org/abs/1811.04772)


### Earlier

#### 2D
- An Evaluation Of Region Based Object Detection Strategies Within X-Ray Baggage Security Imagery [[Link]](https://breckon.org/toby/publications/papers/akcay17region.pdf)
- On using Feature Descriptors as Visual Words for Object Detection within X-ray Baggage Security Screening [[Link]](https://breckon.org/toby/publications/papers/kundegorski16xray.pdf)
- Transfer Learning Using Convolutional Neural Networks For Object Classification Within X-Ray Baggage Security Imagery [[Link]](https://breckon.org/toby/publications/papers/akcay16transfer.pdf)
- Improving Feature-based Object Recognition for X-ray Baggage Security Screening using Primed Visual Words [[Link]](https://breckon.org/toby/publications/papers/turcsany13xray.pdf)
- A Combinational Approach to the Fusion, De-noising and Enhancement of Dual-Energy X-Ray Luggage Images [[Link]](https://ieeexplore.ieee.org/document/1565297)
- Improving Weapon Detection In Single Energy X-ray Images Through Pseudocoloring [[Link]](https://ieeexplore.ieee.org/abstract/document/1715507)
- A review of X-ray explosives detection techniques for checked baggage [[Link]](https://www.sciencedirect.com/science/article/pii/S0969804312000127)
- A Logarithmic X-Ray Imaging Model for Baggage Inspection: Simulation and Object Detection [[Link]](https://ieeexplore.ieee.org/document/8014771)
- Automatic Defect Recognition in X-Ray Testing Using Computer Vision [[Link]](https://ieeexplore.ieee.org/document/7926702)
- Modern Computer Vision Techniques for X-Ray Testing in Baggage Inspection[[Link]](https://ieeexplore.ieee.org/document/7775025)
- Inspection of Complex Objects Using Multiple-X-Ray Views [[Link]](https://ieeexplore.ieee.org/document/6782468)
- Automated X-Ray Object Recognition Using an Efficient Search Algorithm in Multiple Views [[Link]](https://ieeexplore.ieee.org/document/6595901)
- X-Ray Testing by Computer Vision [[Link]](https://ieeexplore.ieee.org/document/6595900)
- Automated detection in complex objects using a tracking algorithm in multiple X-ray views [[Link]](https://ieeexplore.ieee.org/document/5981715)
- Threat Objects Detection in X-ray Images Using an Active Vision Approach [[Link]](https://link.springer.com/article/10.1007/s10921-017-0419-3)
- Object recognition in X-ray testing using an efficient search algorithm in multiple views [[Link]](https://www.ingentaconnect.com/content/bindt/insight/2017/00000059/00000002/art00008;jsessionid=1n6a28jhds21c.x-ic-live-02)
- Modern Computer Vision Techniques for X-Ray Testing in Baggage Inspection [[Link]](https://ieeexplore.ieee.org/document/7775025)
- Automated Detection of Threat Objects Using Adapted Implicit Shape Model [[Link]](https://ieeexplore.ieee.org/document/7123190)
- A review of X-ray explosives detection techniques for checked baggage [[Link]](https://www.sciencedirect.com/science/article/pii/S0969804312000127)
- Explosives detection systems (EDS) for aviation security [[Link]](https://www.sciencedirect.com/science/article/pii/S0165168402003912)

#### 3D CT
- Geometrical Approach for the Automatic Detection of Liquid Surfaces in 3D Computed Tomography Baggage Imagery [[Link]](https://breckon.org/toby/publications/papers/chermak15liquids.pdf)
- Materials-Based 3D Segmentation of Unknown Objects from Dual-Energy Computed Tomography Imagery in Baggage Security Screening [[Link]](https://breckon.org/toby/publications/papers/mouton15segmentation.pdf)
- Object Classification in 3D Baggage Security Computed Tomography Imagery using Visual Codebooks [[Link]](https://breckon.org/toby/publications/papers/flitton15codebooks.pdf)
- 3D Object Classification in Baggage Computed Tomography Imagery using Randomised Clustering Forests [[Link]](https://breckon.org/toby/publications/papers/mouton14randomised.pdf)
- Investigating Existing Medical CT Segmentation Techniques within Automated Baggage and Package Inspection [[Link]](https://breckon.org/toby/publications/papers/megherbi13segmentation.pdf)
- Radon Transform based Metal Artefacts Generation in 3D Threat Image Projection [[Link]](https://breckon.org/toby/publications/papers/megherbi13radon.pdf)
- A Comparison of 3D Interest Point Descriptors with Application to Airport Baggage Object Detection in Complex CT Imagery [[Link]](https://breckon.org/toby/publications/papers/flitton13interestpoint.pdf)
- A Distance Weighted Method for Metal Artefact Reduction in CT [[Link]](https://breckon.org/toby/publications/papers/mouton13mar.pdf)
- An Experimental Survey of Metal Artefact Reduction in Computed Tomography [[Link]](https://breckon.org/toby/publications/papers/mouton13survey.pdf)
- An Evaluation of CT Image Denoising Techniques Applied to Baggage Imagery Screening [[Link]](https://breckon.org/toby/publications/papers/mouton13denoising.pdf)
- Fully Automatic 3D Threat Image Projection: Application to Densely Cluttered 3D Computed Tomography Baggage Images [[Link]](https://breckon.org/toby/publications/papers/megherbi12tip.pdf)
- A Comparison of Classification Approaches for Threat Detection in CT based Baggage Screening [[Link]](https://breckon.org/toby/publications/papers/megherbi12baggage.pdf)
- A Novel Intensity Limiting Approach to Metal Artefact Reduction in 3D CT Baggage Imagery [[Link]](https://breckon.org/toby/publications/papers/mouton12mar.pdf)
- A 3D Extension to Cortex Like Mechanisms for 3D Object Class Recognition [[Link]](https://breckon.org/toby/publications/papers/flitton12cortex.pdf)
- Object Recognition using 3D SIFT in Complex CT Volumes [[Link]](https://breckon.org/toby/publications/papers/flitton10baggage.pdf)
- A Classifier based Approach for the Detection of Potential Threats in CT based Baggage Screening [[Link]](https://breckon.org/toby/publications/papers/megherbi10baggage.pdf)
- A review of automated image understanding within 3D baggage computed tomography security screening [[Link]](https://breckon.org/toby/publications/papers/mouton15review.pdf)
- A volumetric object detection framework with dual-energy CT [[Link]](https://ieeexplore.ieee.org/document/4774641)
- Exact Reconstruction for Dual Energy Computed Tomography Using an H-L Curve Method [[Link]](https://ieeexplore.ieee.org/document/4179793)
- Automatic segmentation of CT scans of checked baggage [[Link]](https://www.stratovan.com/sites/default/files/AutomaticSegmentationOfCtScansOfCheckedBaggage.pdf)
- Automatic Segmentation of Unknown Objects, with Application to Baggage Security [[Link]](https://link.springer.com/chapter/10.1007/978-3-642-33709-3_31)
- ALERT Strategic Studies [[Link]]()
- Joint metal artifact reduction and segmentation of CT images using dictionary-based image prior and continuous-relaxed potts model [[Link]](https://ieeexplore.ieee.org/document/7350909)
- Using Threat Image Projection Data Forassessing Individual Screener Performance [[Link]](https://www.witpress.com/elibrary/wit-transactions-on-the-built-environment/82/15153)
- 3D threat image projection [[Link]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6805/680508/3D-threat-image-projection/10.1117/12.766432.full?SSO=1)
- Learning-Based Object Identification and Segmentation Using Dual-Energy CT Images for Security [[Link]](https://ieeexplore.ieee.org/document/7159062)