# Deep Aggregation Vision Transformer for Infant Pose Estimation    

[中文版](https://github.com/SZAR-LAB/AggPose/blob/main/CHINESE_README.md)  

## Introduction     

Movement and pose assessment of newborns lets trained pediatricians predict neurodevelopmental disorders, allowing early intervention for related diseases. However, most of newest approaches for human pose estimation method focus on adults, lacking publicly large-scale dataset and powerful deep learning framework for infant pose estimation. In this paper, we fill this gap by proposing Deep Aggregation Vision Transformer for human (infant) posture estimation (AggPose), which introduces a high-resolution transformer framework without using convolution operations to extract features in the early stages. It generalizes Transformer + MLP to multi-scale deep layer aggregation within feature maps, thus enabling information fusion between different levels of vision tokens. We pre-train AggPose on COCO pose estimation and apply it on our newly released large-scale infant pose estimation dataset. The results show that AggPose could effectively learn the multi-scale features among different resolutions and significantly improve the performance.  

This work was accepted by IJCAI-ECAI 2022 AI for Good Track

## Requirements


## COCO Dataset

Download the dataset: [COCO 2017](https://cocodataset.org/#download)

The dataset folder should like this:    

```
── coco
  │-- annotations
  │   │-- person_keypoints_train2017.json
  │   |-- person_keypoints_val2017.json
  │   |-- person_keypoints_test-dev-2017.json
  |-- person_detection_results
  |   |-- COCO_val2017_detections_AP_H_56_person.json
  |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
  │-- train2017
  │   │-- 000000000009.jpg
  │   │-- 000000000025.jpg
  │   │-- 000000000030.jpg
  │   │-- ...
  `-- val2017
      │-- 000000000139.jpg
      │-- 000000000285.jpg
      │-- 000000000632.jpg
      │-- ...
```


AggPose Model Parameter:    
[COCO 2017 AggPose-L 256x192](https://drive.google.com/file/d/1h9mu7EDwwWLmYcWh0z1ToknCbR08pFZU/view?usp=sharing)    
[infantpose 2022 AggPose-L 256x192](https://drive.google.com/file/d/1wvQo5tqVr39fopm3gf9likUU1xk_sUPf/view?usp=sharing)
      
More modified code will release soon. I will update this repo continually. I am trying to re-train the model with 384x288 input size. For 256x192 input size, please use the parm in above link. Thanks! 




