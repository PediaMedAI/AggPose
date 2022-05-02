# AggPose: Deep Aggregation Vision Transformer for Infant Pose Estimation (Pytorch官方实现版本)
## 介绍

[English Version](https://github.com/SZAR-LAB/AggPose/blob/main/README.md)     

新生儿的姿势评估可以用于辅助儿科医生预测神经发育障碍，进而早期干预相关疾病。然而，绝大多数的人体姿态估计方法都以成年人为基准，缺乏大规模的婴儿姿态估计数据集和深度学习框架。在本文中，我们通过提出用于人类（重点是婴幼儿）姿势估计的AggPose来填补这一空白，效仿HRNet，它引入了一个多分辨率的Transformer框架，但是在模型早期阶段用Transformer取代卷积运算来提取特征。该算法将 Transformer + MLP 推广到特征图中的多尺度深层聚合，从而实现不同分辨率下视觉标记之间的信息融合。本文中的Transformer结构使用了和SegFormer一致的Mix-Transformer / Mix-FFN结构，代码部分容易理解。我们在 COCO 姿势估计上对 AggPose 进行了预训练，并预训练模型在我们新发布的大规模婴儿姿势估计数据集上finetune。结果表明，AggPose 可以有效地学习不同分辨率之间的多尺度特征，并显着提高性能。虽然模型相较HRFormer, TokenPose等2021年的baseline提升不明显，但是这一结构证实了全Transformer结构也可以用于人体姿态评估。在应用侧，作者希望能在未来将这一算法和自监督学习相结合，扩展人体姿态评估在临床研究和产品开发中的可能性。    
 
本文的数据来源于深圳宝安妇幼保健院、深圳儿童医院。在数据集整理完备后将通过infantpose@gmail.com这一邮箱来接受临床数据访问申请。    

