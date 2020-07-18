# EfficientNet_based_on_Transfer_Learning

基于猫猫和狗狗的图像二分类。

Image classification based on two categories of cats and dogs.


第一部分
EfficientNet网络模型的训练代码，基于迁移学习，分为“transfer learning”和“fine-tuning”两种模式，
分别对应training_efficientnet_with_transfer_learning.py和training_efficientnet_with_fine_tuning.py两个代码文件。
典型的迁移学习过程，首先通过“transfer learning”对新的数据集进行训练，训练过一定epoch之后，改用“fine-tune”方法继续训练，同时降低学习率。

PART 1
The training code of the EfficientNet network model is based on transfer learning and is divided into two modes: "transfer learning" and "fine-tuning".
Corresponding to the two code files training_efficientnet_with_transfer_learning.py and training_efficientnet_with_fine_tuning.py respectively.
In a typical transfer learning process, a new data set is trained through "transfer learning" first, and after a certain epoch, the "fine-tune" method is used to continue training while reducing the learning rate.


第二部分
模型训练完成后，使用真实图像进行测试，可以分为基于“transfer learning”模式和基于“fine-tuning”模式，
分别对应validation_efficientnet_with_tl.py和validation_efficientnet_with_ft.py两个代码文件。

PART 2
After the model training is completed, use real images for testing, which can be divided into "transfer learning" mode and "fine-tuning" mode. 
Corresponding to the two code files validation_efficientnet_with_tl.py and validation_efficientnet_with_ft.py respectively.
