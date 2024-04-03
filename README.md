# Foul Prediction in Soccer Broadcast Videos
This the repo for the paper "Foul prediction with estimated poses from soccer broadcast video" (Fang et el., 2023)
This project aims to predict fouls in soccer matches using estimated poses from broadcast video footage. It utilizes advanced computer vision techniques to analyze video data and identify potential foul scenarios.

## Introduction
Recent advances in computer vision have made significant progress in tracking
and pose estimation of sports players. However, there have been fewer studies on
behavior prediction with pose estimation in sports, in particular, the prediction
of soccer fouls is challenging because of the smaller image size of each player and
of difficulty in the usage of e.g., the ball and pose information. In our research,
we introduce an innovative deep learning approach for anticipating soccer fouls.
This method integrates video data, bounding box positions, image details, and
pose information by curating a novel soccer foul dataset. Our model utilizes a
combination of convolutional and recurrent neural networks (CNNs and RNNs)
to effectively merge information from these four modalities. The experimental
results show that our full model outperformed the ablated models, and all of the
RNN modules, bounding box position and image, and estimated pose were useful
for the foul prediction. Our findings have important implications for a deeper
understanding of foul play in soccer and provide a valuable reference for future
research and practice in this area.

## Dataset Description
dataset: 
## Inference
```
python futurefoul.py
```
## Reference
```
[1] Giancola, S. et al. IEEE Conference, 1711–1721, 2018 
[2] Zhang, et al. IEEE/CVF Conference , 889--898, 2019
[3] Giancola, S. et al. IEEE Conference, 4490–4499, 2021 
[4] Scott, A. et al. IEEE/CVF Conference , 3569–3579, 2022
[5] Cioppa, A. et al. IEEE/CVF Conference , 3491–3502, 2022
...
```
