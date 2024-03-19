# ViComp
## Introduction
This is the PyTorch implementation of the paper “ViComp: Video Compensation for Projector-Camera Systems”.

    @ARTICLE{Wang2024TVCG,
        author={Wang, Yuxi and Ling, Haibin and Huang, Bingyao},
        journal={IEEE Transactions on Visualization and Computer Graphics}, 
        title={ViComp: Video Compensation for Projector-Camera Systems}, 
        year={2024},
        pages={1-10},}


    
## Datasets
The datasets used in the paper are collected from two blender open source films --"Big Buck Bunny" and "Spring".
The shots are detected by [TransNet V2: Shot Boundary Detection Neural Network](https://github.com/soCzech/TransNetV2).
The image named "textImg.png" is downloaded from [this website](http://www.lybczcw.com/News_read_id_397.shtml) for displacement estimation, you can replace it with any texture-rich image.  

## Usage
   1. Clone this repo:
  
     git clone https://github.com/cyxwang/ViComp

   2. Download the videos and extract the frames to "DATANAME", generate the shot index file named "DATANAME_shot_index.txt". Put these files into "data".
     
   3. Download the [pre-trained model of CompenNeSt](https://github.com/BingyaoHuang/CompenNeSt-plusplus) to "pretrain".  
   
   4. Download the code of [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) to "src/python/compensation".
   
   5. Download the [pre-trained model of FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) to "pretrain".

   6. cd to "src/python", edit data path and hyper-parameters in "online.yaml", then run testing.sh to start the system.
      
     cd src/python
     
     sh testing.sh
     


