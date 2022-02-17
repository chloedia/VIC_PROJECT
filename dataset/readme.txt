This dataset is collected from two Internet videos.
Game 1: https://www.youtube.com/watch?v=XRfXrj9Bkiw
Game 2: https://www.youtube.com/watch?v=R_IlsfVMQBk
This dataset is used for research only.

Files:
1. annotation_1.mat and annotation_2.mat. Each file has “BBox” for bounding box of players and “imgName” for image names.
2. DataSet_001 and DataSet_002 folders. Each folder contains images (.jpg) from the original video. The file name is the frame number. For example, DataSet_001/0186.jpg is the 186th frame of the Game 1 video. The frame number is 0 based and the fps is 30.
3 ReadDemo.m. A Matlab demo file that shows the ground truth bounding boxes of players in each frame.

If you use this dataset, please cite following paper:
@inproceedings{lu2017light,
  title={Light Cascaded Convolutional Neural Networks for Accurate Player Detection.},
  author={Lu, keyu and Chen, Jianhui and Little, James J},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2017}
}
