# TestFace

## Model Structure

![more_detail.png](https://i.loli.net/2019/12/16/szEXB7QebOyScP3.png)

## Introduction 

This code was my initial trying for a end2end-trained parts-selective faceparsing, actually it could be seem as a Three Stage model:

- Stage1: Detection 
- Stage2:Trainable Cropper: 
Using the split feturemap from Stage1 to learn how to crop it properly.
- Stage3: Segmentation
Do segmentation for all the single cropped parts.

## Latest Result
Because I only supervised the segmentation at that time, the learning result was strange. It was not doing segmentation but just generating parts. Later, I did not continue to make this idea, but fixed the theta matrix, and I plan to consider it after all parts have been verfied.

Improved version please check ["new_end2end"](https://github.com/aod321/new_end2end)
![result.png](https://i.loli.net/2019/12/16/5xqsoRhbZa2r3dF.png)


