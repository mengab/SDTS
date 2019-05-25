# Enhancing Quality for VVC Compressed Videos by Jointly Exploiting Spatial Details and Temporal Structure

TensorFlow implementation of "Enhancing Quality for VVC Compressed Videos by Jointly Exploiting Spatial Details and Temporal Structure" 

[[arXiv]](https://arxiv.org/pdf/1901.09575.pdf)      [[ICIP2019]](https://cmsworkshops.com/ICIP2019/Papers/AcceptedPapers.asp)

## Framework

<p align="center">
    <img src="files/overview.png" width="750"> <br />
    <em> The proposed quality enhancement network</em>
</p>


##
## Motion Compensation (MC)
<p align="center">
    <img src="files/MC.png" width="600"> <br />
    <em> Top: flow map estimated relating the original frame. Bottom: the consecutive frames without and with motion
compensation (No MC and MC). </em>
</p>

## Installation
The code was developed using Python 3.6 & TensorFlow 1.3 & CUDA 8.0. 

## Code v1.0
Currently, we release our research code for testing. It should produce the same results as in the paper under LD configuration.
## Testing
* It's easy to understand testing functions and to test your own data.
* An example of test usage is shown as follows:
```bash 
python CUDA_VISIBLE_DEVICES=0 SDTS_test.py
```


## Citation

If you use any part of our code, or our method is useful for your research, please consider citing:

```
@inproceedings{SDTS2019,
author = {Xiandong, Meng and Xuan, Deng and Shuyuan, Zhu and Bing, Zeng},
title = {Enhancing Quality for VVC Compressed Videos by Jointly Exploiting Spatial Details and Temporal Structure},
booktitle = {ICIP},
year = 2019
}
```
## Contact
We are glad to hear if you have any suggestions and questions. 
Please send email to xmengab@connect.ust.hk
