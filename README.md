# LSTC_VAD
This repo is the official Pytorch implementation of paper:
>"[**Long-Short Temporal Co-Teaching for Weakly Supervised Video Anomaly Detection**](https://arxiv.org/pdf/2303.18044.pdf)"
>

## Requirements
Please follow the `requirements.txt`

## Training
### Setup
- You can use the pre-trained I3D such [pytorch-resnet3d](https://github.com/Tushar-N/pytorch-resnet3d) or [C3D model](https://github.com/jfzhang95/pytorch-video-recognition).

- You can also download the extracted I3D features from links below:
>[**ShanghaiTech I3D features (code:8XJB)**](https://web.ugreen.cloud/web/#/share/EC-6Tks554JJ122RscV-309B96)
>
>[**UCF-Crime I3D features (code:6EB8)**](https://web.ugreen.cloud/web/#/share/EC-2aki554JJ122lTql-309B96)
>
>[**UBnormal I3D features (code:PYL5)**](https://web.ugreen.cloud/web/#/share/EC-uQCN554JJ122RV6e-309B96)

Take the example of ShanghaiTech, run the following commands:
### Train the Spatial-Transformer (the number of input clip is one), STN:
```shell
python spatio_transformer_shanghaitech.py --encoder_weight_init --regressor_weight_init --FFN_layerNorm --MHA_dropout 0.3 --FFN_dropout 0.3 --dataset_path SHT_I3D_16PATCH.h5 --gpu 0
```
Generating the pseudo labels of spatio-transformer:
```shell
python pseudo_labels_generator_spatio.py --dataset SHT --n_patch 16 --FFN_layerNorm --threshold 0.9 --pseudo_labels_path STN_pseudo_labels.npy --training_txt SH_Train_new.txt --dataset_path SHT_I3D_16PATCH.h5 --gpu 0
```
### Train the Temporal-Transformer, LTN:
```shell
python temporal_transformer_shanghaitech.py --part_len 3 --MHA_layerNorm --FFN_layerNorm --relative_position_encoding --pseudo_labels_path STN_pseudo_labels.npy --dataset_path SHT_I3D_16PATCH.h5 --gpu 0
```
Generating the pseudo labels of temporal-transformer:
```shell
python pseudo_labels_generator_temporal.py --dataset SHT --relative_position_encoding --n_hidden 4096 --n_patch 16 --n_head 8 --d_k 256 --d_v 256 --part_len 3 --MHA_layerNorm --FFN_layerNorm --dataset_path SHT_I3D_16PATCH.h5 --temporal_model_path temporal_model --classifier_model_path classifier_model --pseudo_labels_path LTN_pseudo_labels.npy --training_txt SH_Train_new.txt --threshold 0.65 --gpu 0
```
>For multi-gpu training, you can use the command: `--data_parallel --gpu id0,id1`

## Inference

- You can download the checkpoint models from links below:
>[**ShanghaiTech (code:L958)**](https://web.ugreen.cloud/web/#/share/EC-fbmW554JJ122VSms-309B96)
>
>[**UCF-Crime (code:8383)**](https://web.ugreen.cloud/web/#/share/EC-FZBy554JJ1226WoA-309B96)
>
>[**UBnormal (code:3X3N)**](https://web.ugreen.cloud/web/#/share/EC-Ifji554JJ122kFxa-309B96)
>

for ShanghaiTech:
```shell
python evaluation_shanghaitech_ubnormal.py --dataset SHT --temporal_MHA_layerNorm --temporal_FFN_layerNorm --temporal_relative_position_encoding --dataset_path SHT_I3D_16PATCH.h5 --temporal_model_path shanghaitech_temporal_model_oneCrop_I3D_RGB_0.9779.ckpt --classifier_model_path shanghaitech_classifier_model_oneCrop_I3D_RGB_0.9779.ckpt --gpu 0
```
for UBnormal:
```shell
python evaluation_shanghaitech_ubnormal.py  --dataset UBnormal  --d_model 1024 --part_len 5 --temporal_MHA_layerNorm --temporal_FFN_layerNorm --temporal_relative_position_encoding --dataset_path UBnormal_I3D_16PATCH.h5 --temporal_model_path UBnormal_temporal_model_oneCrop_I3D_RGB_0.7551.ckpt --classifier_model_path UBnormal_classifier_model_oneCrop_I3D_RGB_0.7551.ckpt --test_mask_dir data/UBnormal/test_frame_mask --training_txt data/UBnormal/train_video_names_frames.txt --testing_txt data/UBnormal/test_video_names_frames.txt --gpu 0
```
for UCF-Crime:
```shell
python evaluation_UCF.py --n_patch 9 --part_num 32 --part_len 2 --dataset_path UCF_I3D_9PATCH.h5 --temporal_MHA_layerNorm --temporal_FFN_layerNorm --temporal_model_path UCF_temporal_model_oneCrop_I3D_RGB_0.8570.ckpt --classifier_model_path UCF_classifier_model_oneCrop_I3D_RGB_0.8570.ckpt --relative_position_encoding --gpu 0
```
>Tips: If the model is trained by multi-gpu mode, you must add the command `--data_parallel` in the inference stage.

## License
This repo is released under the MIT License.

## Citation

If this repo is useful for your research, please consider citing our paper:
```bibtex
@article{sun2023longshort,
      title={Long-Short Temporal Co-Teaching for Weakly Supervised Video Anomaly Detection}, 
      author={Shengyang Sun and Xiaojin Gong},
      year={2023},
      journal={arXiv preprint arXiv:2303.18044},
}
```

## Acknowledgements  

Partial codes are based on [MIST](https://github.com/fjchange/MIST_VAD), we sincerely thank them for their contributions.

