echo 'just test'
cd /home/airstudio/code/2.12/
python3 tools/test.py a_configs/voc/base_regnet/paa_3.2G_f0.5_voc.py /home/airstudio/code/2.12/work_dirs_voc/paa_3.2G_f0.5_voc/epoch_4.pth --eval mAP
python3 tools/test.py b_configs/dior/baseline_regnet_compose/paab_regnet_3.2GF_f0.5_dior.py /home/airstudio/code/2.12/work_dirs/paab_regnet_3.2GF_f0.5_dior/epoch_23.pth --eval mAP
#cd /home/airstudio/code/2.12/
#python3 tools/test.py b_configs/dior/baseline_regnet_compose/paab_regnet_3.2GF_f0.5_dior.py /home/airstudio/code/2.12/work_dirs/paab_regnet_3.2GF_f0.5_dior/epoch_23.pth --eval mAP

#CUDA_VISIBLE_DEVICES=3