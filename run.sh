
CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2 --seed 3552 --AWM BWA_fusion_dropout_feat_v2

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model WIS  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name WIS --seed 3552

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model WIS  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name WIS --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model WIS_2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name WIS --seed 3552 --branch_num 4


#-------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model1  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model1 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model2 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2_WIS  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_WIS --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2_WIS2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_WIS2 --seed 3552 --branch_num 4 --AWM AWM1

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model2 --seed 3552 --branch_num 4 --AWM AWM1

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model3  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model3_1 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model3_contras  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model3_contras --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=3 python teacher_main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model3  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model3_teacher --seed 3552 --branch_num 4 --AWM AWM1


CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model5  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model5 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model6  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model6_2 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model7  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model7 --seed 3552 --branch_num 4
CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model8  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model8 --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model9  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model9 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model10  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model10-2 --seed 3552 --branch_num 4
CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model11  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model11 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model12  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model12 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model13  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model13 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model14  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model14 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model15  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model15 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model19  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model19 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model20  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model20 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model21  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model21 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model22  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model22_1 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model23  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model23 --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model24  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model24_10 --seed 3552 --branch_num 4 --lambda4 0.001

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model25  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model25_1 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model23_v2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model23_v2 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model26  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model26_1 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model27  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model27 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model28  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model28 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model29  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model29_10 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model30  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model30 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model30  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model30_KL --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model31  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model31_feat_norm_0.001 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model31_relu  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model31_relu.001 --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model32  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model32 --seed 3552 --branch_num 4
CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model33  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model33 --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model34  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model34 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model35  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model35 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model36  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model36 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model37  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model37 --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model38  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model38 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model39  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model39 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model40  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model40 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model41  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model41_1 --seed 3552 --branch_num 4

CUDA_VISIBLE_DEVICES=1 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model42  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model42 --seed 3552 --branch_num 4


CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model43  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model43 --seed 3552 --AWM BWA_fusion_dropout_feat_v2

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model44  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model44 --seed 3552 --AWM BWA_fusion_dropout_feat_v2

CUDA_VISIBLE_DEVICES=2 python main.py --max-seqlen 560 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model model45  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name model45_mil_guided_neg_singleFeatHead_attn02 --seed 3552 --branch_num 4



CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2_1  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_1_noshare --seed 3552 --AWM BWA_fusion_dropout_feat_v2

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2_iterate  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_iterate --seed 3552 --AWM BWA_fusion_dropout_feat_v2

CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2_iterate2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_iterate22 --seed 3552 --AWM BWA_fusion_dropout_feat_v3

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2 --seed 3552 --AWM BWA_fusion_dropout_feat_v4

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_5 --seed 3552 --AWM BWA_fusion_dropout_feat_v5

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_6 --seed 3552 --AWM BWA_fusion_dropout_feat_v6

CUDA_VISIBLE_DEVICES=3 python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --num-class 20 --use-model CO2_1  --max-iter 20000  --dataset SampleDataset --weight_decay 0.001 --model-name CO2_1 --seed 3552 --AWM BWA_fusion_dropout_feat_v2


CUDA_VISIBLE_DEVICES='0' python main.py --use-model model9 --lambda_mutual 0 --lambda_coteach 1 --lambda_att_coteach 1 --model-name model9 --seed 3552

#---------------------------------------------------------------------------------------------------
#ActivityNet
CUDA_VISIBLE_DEVICES=0 python main.py --k 5  --dataset-name ActivityNet1.2 --path-dataset path/to/ActivityNet --num-class 100 --use-model ANT_CO2  --dataset AntSampleDataset --lr 3e-5 --max-seqlen 60 --model-name ANTModel45 --seed 3552



#------------------------------------Test-----------------------------------


CUDA_VISIBLE_DEVICES=0 python test.py --use-model Model45 --model-name Model45
CUDA_VISIBLE_DEVICES=0 python test.py --use-model model5 --model-name model5 --resultsName model5



