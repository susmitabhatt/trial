#*******************************************************************************#

# Author  : Ganji Sreeram
# Roll.No : 156102028
# Address : EMST Lab, IIT Guwahati
# Date    : 27-January-2018

#*******************************************************************************#

clear
#set-up for single machine or cluster based execution
. ./cmd.sh
#set the paths to binaries and other executables
[ -f path.sh ] && . ./path.sh
train_cmd=run.pl
decode_cmd=run.pl

numLeavesMLLT=2000
numGaussMLLT=16000
numLeavesSAT=2000
numGaussSAT=16000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

feat_nj=20
train_nj=10
decode_nj=10

#================================================
#	SET SWITCHES
#================================================
mfcc_extract_sw=0
mfcc_pitch_extract_sw=0
mfcc_vtln_extract_sw=0
fbank_extract_sw=0
fbank_pitch_extract_sw=0
bnf_extract_sw=0
#================================================

mono_train_sw=0
mono_test_sw=1

tri1_train_sw=0
tri1_test_sw=1

tri2_train_sw=0
tri2_test_sw=1

tri3_train_sw=0
tri3_test_sw=1

sgmm_train_test_sw=0

dnn_train_sw=0
dnn_test_sw=1

lstm_train_sw=0
lstm_test_sw=0

tdnn_train_sw=0
tdnn_test_sw=0

bnf_train_sw=0
#================================================
#      Set Directories
#================================================

train_dir=data/train
lang_dir=data/lang_trigram

test_dir=data/eval

graph_dir=graph_eval
decode_dir=decode_eval

exp_dir=exp_mfcc_pitch
#====================================================

if [ $mfcc_extract_sw == 1 ]; then

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training               "
echo ============================================================================
#extract MFCC features and perfrom CMVN
mfccdir=mfcc

for x in eval; do 
	utils/fix_data_dir.sh data/$x;
	steps/make_mfcc.sh --cmd "$train_cmd" --nj "$feat_nj" data/$x $exp_dir/make_mfcc/$x $mfccdir || exit 1;
 	steps/compute_cmvn_stats.sh data/$x $exp_dir/make_mfcc/$x $mfccdir || exit 1;
        utils/validate_data_dir.sh data/$x;
done

fi
#====================================================
if [ $mfcc_pitch_extract_sw == 1 ]; then

echo ============================================================================
echo "         MFCC+PITCH Feature Extration & CMVN for Training               "
echo ============================================================================

#extract MFCC features and perfrom CMVN
mfccdir=mfcc_pitch

for x in eval; do 
	utils/fix_data_dir.sh data/$x;
	steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj "$feat_nj" data/$x $exp_dir/make_mfcc_pitch/$x $mfccdir || exit 1;
 	steps/compute_cmvn_stats.sh data/$x $exp_dir/make_mfcc_pitch/$x $mfccdir || exit 1;
	utils/validate_data_dir.sh data/$x;
done

fi
#====================================================
if [ $mfcc_vtln_extract_sw == 1 ]; then

echo ============================================================================
echo "         MFCC+VTLN Feature Extration & CMVN for Training               "
echo ============================================================================

#extract MFCC features and perfrom CMVN
mfccdir=mfcc_vtln

for x in train dev; do 
	utils/fix_data_dir.sh data/$x;
	steps/make_mfcc_vtln.sh --cmd "$train_cmd" --nj "$feat_nj" data/$x $exp_dir/make_mfcc_vtln/$x $mfccdir || exit 1;
 	steps/compute_cmvn_stats.sh data/$x $exp_dir/make_mfcc_vtln/$x $mfccdir || exit 1;
	utils/validate_data_dir.sh data/$x;
done

fi

#====================================================

if [ $fbank_extract_sw == 1 ]; then

echo ============================================================================
echo "         FBANK Feature Extration & CMVN for Training               "
echo ============================================================================
#extract MFCC features and perfrom CMVN
mfccdir=fbank

for x in train dev; do 
	utils/fix_data_dir.sh data/$x;
	steps/make_fbank.sh --cmd "$train_cmd" --nj "$feat_nj" data/$x $exp_dir/make_fbank/$x $mfccdir || exit 1;
 	steps/compute_cmvn_stats.sh data/$x $exp_dir/make_fbank/$x $mfccdir || exit 1;
        utils/validate_data_dir.sh data/$x;
done

fi
#====================================================

if [ $fbank_pitch_extract_sw == 1 ]; then

echo ============================================================================
echo "         FBANK+PITCH Feature Extration & CMVN for Training               "
echo ============================================================================
#extract MFCC features and perfrom CMVN
mfccdir=fbank_pitch

for x in train dev; do 
	utils/fix_data_dir.sh data/$x;
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj "$feat_nj" data/$x $exp_dir/make_fbank_pitch/$x $mfccdir || exit 1;
 	steps/compute_cmvn_stats.sh data/$x $exp_dir/make_fbank_pitch/$x $mfccdir || exit 1;
        utils/validate_data_dir.sh data/$x;
done

fi
#====================================================

if [ $mono_train_sw == 1 ]; then

echo ============================================================================
echo "                   MonoPhone Training                	        "
echo ============================================================================

steps/train_mono.sh --nj "$train_nj" --cmd "$train_cmd" $train_dir $lang_dir $exp_dir/mono || exit 1; 

fi

#====================================================

if [ $mono_test_sw == 1 ]; then

echo ============================================================================
echo "                   MonoPhone Testing             	        "
echo ============================================================================

utils/mkgraph.sh --mono $lang_dir $exp_dir/mono $exp_dir/mono/$graph_dir || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true $exp_dir/mono/$graph_dir $test_dir $exp_dir/mono/$decode_dir || exit 1;


fi

#====================================================

if [ $tri1_train_sw == 1 ]; then

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training           "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" $train_dir $lang_dir $exp_dir/mono $exp_dir/mono_ali || exit 1; 

for sen in 2500; do 
for gauss in 8; do 
gauss=$(($sen * $gauss)) 

echo "========================="
echo " Sen = $sen  Gauss = $gauss"
echo "========================="

steps/train_deltas.sh --cmd "$train_cmd" $sen $gauss $train_dir $lang_dir $exp_dir/mono_ali $exp_dir/tri1_8_$sen || exit 1; 

done;done

fi

#====================================================

if [ $tri1_test_sw == 1 ]; then

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas  Decoding            "
echo ============================================================================

for sen in 2500; do  

echo "========================="
echo " Sen = $sen "
echo "========================="

utils/mkgraph.sh $lang_dir $exp_dir/tri1_8_$sen $exp_dir/tri1_8_$sen/$graph_dir || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true $exp_dir/tri1_8_$sen/$graph_dir $test_dir $exp_dir/tri1_8_$sen/$decode_dir || exit 1;


done

fi

#====================================================

if [ $tri2_train_sw == 1 ]; then

echo ============================================================================
echo "                 tri2 : LDA + MLLT Training                    "
echo ============================================================================


steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" $train_dir $lang_dir $exp_dir/tri1_8_2500 $exp_dir/tri1_8_2500_ali || exit 1;

for numLeavesMLLT in 3500; do 
for numGaussMLLT in 8; do 
numGaussMLLT=$(($numLeavesMLLT * $numGaussMLLT)) 


echo "========================="
echo " Sen = $numLeavesMLLT  Gauss = $numGaussMLLT"
echo "========================="

steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT $numGaussMLLT $train_dir $lang_dir $exp_dir/tri1_8_2500_ali $exp_dir/tri2_8_$numLeavesMLLT || exit 1;

done;done

fi

#====================================================

if [ $tri2_test_sw == 1 ]; then

echo ============================================================================
echo "                 tri2 : LDA + MLLT Decoding                "
echo ============================================================================

for numLeavesMLLT in 3500; do 

echo "========================="
echo " Sen = $numLeavesMLLT "
echo "========================="

utils/mkgraph.sh $lang_dir $exp_dir/tri2_8_$numLeavesMLLT $exp_dir/tri2_8_$numLeavesMLLT/$graph_dir || exit 1;
steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true $exp_dir/tri2_8_$numLeavesMLLT/$graph_dir $test_dir $exp_dir/tri2_8_$numLeavesMLLT/$decode_dir || exit 1;

done

fi

#====================================================

if [ $tri3_train_sw == 1 ]; then

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training               "
echo ============================================================================
# Align tri2 system with train data.

steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true $train_dir $lang_dir $exp_dir/tri2_8_3500 $exp_dir/tri2_8_3500_ali || exit 1;

for numLeavesSAT in 2500; do 
for numGaussSAT in 8; do 
numGaussSAT=$(($numLeavesSAT * $numGaussSAT)) 

echo "========================="
echo " Sen = $numLeavesSAT  Gauss = $numGaussSAT"
echo "========================="
# From tri2 system, train tri3 which is LDA + MLLT + SAT.

steps/train_sat.sh --cmd "$train_cmd" \
 $numLeavesSAT $numGaussSAT $train_dir $lang_dir $exp_dir/tri2_8_3500_ali $exp_dir/tri3_8_$numLeavesSAT || exit 1;

done;done

fi

#====================================================

if [ $tri3_test_sw == 1 ]; then

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Decoding    Start             "
echo ============================================================================

for numLeavesSAT in 2500; do 

echo "========================="
echo " Sen = $numLeavesSAT "
echo "========================="

utils/mkgraph.sh $lang_dir $exp_dir/tri3_8_$numLeavesSAT $exp_dir/tri3_8_$numLeavesSAT/$graph_dir || exit 1;
steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true $exp_dir/tri3_8_$numLeavesSAT/$graph_dir $test_dir $exp_dir/tri3_8_$numLeavesSAT/$decode_dir || exit 1;
done

fi

#====================================================

if [ $sgmm_train_test_sw == 1 ]; then

echo ============================================================================
echo "                        SGMM2 Training and Testing                "
echo ============================================================================
# Align tri3 system with train data.

best_sen_tri3=2500
best_gauss_tri3=8
numGaussUBM=400

steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
$train_dir $lang_dir $exp_dir/tri3"_$best_gauss_tri3"_"$best_sen_tri3" $exp_dir/tri3"_$best_gauss_tri3"_"$best_sen_tri3"_ali || exit 1;

# Train UBM

steps/train_ubm.sh --cmd "$train_cmd" \
 $numGaussUBM $train_dir $lang_dir $exp_dir/tri3"_$best_gauss_tri3"_"$best_sen_tri3"_ali $exp_dir/ubm4 || exit 1;

for sen in 2500; do
for gauss in 8; do
 
product_sgmm=$(($sen * $gauss)) 

echo "==================================="
echo "Sennon=$sen  Gauss=$gauss"
echo "==================================="
# Train SGMM

steps/train_sgmm2.sh --cmd "$train_cmd" $sen $product_sgmm \
 $train_dir $lang_dir $exp_dir/tri3"_$best_gauss_tri3"_"$best_sen_tri3"_ali $exp_dir/ubm4/final.ubm $exp_dir/sgmm"_$gauss"_"$sen" || exit 1;

# Decode using SGMM

utils/mkgraph.sh $lang_dir $exp_dir/sgmm"_$gauss"_"$sen" $exp_dir/sgmm"_$gauss"_"$sen"/$graph_dir || exit 1;
steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"  --transform-dir $exp_dir/tri3"_$best_gauss_tri3"_"$best_sen_tri3"/$decode_dir $exp_dir/sgmm"_$gauss"_"$sen"/$graph_dir $test_dir $exp_dir/sgmm"_$gauss"_"$sen"/$decode_dir || exit 1;

done;done

fi

#====================================================
if [ $dnn_train_sw == 1 ]; then

echo ============================================================================
echo "                    DNN Hybrid Training tri3 Aligned                  "
echo ============================================================================
# Align tri3/sgmm system with train data.

steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 $train_dir $lang_dir $exp_dir/tri3_8_2500 $exp_dir/tri3_8_2500_ali || exit 1;

dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

 steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.015 \
 --final-learning-rate 0.002 --num-hidden-layers 5  \
 --num-jobs-nnet "$train_nj" --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  $train_dir $lang_dir $exp_dir/tri3_8_2500_ali $exp_dir/DNN_tri3_8_2500_ali

fi

#====================================================

if [ $dnn_test_sw == 1 ]; then

echo ============================================================================
echo "                    DNN Hybrid Testing tri3 Aligned                  "
echo ============================================================================

dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}"--skip-scoring true \
 --transform-dir $exp_dir/tri3_8_2500/$decode_dir $exp_dir/tri3_8_2500/$graph_dir $test_dir \
  $exp_dir/DNN_tri3_8_2500_ali/$decode_dir | tee $exp_dir/DNN_tri3_8_2500_ali/$decode_dir/decode.log

fi

#====================================================

if [ $lstm_train_sw == 1 ]; then

echo ============================================================================
echo "                    LSTM Hybrid Training tri3 Aligned                  "
echo ============================================================================
#LSTM training

steps/nnet3/lstm/train.sh --cmd "$train_cmd" --num-epochs 15 --cell-dim 256 --hidden-dim 1024 --num-lstm-layers 2 --lstm-delay " -1 -2 " \
 --initial-effective-lrate 0.0015 --final-effective-lrate 0.002 \
 $train_dir $lang_dir $exp_dir/tri3_8_3500_ali $exp_dir/LSTM_tri3_8_3500_ali

fi

#====================================================

if [ $lstm_test_sw == 1 ]; then

echo ============================================================================
echo "                    LSTM Hybrid Testing tri3 Aligned                  "
echo ============================================================================

dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/nnet3/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}"
 --transform-dir $exp_dir/tri3_8_3500/$decode_dir $exp_dir/tri3_8_3500/$graph_dir $test_dir \
 $exp_dir/LSTM_tri3_8_3500_ali/$decode_dir | tee $exp_dir/LSTM_tri3_8_3500_ali/$decode_dir/decode.log
fi

#====================================================

if [ $tdnn_train_sw == 1 ]; then

echo ============================================================================
echo "                    TDNN Hybrid Training tri3 Aligned                  "
echo ============================================================================
#Note: Check the  steps/nnet3/tdnn/train.sh to know whether we are using i-vectors or not
#Note: Please update the ivector features location in  steps/nnet3/tdnn/train.sh 

steps/nnet3/tdnn/train.sh --cmd "$train_cmd" --num-epochs 15 --minibatch-size 128 \
--num-jobs-initial 4 --num-jobs-final 4 --initial-effective-lrate 0.0015 --final-effective-lrate 0.002 \
$train_dir $lang_dir $exp_dir/tri3_8_2500_ali $exp_dir/TDNN_tri3_8_2500_ali

fi

#====================================================

if [ $tdnn_test_sw == 1 ]; then

echo ============================================================================
echo "                    TDNN Hybrid Testing tri3 Aligned                  "
echo ============================================================================
dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/nnet3/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}"
 --transform-dir $exp_dir/tri3_8_2500/$decode_dir $exp_dir/tri3_8_2500/$graph_dir $test_dir \
 $exp_dir/TDNN_tri3_8_2500_ali/$decode_dir | tee $exp_dir/TDNN_tri3_8_2500_ali/$decode_dir/decode.log

fi


#====================================================
if [ $bnf_train_sw == 1 ]; then

echo ============================================================================
echo "                    BNF Network Training tri3 Aligned                  "
echo ============================================================================
# Align tri3/sgmm system with train data.

steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 $train_dir $lang_dir $exp_dir/tri3_8_2500 $exp_dir/tri3_8_2500_ali || exit 1;

dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

 steps/nnet2/train_tanh_bottleneck.sh --initial-learning-rate 0.015 \
 --final-learning-rate 0.002 --num-hidden-layers 5  \
 --num-jobs-nnet "$train_nj" --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  $train_dir $lang_dir $exp_dir/tri3_8_2500_ali $exp_dir/BNF_tri3_8_2500_ali
fi
#====================================================
if [ $bnf_extract_sw == 1 ]; then

#for x in train test; do 
echo ============================================================================
echo "                    Extract BNF Features                 "
echo ============================================================================
steps/nnet2/dump_bottleneck_features.sh data/$x data/bnf_$x $exp_dir/BNF_tri3_8_2500_ali $mfccdir $exp_dir/dump_bnf

steps/append_feats.sh data/$x data/bnf_$x data/combined_$x exp/append_mfcc_bnf $mfccdir
steps/compute_cmvn_stats.sh data/combined_$x $exp_dir/make_mfcc/data/combined_$x $mfccdir || exit 1;
utils/validate_data_dir.sh data/combined_$x;
fi

echo ============================================================================
echo "                     Training Testing Finished                      "
echo ============================================================================

