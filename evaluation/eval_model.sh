# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J circEWS -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name shap_top500_features --split_name $split
# bsub -W 24:00 -J circEWS-random -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name shap_top500_features --split_name $split --random_classifier
# bsub -W 24:00 -J circEWS-lite -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name shap_top20_variables_MIMIC_BERN --split_name $split
# done


# for model in circewslite_train_binaryind_test_binaryind_usds_rev2_no_shapelets
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J trorig-teusds -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# for model in circewslite_train_orig_test_usds_with_shapelets
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J trorig-teusds -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# with shapelets
# for model in circewslite_binaryind_with_shapelets_redisc circewslite_train_binaryind_test_binaryind_usds_no_shapelets
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J onlydt -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# for model in only_reldatetime
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# #bjobs
# # bsub -W 4:00 -J onlydt -g /icuscore/event_based -R "rusage[mem=3072]" python compute_percentile.py --model_name $model --split_name $split
# bsub -W 120:00 -J onlydt -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# for model in circewslite_binarized_upsampled_downsampled_no_drugs circewslite_binaryind_v2
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J bin -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# for model in circewslite_binaryind_v2 circewslite_binarized_upsampled_downsampled_no_drugs
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 120:00 -J bin-random -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split --random_classifier
# done
# done


# for model in circewslite_train_orig_test_usds circewslite_train_usds_test_orig
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J usds -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done


# for model in circewslite_upsampled_downsampled_with_shapelets_redisc
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J usds -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# for model in shap_top500_features
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do

# for t in $(seq 0 5 60)
# do
# bsub -W 24:00 -J reset -g /icuscore/event_based -W 12:00 -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split --t_reset $t
# done

# for t in 5 $(seq 30 30 480)
# do
# bsub -W 24:00 -J silence -g /icuscore/event_based -W 12:00 -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split --t_reset 0 --t_silence $t
# done

# done
# done

# for model in  impute_only_forward_filling impute_no_imputation
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J imputation -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# for model in alt_endpoint_MAP60_with_shapelets alt_endpoint_MAP67_with_shapelets
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# # bsub -W 24:00 -J eps -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# bsub -W 24:00 -J eps-random -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split --random_classifier
# done
# done

# for model in downsampled_hirid_compact_orig
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -W 24:00 -J ds-mimic -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

# for model in Clinical All MWES
# do
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# echo bsub -W 24:00 -J baseline -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model --split_name $split
# done
# done

## train on mimic test on 
# for model in MIMIConly_
# do
# for split in random_0 random_1 random_2 random_3 random_4 
# do
# # bsub -W 24:00 -J mimiconly -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name shap_top20_variables_$model$split 
# bsub -W 24:00 -J mimiconly -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name shap_top20_variables_$model$split --random_classifier
# # bsub -W 24:00 -J mimiconlyss -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name shap_top20_variables_$model$split --use_subsample
# done
# done

# for split in $(seq 0 4)
# do
# tmp=_orig
# bsub -W 24:00 -g /icuscore/event_based -R "rusage[mem=3072]" -J mimiconly python precision_recall_revised.py --model_name ds_hirid_compact_MIMIConly_r$split$tmp  
# bsub -W 24:00 -g /icuscore/event_based -R "rusage[mem=3072]" -J mimiconly-rand python precision_recall_revised.py --model_name ds_hirid_compact_MIMIConly_r$split$tmp  --random_classifier
# done

# ### train on hirid test on mimic
# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# for i in $(seq 0 4)
# do
# bsub -W 24:00 -g /icuscore/event_based -R "rusage[mem=3072]" -J mimicval python precision_recall_revised.py --model_name shap_top20_variables_MIMIC --mimic_test_set $i --split_name $split
# bsub -W 24:00 -g /icuscore/event_based -R "rusage[mem=3072]" -J mimicval-rand python precision_recall_revised.py --model_name shap_top20_variables_MIMIC --mimic_test_set $i --split_name $split --random_classifier
# done
# done

# for model in shap_top500_features
#  # circewslite_upsampled_downsampled_with_shapelets_redisc shap_top20_variables_MIMIC_BERN 
# do

# for split in temporal_1 temporal_2 temporal_3 temporal_4 temporal_5 held_out
# do
# bsub -J alarms -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model  --split_name $split --fixed_recall 0.957
# # bsub -J alarms -g /icuscore/event_based -R "rusage[mem=3072]" python precision_recall_revised.py --model_name $model  --split_name $split
# done

# # bsub -J alarm_dist -g /icuscore/event_based -R "rusage[mem=2048]" python alarm_distribution.py --model_name $model
# # bsub -J recall_dist -g /icuscore/event_based -R "rusage[mem=2048]" python time2lastevent.py --model_name $model 

# done



for pyscript in circewslite_analysis.py
do

for batch in batch_27.h5 batch_28.h5 batch_29.h5 batch_30.h5
do
bsub -J circews-analysis -g /icuscore/event_based -R "rusage[mem=3072]" python $pyscript temporal_1 $batch
done

for batch in batch_30.h5 batch_31.h5 batch_32.h5 batch_33.h5 batch_34.h5
do
bsub -J circews-analysis -g /icuscore/event_based -R "rusage[mem=3072]" python $pyscript temporal_2 $batch
done

for batch in batch_34.h5 batch_35.h5 batch_36.h5 batch_37.h5
do
bsub -J circews-analysis -g /icuscore/event_based -R "rusage[mem=3072]" python $pyscript temporal_3 $batch
done

for batch in batch_37.h5 batch_38.h5 batch_39.h5 batch_40.h5 batch_41.h5
do
bsub -J circews-analysis -g /icuscore/event_based -R "rusage[mem=3072]" python $pyscript temporal_4 $batch
done

for batch in batch_41.h5 batch_42.h5 batch_43.h5 batch_44.h5
do
bsub -J circews-analysis -g /icuscore/event_based -R "rusage[mem=3072]" python $pyscript temporal_5 $batch
done

for batch in batch_47.h5 batch_48.h5 batch_49.h5
do
bsub -J circews-analysis -g /icuscore/event_based -R "rusage[mem=3072]" python $pyscript held_out $batch
done

done
