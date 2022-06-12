export CUDA_VISIBLE_DEVICES=1

python main.py --output_dir experiments --comment "classification for flowformer" --name EthanolConcentration_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/EthanolConcentration --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification for flowformer" --name Handwriting_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/Handwriting --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification for flowformer" --name Heartbeat_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/Heartbeat --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification for flowformer" --name JapaneseVowels_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/JapaneseVowels --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification from Scratch" --name SelfRegulationSCP2_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SelfRegulationSCP2 --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification from Scratch" --name PEMS-SF_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/PEMS-SF --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 400 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification from Scratch" --name SelfRegulationSCP1_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SelfRegulationSCP1 --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification from Scratch" --name UWaveGestureLibrary_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/UWaveGestureLibrary --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification from Scratch"   --name SpokenArabicDigits_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SpokenArabicDigits --data_class tsra  --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer Adam  --pos_encoding learnable  --task classification  --key_metric accuracy
#
#python main.py --output_dir experiments --comment "classification from Scratch" --name FaceDetection_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/FaceDetection --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy