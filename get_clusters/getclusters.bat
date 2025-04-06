@echo off
REM Windows equivalent of getclusters.sh

set scriptDir=..\scripts
set inputPath=..\data
set input=text.in
set model=bert-base-cased
set sentence_length=300
set layer=12
set working_file=%input%.tok.sent_len

echo Step 1: Tokenizing text with moses tokenizer...
perl %scriptDir%\tokenizer\tokenizer.perl -l en -no-escape < %inputPath%\%input% > %input%.tok

echo Step 2: Filtering sentences with max length of %sentence_length%...
python %scriptDir%\sentence_length.py --text-file %input%.tok --length %sentence_length% --output-file %input%.tok.sent_len

echo Step 3: Activating neurox environment and extracting layer-wise activations...
call conda activate neurox_pip
python -m neurox.data.extraction.transformers_extractor --decompose_layers --filter_layers %layer% --output_type json %model% %working_file% %working_file%.activations.json

echo Step 4: Creating dataset file with word and sentence indexes...
python %scriptDir%\create_data_single_layer.py --text-file %working_file% --activation-file %working_file%.activations-layer%layer%.json --output-prefix %working_file%

echo Step 5: Calculating vocabulary size...
python %scriptDir%\frequency_count.py --input-file %working_file% --output-file %working_file%.words_freq

echo Step 6: Filtering tokens for clustering...
set minfreq=0
set maxfreq=1000000
set delfreq=1000000
python %scriptDir%\frequency_filter_data.py --input-file %working_file%-dataset.json --frequency-file %working_file%.words_freq --sentence-file %working_file%-sentences.json --minimum-frequency %minfreq% --maximum-frequency %maxfreq% --delete-frequency %delfreq% --output-file %working_file%

echo Step 7: Activating clustering environment and running clustering...
call conda activate clustering

if not exist results mkdir results
set DATASETPATH=%working_file%_min_%minfreq%_max_%maxfreq%_del_%delfreq%-dataset.json
set VOCABFILE=processed-vocab.npy
set POINTFILE=processed-point.npy
set RESULTPATH=.\results
set CLUSTERS=50,100,50

echo Extracting data...
python -u %scriptDir%\extract_data.py --input-file %DATASETPATH%

echo Creating clusters...
python -u %scriptDir%\get_agglomerative_clusters.py --vocab-file %VOCABFILE% --point-file %POINTFILE% --output-path %RESULTPATH% --cluster %CLUSTERS% --range 1

echo DONE!
pause
