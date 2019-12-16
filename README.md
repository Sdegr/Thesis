# Bachelor Thesis Sil de Graaf.
Training two Transformer models in combination with the use of BPE with NL-EN and ZH-EN. Investigate which of the two source languages (Dutch vs Chinese) is more brittle to synthetic noise.


# Requirements
- Bilingual corpora
- A GPU
- Python
- Moses Tokenizer
- ~~Stanford Segmenter~~
- OpenNMT-py
- A Python virtual environment containing: \
 (pip install) --upgrade pip \
 (pip install) torch \
 (pip install) torchvision \
 (pip install) torchtext \
 (pip install) configargparse
- A script to create synthetic noise (swap_del.py)
- Perl's multi-bleu


# STEP 1 - Obtaining data

Two corpora are used for this thesis. The first corpus is a Dutch -> English corpus and the other corpus \
is a Chinese -> English corpus. 
These corpora are downloaded from;

https://wit3.fbk.eu/mt.php?release=2014-01

The corpora (containing training/dev/test) can be found in my thesis repository. \
Thesis/dutch-english\
Thesis/chinese-english

# STEP 2 - Cleaning data (Dutch -> English)

The training, development and test data all contained unwanted tags. These are removed using the following lines of code;

Clean training files:

$ sed -i.bak '/^<url>.*url>$/d' train.tags.nl-en.en \
$ sed -i.bak '/^<description>.*description>$/d' train.tags.nl-en.en \
$ sed -i.bak '/^<keywords>.*keywords>$/d' train.tags.nl-en.en \
$ sed -i.bak '/^<title>.*title>$/d' train.tags.nl-en.en \
$ sed -i.bak '/^<talkid>.*talkid>$/d' train.tags.nl-en.en 

$ sed -i.bak '/^<url>.*url>$/d' train.tags.nl-en.nl \
$ sed -i.bak '/^<description>.*description>$/d' train.tags.nl-en.nl \
$ sed -i.bak '/^<keywords>.*keywords>$/d' train.tags.nl-en.nl \
$ sed -i.bak '/^<title>.*title>$/d' train.tags.nl-en.nl \
$ sed -i.bak '/^<talkid>.*talkid>$/d' train.tags.nl-en.nl 

Clean development files:

$ sed -i.bak '/^<talkid>.*talkid>$/d' dev.en.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' dev.en.xml \
$ sed -i.bak '/^<url>.*url>$/d' dev.en.xml \
$ sed -i.bak '/^<description>.*description>$/d' dev.en.xml \
$ sed -i.bak '/^<title>.*title>$/d' dev.en.xml 

$ sed -i.bak '/^<talkid>.*talkid>$/d' dev.nl.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' dev.nl.xml \
$ sed -i.bak '/^<url>.*url>$/d' dev.nl.xml \
$ sed -i.bak '/^<description>.*description>$/d' dev.nl.xml \
$ sed -i.bak '/^<title>.*title>$/d' dev.nl.xml 

Clean test files:

$ sed -i.bak '/^<talkid>.*talkid>$/d' test.en.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' test.en.xml \
$ sed -i.bak '/^<url>.*url>$/d' test.en.xml \
$ sed -i.bak '/^<description>.*description>$/d' test.en.xml \
$ sed -i.bak '/^<title>.*title>$/d' test.en.xml 

$ sed -i.bak '/^<talkid>.*talkid>$/d' test.nl.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' test.nl.xml \
$ sed -i.bak '/^<url>.*url>$/d' test.nl.xml \
$ sed -i.bak '/^<description>.*description>$/d' test.nl.xml \
$ sed -i.bak '/^<title>.*title>$/d' test.nl.xml 



# STEP 3 - Converting XML files (dev/test) to text (Dutch -> English)

The development and test files were in XML format and had to be converted to text files in order to work with them.

$ python3
>>> file = open("test.nl.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('test.nl.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags)

$ python3
>>> file = open("test.en.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('test.en.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags) 

$ python3
>>> file = open("dev.nl.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('dev.nl.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags) 

$ python3
>>> file = open("dev.en.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('dev.en.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags) 

# STEP 4 - Tokenization (Dutch -> English)

wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl \
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en \

$ perl tokenizer.perl -l en -q < train.en.txt > train.en.tok \
$ perl tokenizer.perl -l en -q < train.nl.txt > train.nl.tok \
$ perl tokenizer.perl -l en -q < test.nl.txt > test.nl.tok \
$ perl tokenizer.perl -l en -q < test.en.txt > test.en.tok \
$ perl tokenizer.perl -l en -q < dev.en.txt > dev.en.tok \
$ perl tokenizer.perl -l en -q < dev.nl.txt > dev.nl.tok 

# STEP 5 - Get access to a GPU

The High Performance Computing (HPC) cluster from the RUG is used for this thesis (Peregrine). Containing 36 Tesla v100 gpu's.

# STEP 6 - Cloning OpenNMT

In order to perform steps 8,9,10 and 11, Github had to be cloned to Peregrine.

git clone https://github.com/OpenNMT/OpenNMT

# STEP 7 - Creating a Python virtual environment

Because of restrictions on Peregrine, a Python virtual environment had to be created in order to install the correct modules.

* Create the virtual environment: \
python3 -m venv thesis-env 
* Activate the virtual environment: \
source thesis-env/bin/activate

In the virtual environment install the following modules.

 pip install --upgrade pip \
 pip install torch \
 pip install torchvision \
 pip install torchtext \
 pip install configargparse


# STEP 8 - Applying BPE (Dutch -> English)

(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/learn_bpe.py -s 40000 < OpenNMT-py/nl-en/train.nl.tok > bpe-codes.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.src < OpenNMT-py/nl-en/train.nl.tok > train.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.src < OpenNMT-py/nl-en/dev.nl.tok > valid.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.src < OpenNMT-py/nl-en/test.nl.tok > test.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/learn_bpe.py -s 40000 < OpenNMT-py/nl-en/train.en.tok > bpe-codes.tgt \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.tgt < OpenNMT-py/nl-en/test.en.tok > test.tgt \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.tgt < OpenNMT-py/nl-en/dev.en.tok > valid.tgt \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.tgt < OpenNMT-py/nl-en/train.en.tok > train.tgt


# STEP 9 - Preprocess the data (Dutch -> English)

python OpenNMT-py/preprocess.py -train_src train.src -train_tgt train.tgt -valid_src valid.src -valid_tgt valid.tgt -save_dat
a preprocessed -src_seq_length 100 -tgt_seq_length 100 -seed 100 -log_file log.preprocess

# STEP 10 - Creating a Batch File (Dutch -> English)

\#!/bin/bash \
\#SBATCH --job-name="nl-en" \
\#SBATCH --time=08:30:00 \
\#SBATCH --ntasks=1 \
\#SBATCH --mem=10GB \
\#SBATCH --partition=gpu \
\#SBATCH --gres=gpu:v100:1 \
\#SBATCH --mail-type=ALL \
\#SBATCH --mail-user=S.M.de.Graaf.3@student.rug.nl 

source thesis-env/bin/activate \
module load Python \
\# pip install --upgrade pip \
\# pip install torch \
\# pip install torchvision \
\# pip install torchtext \
\# pip install configargparse \
\# pip install OpenNMT-py \

export CUDA_VISIBLE_DEVICES=0 \

python  OpenNMT-py/train.py -data pre_NL/preprocessed -save_model train_NL/NL-en_model \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 60000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -log_file logs/NL/log.train -world_size 1 -gpu_ranks 0
        
# STEP 11 - Training (Dutch -> English)

sbatch script-nl.sh

# STEP 12 - Create a Batch File (Dutch -> English)

#!/bin/bash\
\#SBATCH --job-name="translate-nl-en"\
\#SBATCH --time=00:10:00\
\#SBATCH --ntasks=1\
\#SBATCH --mem=10GB\
\#SBATCH --partition=gpu\
\#SBATCH --gres=gpu:v100:1\
\#SBATCH --mail-type=ALL\
\#SBATCH --mail-user=S.M.de.Graaf.3@student.rug.nl

source thesis-env/bin/activate\
module load Python\
\# pip install --upgrade pip\
\# pip install torch\
\# pip install torchvision\
\# pip install torchtext\
\# pip install configargparse\
\# pip install OpenNMT-py


export CUDA_VISIBLE_DEVICES=0

python OpenNMT-py/translate.py -gpu 0 -model nl-en_model_step_60000.pt \
-src test.src -tgt test.tgt -replace_unk -output nl-en3.pred.atok


# STEP 13 - Translate (Dutch -> English)

sbatch script2.sh

# STEP 13 - Obtaining BLEU-scores for clean texts (Dutch -> English)

wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

perl multi-bleu.perl BPE_nl-en/test.tgt < nl-en.pred.atok

BLEU = 31.22, 61.7/38.7/26.1/18.0

# STEP 14 - Apply synthetic noise to the Dutch test file by using the swap_del.py script (Dutch -> English)

python3 swap_del.py test.nl.txt swap.nl.txt

swap_del.py is found under Thesis/swap_del.py


# STEP 15 - Obtaining BLEU-scores for noisy texts (Dutch -> English)

BLEU = 21.79 (without BPE on test file, but after tokenizaten)


# STEP 16 - Compare BLEU scores of STEP 13 and 15 (Dutch -> English)



# STEP 17 - Clean files (Chinese -> English)

Clean training files:

$ sed -i.bak '/^<talkid>.*talkid>$/d' train.en \
$ sed -i.bak '/^<keywords>.*keywords>$/d' train.en \
$ sed -i.bak '/^<url>.*url>$/d' train.en \
$ sed -i.bak '/^<description>.*description>$/d' train.en \
$ sed -i.bak '/^<title>.*title>$/d' train.en
 
Note: see STEP 19! 
 
$ sed -i.bak '/^<talkid>.*talkid>$/d' train.zh \
$ sed -i.bak '/^<keywords>.*keywords>$/d' train.zh \
$ sed -i.bak '/^<url>.*url>$/d' train.zh \
$ sed -i.bak '/^<description>.*description>$/d' train.zh \
$ sed -i.bak '/^<title>.*title>$/d' train.zh

Clean development files:

$ sed -i.bak '/^<talkid>.*talkid>$/d' dev.zh.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' dev.zh.xml \
$ sed -i.bak '/^<url>.*url>$/d' dev.zh.xml \
$ sed -i.bak '/^<description>.*description>$/d' dev.zh.xml \
$ sed -i.bak '/^<title>.*title>$/d' dev.zh.xml
 
$ sed -i.bak '/^<talkid>.*talkid>$/d' dev.en.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' dev.en.xml \
$ sed -i.bak '/^<url>.*url>$/d' dev.en.xml \
$ sed -i.bak '/^<description>.*description>$/d' dev.en.xml \
$ sed -i.bak '/^<title>.*title>$/d' dev.en.xml 
 
 Clean test files:
 
$ sed -i.bak '/^<talkid>.*talkid>$/d' test.zh.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' test.zh.xml \
$ sed -i.bak '/^<url>.*url>$/d' test.zh.xml \
$ sed -i.bak '/^<description>.*description>$/d' test.zh.xml \
$ sed -i.bak '/^<title>.*title>$/d' test.zh.xml 
 
$ sed -i.bak '/^<talkid>.*talkid>$/d' test.en.xml \
$ sed -i.bak '/^<keywords>.*keywords>$/d' test.en.xml \
$ sed -i.bak '/^<url>.*url>$/d' tesv.en.xml \
$ sed -i.bak '/^<description>.*description>$/d' test.en.xml \
$ sed -i.bak '/^<title>.*title>$/d' test.en.xml



# STEP 18 - Converting XML files (dev/test) to text (Chinese -> English)

$ python3 
>>> file = open("dev.en.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('dev.en.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags)

$ python 3 
>>> file = open("dev.zh.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('dev.zh.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags)

$ python3 
>>> file = open("test.en.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('test.en.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags)

$ python3 
>>> file = open("test.zh.txt", "wb") \
>>> from lxml import etree \
>>> tree = etree.parse('test.zh.xml') \
>>> notags = etree.tostring(tree, encoding='utf8', method='text') \
>>> file.write(notags)




# STEP 19 - Tokenization (Chinese -> English)

Download a chinese prefix to tokenize to chinese!

wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.zh

$ perl tokenizer.perl -l en -q < train.en > train.en.tok \
$ perl tokenizer.perl -l zh -q < train.zh > train.zh.tok \
$ perl tokenizer.perl -l zh -q < test.zh.txt > test.zh.tok \
$ perl tokenizer.perl -l en -q < test.en.txt > test.en.tok \
$ perl tokenizer.perl -l en -q < dev.en.txt > dev.en.tok \
$ perl tokenizer.perl -l zh -q < dev.zh.txt > dev.zh.tok 


Or download the Stanford Segmenter. Use the following code to tokenize the Chinese files with the Stanford Segmenter:

./segment.sh pku cleandata/test.zh.txt UTF-8 0 > test.zh.tok \
./segment.sh pku cleandata/dev.zh.txt UTF-8 0 > dev.zh.tok \
./segment.sh pku cleandata/train.zh.txt UTF-8 0 > train.zh.tok

NOTE: There are three errors in the train.en file. If a combination of the \
Stanford Segmenter (for Chinese) and Moses Tokenizer (for English) is used. \
This can be solved by tokenizing the train.en file with the mozes tokenizer. \
After this, search for lines; 12636, 30908 and 39265 and removing these exact lines \
before STEP 17. (I only wrote down the number of the lines after I tokenized so the \
lines to be deleted have another number, but you should be able to find them with the instructions above. \
If this is not done correctly, the train.tgt file will contain three lines more than the train.src file \
and it is not possible to solve this by using any script/program. It took me 6 hours \
to find these errors manually..

NOTE: I would go with the Moses tokenizer eitherway.



# STEP 20 - Apply BPE (Chinese -> English)


(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/learn_bpe.py -s 40000 < OpenNMT-py/zh-en/train.nl.tok > bpe-codes.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.src < OpenNMT-py/zh-en/train.src > train.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.src < OpenNMT-py/zh-en/dev.src > valid.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.src < OpenNMT-py/zh-en/test.src > test.src \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/learn_bpe.py -s 40000 < OpenNMT-py/nl-en/train.tgt > bpe-codes.tgt \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.tgt < OpenNMT-py/zh-en/test.tgt > test.tgt \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.tgt < OpenNMT-py/zh-en/valid.tgt > valid.tgt \
(thesis-env) [s2615703@pg-gpu ~]$ OpenNMT-py/tools/apply_bpe.py -c bpe-codes.tgt < OpenNMT-py/zh-en/train.tgt > train.tgt






# STEP 21 - Preprocess (Chinese -> English)

python OpenNMT-py/preprocess.py -train_src train.src -train_tgt train.tgt -valid_src valid.src -valid_tgt valid.tgt -save_data preprocessed2 -src_seq_length 100 -tgt_seq_length 100 -seed 100 -log_file log.preprocess2


# STEP 22 - Creating a Batch File (Chinese -> English)

\#!/bin/bash \
\#SBATCH --job-name="zh-en" \
\#SBATCH --time=09:15:00 \
\#SBATCH --ntasks=1 \
\#SBATCH --mem=10GB \
\#SBATCH --partition=gpu \
\#SBATCH --gres=gpu:v100:1 \
\#SBATCH --mail-type=ALL \
\#SBATCH --mail-user=S.M.de.Graaf.3@student.rug.nl

source thesis-env/bin/activate \
module load Python \
\# pip install --upgrade pip \
\# pip install torch \
\# pip install torchvision \
\# pip install torchtext \
\# pip install configargparse \
\# pip install OpenNMT-py 


export CUDA_VISIBLE_DEVICES=0

python  OpenNMT-py/train.py -data preprocessed2 -save_model train_zh/zh-en_model \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 60000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -log_file logs/zh/log.train -world_size 1 -gpu_ranks 0

# STEP 23 - Training (Chinese -> English)

sbatch script-zh.sh

# STEP 24 - Create a batch file (Chinese -> English)

\#!/bin/bash \
\#SBATCH --job-name="zh-en" \
\#SBATCH --time=09:15:00 \
\#SBATCH --ntasks=1 \
\#SBATCH --mem=10GB \
\#SBATCH --partition=gpu \
\#SBATCH --gres=gpu:v100:1 \
\#SBATCH --mail-type=ALL \
\#SBATCH --mail-user=S.M.de.Graaf.3@student.rug.nl

source thesis-env/bin/activate \
module load Python \
\# pip install --upgrade pip \
\# pip install torch \
\# pip install torchvision \
\# pip install torchtext \
\# pip install configargparse \
\# pip install OpenNMT-py 


export CUDA_VISIBLE_DEVICES=0


python OpenNMT-py/translate.py -gpu 0 -model train_zh/zh-en_model_step_20000.pt \
-src bpe_zh/test.src -tgt bpe_zh/test.tgt -replace_unk -output translate_zh/zh-en.pred.atok


# STEP 25 - Translate (Chinese -> English)

sbatch translate-zh.sh


# STEP 26 - Obtaining BLEU-scores for clean texts (Chinese -> English)

perl multi-bleu.perl BPE_zh_en/test.tgt < translate_zh/zh-en.pred.atok

BLEU = 25.22, 55.4/34.2/24.8/18.1 (Moses tokenizer) \


# STEP 27 - Apply synthetic noise to test file by using the swap_del.py script (Chinese -> English)

python3 swap_del.py test.zh.txt swap.zh.txt

swap_del.py is found under Thesis/swap_del.py

# STEP 28 - Create a Batch File for noisy data (Chinese -> English)

\#!/bin/bash \
\#SBATCH --job-name="zh-en" \
\#SBATCH --time=09:15:00 \
\#SBATCH --ntasks=1 \
\#SBATCH --mem=10GB \
\#SBATCH --partition=gpu \
\#SBATCH --gres=gpu:v100:1 \
\#SBATCH --mail-type=ALL \
\#SBATCH --mail-user=S.M.de.Graaf.3@student.rug.nl

source thesis-env/bin/activate \
module load Python \
\# pip install --upgrade pip \
\# pip install torch \
\# pip install torchvision \
\# pip install torchtext \
\# pip install configargparse \
\# pip install OpenNMT-py 


export CUDA_VISIBLE_DEVICES=0


python OpenNMT-py/translate.py -gpu 0 -model train_zh/zh-en_model_step_20000.pt \
-src bpe_zh/test.src -tgt bpe_zh/test.tgt -replace_unk -output translate_zh/zh-noise-en.pred.atok

# STEP 28 - Translate noisy text (Chinese -> English)

sbatch translate-noise-zh.sh

# STEP 29 - Obtain BLEU-score for noisy text (Chinese -> English)

perl multi-bleu.perl bpe_zh/test.tgt < translate_zh/zh-noise-en.pred.atok

BLEU = 2.83 (without BPE on test file, but after tokenizaten)

# STEP 30 - Compare Dutch -> English BLEU-score with Chinese -> English BLEU-score











