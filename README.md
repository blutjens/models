# Deeplab installation: From TensorFlow Models
## Co-authors: Bjorn Lutjens, Michael Everett

## Installation
Download tensorflow/models
Download voc dataset
Create AWS instance (e.g. g3.8xlarge, 150GB storage)

### Set ENV vars for every new terminal
'''
AWS_SSH_ADDRESS="ec2-18-232-123-147.compute-1.amazonaws.com" 
AWS_ROOT="/home/$USER/Desktop/acl/aws" # Path of aws key
PROJECT_NAME="crowdai"
PROJECT_ROOT="/home/$USER/Desktop/$PROJECT_NAME/"
'''

### Pass folder from computer to AWS (zip datasets with many files) 
'''
ssh -i $AWS_ROOT/bjorn.pem ubuntu@$AWS_SSH_ADDRESS 'mkdir -p /home/ubuntu/'$PROJECT_NAME
scp -i $AWS_ROOT/bjorn.pem -r $PROJECT_ROOT/models/research/deeplab/ ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/deeplab
scp -i $AWS_ROOT/bjorn.pem -r $PROJECT_ROOT/models/research/slim/ ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/slim
scp -i $AWS_ROOT/bjorn.pem -r $PROJECT_ROOT/models/research/deeplab/datasets/ ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/deeplab/
scp -i $AWS_ROOT/bjorn.pem -r $PROJECT_ROOT/models/crowdai_related ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/
'''

### Connect with ssh into AWS
'''
ssh -L localhost:6022:localhost:6006 -L localhost:8822:localhost:8888 -i $AWS_ROOT/bjorn.pem ubuntu@$AWS_SSH_ADDRESS
'''

### Build docker in AWS terminal
'''
cd /home/ubuntu/crowdai/deeplab/
source build_deeplab_docker.sh
'''
### Start docker and add directory to python path
'''
source run_deeplab_docker.sh
cd home/ubuntu/crowdai/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
'''
### Test tensorflow implementation
'''
python deeplab/model_test.py -v
'''
### Download and set pretrained checkpoint
'''
TF_INIT_ROOT="http://download.tensorflow.org/models"
CKPT_NAME="deeplabv3_mnv2_pascal_train_aug" # MobileNetv2
TF_INIT_CKPT="${CKPT_NAME}_2018_01_29.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"
'''

### Run on MS-coco building crowdai dataset
'''
cd ~/Desktop/crowdai/mapping-challenge-starter-kit/crowdai/
Download dataset from https://www.crowdai.org/challenges/mapping-challenge/dataset_files
tar -xf *
Set up file structure
-crowdai
--crowdai_data
--- crowdai_data_voc
---- ImageSets
----- Segmentation
---- JPEGImages
----- train
----- val
---- SegmentationClassRaw 
----- train
----- val
---- exp
--tfrecord
'''
####Convert crowdai dataset from MS-coco format to Pascal VOC format:
Run the full notebook for all datasets (can take several hours)
'''
~/Desktop/crowdai/mapping-challenge-starter-kit/Dataset Utils.ipynb
'''
Copy JPEG images and Segmentation Mask ( do for val and train ) 
''' 
DATASET_SPLIT="train"
JPEG_PATH="crowdai/crowdai_data/crowdai_data_voc/JPEGImages"
SEG_PATH="crowdai/crowdai_data/crowdai_data_voc/SegmentationClassRaw/"
SEG_PATH_TXT="crowdai/crowdai_data/crowdai_data_voc/ImageSets/Segmentation"
SEG_JPEG_PATH=${JPEG_PATH}
cd ~/Desktop/crowdai/mapping-challenge-starter-kit/${SEG_JPEG_PATH}
zip -r ${DATASET_SPLIT}.zip ${DATASET_SPLIT}
scp -i $AWS_ROOT/bjorn.pem -r $PROJECT_ROOT/mapping-challenge-starter-kit/${SEG_JPEG_PATH}/${DATASET_SPLIT}.zip ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/deeplab/datasets/crowdai_data/crowdai_data_voc/
scp -i $AWS_ROOT/bjorn.pem -r $PROJECT_ROOT/mapping-challenge-starter-kit/${SEG_PATH_TXT}/train.txt ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/deeplab/datasets/crowdai_data/crowdai_data_voc/ImageSets/Segmentation
cd /home/ubuntu/$PROJECT_NAME/deeplab/datasets
unzip ${DATASET_SPLIT}.zip
'''
####Convert Pascal VOC format to tfrecord (for train and val)
'''
python ./build_crowdai_data.py \
  --image_folder="crowdai_data/crowdai_data_voc/JPEGImages/train" \
  --semantic_segmentation_folder="crowdai_data/crowdai_data_voc/SegmentationClassRaw/train" \
  --list_folder="crowdai_data/crowdai_data_voc/ImageSets/Segmentation" \
  --image_format="jpg" \
  --label_format="png" \
  --output_dir="crowdai_data/tfrecord"
'''
### Set environment variables for crowdai dataset
'''
TRAIN_LOGDIR_CROWDAI="${WORK_DIR}/${DATASET_DIR}/crowdai_data/exp/train_on_train_set_mobilenetv2/train"
EVAL_LOGDIR_CROWDAI="${WORK_DIR}/${DATASET_DIR}/crowdai_data/exp/train_on_train_set_mobilenetv2/eval"
VIS_LOGDIR_CROWDAI="${WORK_DIR}/${DATASET_DIR}/crowdai_data/exp/train_on_train_set_mobilenetv2/vis"
EXPORT_DIR_CROWDAI="${WORK_DIR}/${DATASET_DIR}/crowdai_data/exp/train_on_train_set_mobilenetv2/export"
CROWDAI_DATASET="${WORK_DIR}/${DATASET_DIR}/crowdai_data/tfrecord"
mkdir -p "${TRAIN_LOGDIR_CROWDAI}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

TF_INIT_CKPT="${WORK_DIR}/${DATASET_DIR}/crowdai_data/train/model.ckpt-10000"
TF_INIT_CKPT="${INIT_FOLDER}/deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000"
'''

### Train the model
'''
NUM_ITERATIONS=30000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --num_clones=2 \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=8 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${TF_INIT_CKPT}" \
  --train_logdir="${TRAIN_LOGDIR_CROWDAI}" \
  --dataset="crowdai" \
  --dataset_dir="${CROWDAI_DATASET}"
'''

  
### Display training progress
'''
cd /home/ubuntu/crowdai/deeplab/datasets/crowdai_data
tensorboard --logdir=train
'''

### Evaluate the model
'''
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --checkpoint_dir="${WORK_DIR}/${DATASET_DIR}/crowdai_data/train" \
  --eval_logdir="${EVAL_LOGDIR_CROWDAI}" \
  --dataset_dir="${CROWDAI_DATASET}" \
  --max_number_of_evaluations=1
'''

### Visualize the results
'''
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="mobilenet_v2" \
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --checkpoint_dir="${WORK_DIR}/${DATASET_DIR}/crowdai_data/train" \
  --vis_logdir="${VIS_LOGDIR_CROWDAI}" \
  --dataset_dir="${CROWDAI_DATASET}" \
  --max_number_of_iterations=1
'''

Copy visualizations back to computer
'''
cd /home/ubuntu/$PROJECT_NAME/deeplab/datasets/crowdai_data/exp/
zip -r segmentation_results.zip segmentation_results
scp -i $AWS_ROOT/bjorn.pem -r ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/deeplab/datasets/crowdai_data/exp/segmentation_results.zip $PROJECT_ROOT/mapping-challenge-starter-kit/crowdai/crowdai_data/
'''

Copy crowdai dataset to aws



# Train and evaluate on Pascal VOC 2012 dataset
### Download, unpack and convert datasets to tfrecord
cd "/home/ubuntu/crowdai/deeplab/datasets"
sh download_and_convert_voc2012.sh
(If download doesn't work, download manually from "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")

### Set environment variables
'''
cd /home/ubuntu/crowdai
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"
PASCAL_FOLDER="pascal_voc_seg"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"
PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"
'''


### Train the model
# Train 10 iterations.
NUM_ITERATIONS=10
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="trainval" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/model.ckpt-30000" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}"

### Evaluate the model
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_evaluations=1

### Visualize the results
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="mobilenet_v2" \
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_iterations=1

### Pass visualizations back to computer
scp -i $AWS_ROOT/bjorn.pem -r ubuntu@$AWS_SSH_ADDRESS:/home/ubuntu/$PROJECT_NAME/deeplab/datasets/pascal_voc_seg/exp/train_on_trainval_set/vis $PROJECT_ROOT/models/research/deeplab/datasets/pascal_voc_seg/exp/train_on_trainval_set/vis



## License

[Apache License 2.0](LICENSE)
