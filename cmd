python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/ssd_inception_v2_coco.config
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_inception_v2_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-3428 --output_directory models
python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/ssd_mobilenet_v1_coco.config
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-20000 --output_directory models
