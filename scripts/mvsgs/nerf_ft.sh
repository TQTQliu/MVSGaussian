export CUDA_VISIBLE_DEVICES=1
data_dir="nerf_synthetic"
dir_ply="mvsgs_pointcloud"
scenes=(chair drums ficus hotdog lego materials mic ship)
iter=5000
python run.py --type evaluate --cfg_file configs/mvsgs/nerf_eval.yaml save_ply True dir_ply $dir_ply

for scene in ${scenes[@]}
do  
python lib/train.py  --eval --iterations $iter -s $data_dir/$scene -p $dir_ply
python lib/render.py -m output/$scene --iteration $iter -p $dir_ply
python lib/metrics.py -m output/$scene
done

python lib/utils/read_json.py --root output --scenes ${scenes[@]} --iter $iter