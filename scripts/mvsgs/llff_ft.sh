export CUDA_VISIBLE_DEVICES=0
data_dir="nerf_llff_data"
dir_ply="mvsgs_pointcloud"
scenes=(fern flower fortress horns leaves orchids room trex)
iter=2500

python run.py --type evaluate --cfg_file configs/mvsgs/llff_eval.yaml save_ply True dir_ply $dir_ply

for scene in ${scenes[@]}
do  
python lib/train.py  --eval --iterations $iter -s $data_dir/$scene -p $dir_ply
python lib/render.py -c -m output/$scene --iteration $iter -p $dir_ply
python lib/metrics.py -m output/$scene
done

python lib/utils/read_json.py --root output --scenes ${scenes[@]} --iter $iter