export CUDA_VISIBLE_DEVICES=2
data_dir="nerf_synthetic"
scenes=(chair drums ficus hotdog lego materials mic ship)

for scene in ${scenes[@]}
do  
python lib/train.py  --eval  -s $data_dir/$scene 
python lib/render.py -m output/$scene 
python lib/metrics.py -m output/$scene
done

python lib/utils/read_json.py --root output --scenes ${scenes[@]}