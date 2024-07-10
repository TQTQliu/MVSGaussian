export CUDA_VISIBLE_DEVICES=1
data_dir="nerf_llff_data"
scenes=(fern flower fortress horns leaves orchids room trex)

for scene in ${scenes[@]}
do  
python lib/train.py  --eval -s $data_dir/$scene
python lib/render.py -c -m output/$scene
python lib/metrics.py -m output/$scene
done

python lib/utils/read_json.py --root output --scenes ${scenes[@]}