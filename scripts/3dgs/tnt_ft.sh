export CUDA_VISIBLE_DEVICES=0
data_dir="tnt_data"
scenes=(Train Truck)

for scene in ${scenes[@]}
do  
python lib/train.py  --eval -s $data_dir/$scene
python lib/render.py -c -m output/$scene
python lib/metrics.py -m output/$scene
done

python lib/utils/read_json.py --root output --scenes ${scenes[@]}