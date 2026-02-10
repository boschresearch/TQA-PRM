#pkill -f fastchat.serve.controller 
#pkill -f utils.vllm_worker 
#echo "Waiting for port 21002 to be free..."

#sleep 10
echo "Intializing Controller..."
python3 -m fastchat.serve.controller --port=21001 &
sleep 5
echo "Intializing Worker..."
python3 -m utils.vllm_worker --port=21002 --model-path /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --num-gpus=1 &

sleep 60
#echo "Worker Starts"


#/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
#/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/