# fps with gs
python render_fps.py -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000

# fps with tcgs
python render.py -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000

# Profile original implementation with nsys
cd /vast/ml8347/ml_sys/final_project/gs_profile
nsys profile \
    --output=/vast/ml8347/ml_sys/final_project/log/profile_nsys/gs_profile_$(date +%Y%m%d_%H%M%S) \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    python render_fps.py -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000

# Profile TensorCore implementation with nsys
cd /vast/ml8347/ml_sys/final_project/tcgs_profile
nsys profile \
    --output=/vast/ml8347/ml_sys/final_project/log/profile_nsys/tcgs_profile_$(date +%Y%m%d_%H%M%S) \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    python render.py -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000


