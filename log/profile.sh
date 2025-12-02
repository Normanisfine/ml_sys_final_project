# Set working directory
cd /scratch/ml8347/PCV/3dgs/gaussian-splatting

# Run FPS profiling for garden_3dgs model
python render_fps.py -m /scratch/ml8347/mlsys/tcgs/output/garden_3dgs --iteration 30000

cd /scratch/ml8347/mlsys/tcgs/3DGSTensorCore
python render.py -m /scratch/ml8347/mlsys/tcgs/output/garden_3dgs --iteration 30000

# Profile original implementation with nsys
nsys profile \
    --output=/scratch/ml8347/mlsys/tcgs/profile/nsys_file/gs_profile_$(date +%Y%m%d_%H%M%S) \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    python render_fps.py -m /scratch/ml8347/mlsys/tcgs/output/garden_3dgs --iteration 30000

    # Profile TensorCore implementation with nsys
nsys profile \
    --output=/scratch/ml8347/mlsys/tcgs/profile/nsys_file/tcgs_profile_$(date +%Y%m%d_%H%M%S) \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    python render.py -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000


# profile with gs
python render_fps.py -m /vast/ml8347/ml_sys/final_project/assets/garden_3dgs --iteration 30000