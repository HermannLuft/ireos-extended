#!/bin/bash
sbatch lymphography_job
sbatch hpc_job
sed -i 's/Hepatitis/Arrhythmia/g' hpc_job
sbatch hpc_job
sed -i 's/Arrhythmia/Parkinson/g' hpc_job
sbatch hpc_job
sed -i 's/Parkinson/WBC/g' hpc_job
sed -i 's/norm_05_/norm_/g' hpc_job
sbatch hpc_job
sed -i 's/WBC/Hepatitis/g' hpc_job
sed -i 's/norm_/norm_05_/g' hpc_job
