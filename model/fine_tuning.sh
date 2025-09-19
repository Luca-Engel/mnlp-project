#!/bin/bash -l
#SBATCH --chdir /scratch/izar/dkopp/project-m2-2024-start-unk-stop-pad
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 90G
#SBATCH --time 04:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

date
module load gcc python

source ~/venvs/mnlp_m2/bin/activate

pip install --upgrade pip
pip3 install -r model/requirements.txt

echo "Install Dependencies complete"
echo "Running the code"

python3 model/fine_tuning_script.py


echo "Run completed on $(hostname)"
sleep 2
date
