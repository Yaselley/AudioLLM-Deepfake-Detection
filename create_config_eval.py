import os
import yaml
import copy

run_file = open("/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/eval_suite.txt", "a+")

for checkpoint in os.listdir("/netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/suite2"):
    run_file.write(f"sh run_eval.sh /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/suite2/{checkpoint} &> /netscratch/yelkheir/DeepFense/AudioLLM-Deepfake-Detection/outputs/out/{checkpoint}_eval.txt & \n")