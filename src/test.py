from data_preprocess import preprocess
from utils import evaluate, evaluate_valid, data_partition, WarpSampler

preprocess()

print(data_partition())