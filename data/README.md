# Dataset Directory
# =================
# Download datasets with:  python data/download_datasets.py --download
#
# DATASETS USED:
#   1. SmartBugs Curated   (143 contracts)     https://github.com/smartbugs/smartbugs-curated
#   2. SolidiFI Benchmark  (9,369 contracts)   https://github.com/DependableSystemsLab/SolidiFI-benchmark
#   3. SmartBugs Wild      (47,518 contracts)   https://github.com/smartbugs/smartbugs-wild
#
# HOW THEY ARE ACCESSED:
#   The training script (models/train_model.py) generates synthetic HCPG
#   feature vectors that simulate the graph properties extracted from real
#   contracts. The real datasets can be used for extended training and
#   benchmarking by parsing .sol files with solc and building HCPG graphs.
