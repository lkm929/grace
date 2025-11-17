import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewbirnbaum/full-head-mri-and-segmentation-of-stroke-patients")

print("Path to dataset files:", path)