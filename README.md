# Face_image_score
My implementation of Face image quality score related papers/projects. 


# Papers
### 1. SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance
**source**: https://arxiv.org/abs/2103.05977

**official repo**: https://github.com/Tencent/TFace/tree/quality

Unofficial implementation of SDD-FIQA, pseudo label generation code. 

Based on my test, this implementation is 30x faster than the official repo, and reduce the memory usage to a reasonable amount when dealing with large datasets(5 million ids).

### usage:

**Required Parameters**:


`--image-list-file`: This parameter specifies the path to the file containing the list of images. Each line in the file should represent a unique image, typically with paths or identifiers for the images you wish to process.

`--image-root`: This parameter specifies the root directory where the images are stored. 
The script will use this path to locate and process each image listed in the --image-list-file.

`--score-dst`: 
This parameter defines the destination directory where the output scores will be stored after processing the images.

`--id-key-index`: 
This parameter indicates the index of the key that uniquely identifies each image or subject in the path.

Adjust this index based on the structure of your file paths to correctly extract the unique identifiers.

such as: -3 in "/src/**person_id**/feature/name.jpg" indicates the index of the key "person_id" in the path.

`--feature-root`: 
The root directory where the image features are stored. Such face feature can be extracted using the face recognition model.


`-load-feature`: This flag indicates whether to load features into memory from the --feature-root directory. 
If set to True, the script will attempt to load features instead of reading them in each calculation iterations.

Set to False by default due to memory constraints.

`--process-num`: 
This parameter defines the number of processes to use for parallel computation, potentially speeding up the processing time.

`--fix-num`: This parameter is used to set a fixed number for feature comparison(Default: 24), more information check the original paper.

```python
python Generate_labels.py --image-list-file /path/to/image_list.txt \
                   --image-root /path/to/image_root \
                   --score-dst /path/to/score_dst \
                   --id-key-index -3 \
                   --feature-root /path/to/feature_root \
                   --load-feature False \
                   --process-num 10 \
                   --fix-num 24
```

### 2. Harnessing Unrecognizable Faces for Improving Face Recognition
**Source**:
https://arxiv.org/pdf/2106.04112.pdf