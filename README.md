# MCCocs
## Requirements
torch == 1.12.1; torchvision == 0.13.1; timm == 0.6.13; scikit-learn; nilearn; nibabel

## Clustering Visualization
During the study, we attempted to map data points from different stages of the model to specific brain regions and visualize the clustering results. However, this process posed several technical challenges, which limited the quality and effectiveness of the visualizations:

> - When using the AAL brain atlas, the clustering involved 116 brain regions, requiring the differentiation of 116 distinct clusters by color. While it is technically feasible to assign a unique color to each cluster, human perception faces significant difficulty in distinguishing such a large number of colors. Moreover, some widely used atlases include even more brain regions, further reducing the intuitiveness and readability of the visualizations.
> - Mapping data points to brain regions necessitates constructing a robust and clear 3D brain model, which requires significant time and resources and goes beyond the scope of this study.

Nevertheless, due to the large number of brain regions and the less intuitive voxel-to-region mapping, we acknowledge that the initial visualization objectives were not fully achieved. To promote the development of more effective visualization tools, we have open-sourced the preliminary implementation of our code on GitHub. We hope this will attract contributions from researchers and developers, enabling innovative solutions and ultimately leading to the creation of more efficient and intuitive visualization tools for this field.
Nonetheless, we believe that promoting this would be highly worthwhile. To this end, we have open-sourced the preliminary implementation of our code in the script *cluster_visualize.py*. We hope this will attract contributions from researchers and developers, enabling innovative solutions and ultimately leading to the development of more efficient and intuitive visualization tools for the field.
