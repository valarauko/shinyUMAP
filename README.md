# shinyUMAP

**Interactive UMAP parameter exploration for single-cell data.**  
A Shiny web app that helps researchers understand how UMAP parameters affect embeddings and avoid misinterpreting inter-cluster distances.

## Features
- **Interactive parameter tuning**: adjust *n_neighbors*, *min_dist*, *spread*, etc.  
- **Expression dendrograms**: compare clusters based on expression similarity.  
- **Built-in Scanpy workflow**: process unnormalized data.  
- **Smart downsampling**: handle datasets >50k cells efficiently.  
- **Flexible I/O**: load `.h5ad` (AnnData) and export coordinates, embeddings, and parameter logs.

## Access
- **Live app**: [shinyUMAP on shinyapps.io](https://scviewer.shinyapps.io/shinyUMAP/)  
- **Local install**:
```bash
git clone 
cd shinyUMAP
pip install -r requirements.txt
shiny run app.py
```

## Usage
**Input**: preprocessed `.h5ad` files (normalized, HVG selection, PCA).  
For Seurat users, convert via `anndataR` or the provided R script.  

**Outputs**:  
- UMAP plots (metadata or gene expression coloring)  
- UMAP coordinates (`.csv`)  
- Updated AnnData with new embeddings  
- Expression similarity dendrograms  

## Interpretation
Reminder:  
- UMAP distances do **not** imply biological similarity.  
- Clusters may appear close but be functionally unrelated.  
- Always verify with the dendrogram feature or other expression-based analyses.

## Citation
If you use shinyUMAP, please cite:  
*(placeholder)*
