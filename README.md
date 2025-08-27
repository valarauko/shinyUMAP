# shinyUMAP: Interactive UMAP Parameter Exploration for Single-Cell Data

An interactive web application for understanding how UMAP parameters affect the visualization of single-cell omics data, helping researchers avoid common misinterpretations of cell cluster relationships.

## Overview

shinyUMAP addresses a critical issue in single-cell data analysis: the spatial distances between cell clusters in UMAP visualizations are highly dependent on parameter choices and may not reflect true biological relationships. This tool enables researchers to interactively adjust UMAP parameters with their own data to witness how cluster distributions change, promoting more appropriate interpretation of single-cell embeddings.

### Why shinyUMAP?

While UMAP has become ubiquitous in single-cell publications, experts have raised concerns about frequent misinterpretation of inter-cluster distances as biologically meaningful. shinyUMAP helps researchers:

- Understand that there is no single "correct" UMAP of their data
- Visualize how parameter changes affect cluster positioning
- Avoid inferring functional relationships based solely on spatial proximity in UMAP
- Compare cluster relationships using expression-based dendrograms

## Access

**Live Application**: https://scviewer.shinyapps.io/shinyUMAP/

**Local Installation**:

git clone https://github.com/[username]/shinyUMAP.git
cd shinyUMAP
pip install -r requirements.txt
shiny run app.py


Features
Core Functionality

Interactive Parameter Adjustment: Real-time modification of UMAP parameters including:

Number of neighbors (n_neighbors)
Minimum distance (min_dist)
Spread
Additional advanced parameters


Expression-Based Dendrograms: Hierarchical clustering showing actual gene expression similarity between cell types/clusters
Standard Processing Pipeline: Built-in Scanpy workflow for unprocessed data
Smart Downsampling: Optional random sampling for large datasets (>50,000 cells)

Input/Output

Input Format: H5AD (AnnData) files
Recommended Input: Data with only highly variable features to reduce memory usage
Outputs:

UMAP plots with customizable coloring (metadata or gene expression)
UMAP coordinates (CSV)
Updated AnnData object with new embeddings
Parameter settings for reproducibility
Expression similarity dendrograms



Usage Guide
Data Preparation

Format: H5AD files from Scanpy workflows
Preprocessing (if not already done):

Normalization and log-transformation
Highly variable gene selection
PCA computation


For Seurat Users: Use provided R conversion script or anndataR package

Interpretation Guidelines
⚠️ Critical Points:

Spatial distances between clusters in UMAP are NOT reliable indicators of biological similarity
Clusters that appear close may be functionally unrelated
Clusters that appear distant may share expression profiles
Use the dendrogram feature to assess actual expression similarity

# shinyUMAP
