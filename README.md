# Protein Structure Clustering Tool

A Python-based tool for clustering protein structures based on structural similarity.

## Overview

The Protein Structure Clustering Tool allows researchers to analyze relationships between protein structures by:
1. Searching the Protein Data Bank (PDB) for proteins by keyword
2. Downloading protein structure files
3. Extracting structural features
4. Clustering proteins based on similarity
5. Visualizing the clustering results through multiple methods

## Features

- **PDB Search**: Find proteins using text-based search queries
- **Custom Input**: Provide your own list of PDB IDs via a file
- **Multiple Clustering Methods**: 
  - Hierarchical clustering
  - K-means clustering
  - DBSCAN (Density-Based Spatial Clustering)
- **Rich Visualizations**:
  - Dendrogram showing hierarchical relationships
  - PCA plots for dimensionality reduction and visualization
  - t-SNE plots for non-linear dimensionality reduction
- **Detailed Output**: JSON result files and summary statistics

## Installation

```bash
# Clone the repository
git@github.com:parthkhurana07/clusterProteins.git
cd clusterProteins

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib biopython scipy scikit-learn rcsbsearchapi
```

## Usage

### Basic Usage

```bash
python protein_clustering.py --search "hemoglobin" --max-results 20
```

### Provide Your Own PDB IDs

```bash
python protein_clustering.py --file my_pdb_ids.txt
```

### Change Clustering Method

```bash
python protein_clustering.py --search "kinase" --cluster-method kmeans --num-clusters 5
```

### Full Option List

```
usage: protein_clustering.py [-h] (--search SEARCH | --file FILE) [--max-results MAX_RESULTS] [--output-dir OUTPUT_DIR]
                             [--cluster-method {hierarchical,kmeans,dbscan}] [--num-clusters NUM_CLUSTERS]

Protein Structure Clustering Tool

optional arguments:
  -h, --help            show this help message and exit
  --search SEARCH, -s SEARCH
                        Search term to find proteins in PDB
  --file FILE, -f FILE  File containing PDB IDs (one per line)
  --max-results MAX_RESULTS, -m MAX_RESULTS
                        Maximum number of search results (default: 20)
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for results (default: protein_clusters)
  --cluster-method {hierarchical,kmeans,dbscan}, -c {hierarchical,kmeans,dbscan}
                        Clustering method to use (default: hierarchical)
  --num-clusters NUM_CLUSTERS, -n NUM_CLUSTERS
                        Number of clusters for hierarchical and kmeans methods (default: 3)
```

## Output

The tool creates several outputs:

1. **PDB Files**: Downloaded protein structure files
2. **Clustering Results**: JSON file with cluster assignments
3. **Visualizations**:
   - Dendrogram showing hierarchical relationships between proteins
   - PCA plot of proteins colored by cluster
   - t-SNE plot of proteins colored by cluster
4. **Terminal Output**: Summary of clustering results

## Examples

### Searching for Hemoglobin Proteins

```bash
python protein_clustering.py --search "hemoglobin" --max-results 25
```

This command:
1. Searches PDB for hemoglobin proteins
2. Downloads up to 25 hemoglobin protein structures
3. Extracts features from each protein
4. Clusters the proteins using hierarchical clustering
5. Generates visualizations including a dendrogram
6. Prints a summary of the clustering results

### Custom List of Proteins with K-means Clustering

```bash
python protein_clustering.py --file my_proteins.txt --cluster-method kmeans --num-clusters 4
```

## Technical Details

### Feature Extraction

The tool extracts the following features from each protein:
- Number of residues
- Centroid position (average of alpha carbon coordinates)

### Distance Calculation

Euclidean distance is used to measure the dissimilarity between proteins in feature space.

### Visualization Methods

- **Dendrogram**: Shows hierarchical relationships between proteins
- **PCA**: Principal Component Analysis for dimensionality reduction
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for non-linear dimensionality reduction

## Extending the Tool

This tool can be extended in several ways:

1. Add more sophisticated feature extraction methods:
   - Secondary structure content
   - RMSD-based distance measurements
   - Domain-specific structural features

2. Implement additional visualization techniques:
   - 3D plots
   - Interactive visualizations
   - Structure superposition

3. Add functionality for comparing clustering results:
   - Statistical validation of clusters
   - Comparison with sequence-based clustering

## Dependencies

- NumPy: Numerical operations
- Matplotlib: Plotting and visualization
- BioPython: Parsing PDB files and interfacing with PDB
- SciPy: Distance calculations and hierarchical clustering
- Scikit-learn: Machine learning algorithms (clustering, dimensionality reduction)
- rcsbsearchapi: Searching the PDB database

## License

MIT License
