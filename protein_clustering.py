#!/usr/bin/env python3
"""
Protein Structure Clustering Tool

This script allows users to cluster protein structures based on structural similarity.
Users can either search for proteins by keyword or provide a file with PDB IDs.
"""

import os
import sys
import argparse
import requests
import numpy as np
import matplotlib.pyplot as plt
from Bio import PDB
from rcsbsearchapi import TextQuery
from Bio.PDB import PDBParser, PDBList
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import time
import shutil


class ProteinStructureClustering:
    def __init__(self, output_dir="protein_clusters"):
        """Initialize the protein clustering tool"""
        self.pdb_parser = PDBParser(QUIET=True)
        self.pdb_list = PDBList()
        
        # Create necessary directories
        self.base_dir = output_dir
        self.pdb_dir = os.path.join(self.base_dir, "pdb_files")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.figures_dir = os.path.join(self.base_dir, "figures")
        
        self._setup_directories()
        
        # Storage for protein data
        self.proteins = {}
        self.feature_matrix = None
        self.distance_matrix = None
        self.pdb_ids = []
        
    def _setup_directories(self):
        """Create necessary directories for the project"""
        directories = [self.base_dir, self.pdb_dir, self.results_dir, self.figures_dir]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def search_pdb(self, query, max_results=20):
        """
        Search the PDB database for proteins matching the query
        
        Args:
            query (str): Search query string
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of PDB IDs matching the query
        """
        print(f"Searching PDB for: {query}")

        search = TextQuery(value=query)
        results = search()
        r_1 = []
        for i, rid in enumerate(results):
            if i >= max_results:
                break
            r_1.append(rid)

        return(r_1)
            
    def load_pdb_ids_from_file(self, filename):
        """
        Load PDB IDs from a file
        
        Args:
            filename (str): Path to the file containing PDB IDs (one per line)
            
        Returns:
            list: List of PDB IDs from the file
        """
        try:
            with open(filename, 'r') as f:
                pdb_ids = [line.strip() for line in f if line.strip()]
            
            print(f"Loaded {len(pdb_ids)} PDB IDs from {filename}")
            return pdb_ids
        except Exception as e:
            print(f"Error loading PDB IDs from file: {e}")
            return []
            
    def download_proteins(self, pdb_ids):
        """
        Download protein structures for the given PDB IDs
        
        Args:
            pdb_ids (list): List of PDB IDs to download
        """
        self.pdb_ids = pdb_ids
        print(f"Downloading {len(pdb_ids)} protein structures...")
        
        for i, pdb_id in enumerate(pdb_ids):
            print(f"Downloading {pdb_id} ({i+1}/{len(pdb_ids)})...")
            
            try:
                # Download the PDB file
                pdb_file = self.pdb_list.retrieve_pdb_file(
                    pdb_id, 
                    pdir=self.pdb_dir, 
                    file_format="pdb"
                )
                
                # Check if file was downloaded successfully
                if os.path.exists(pdb_file):
                    print(f"Downloaded {pdb_id} successfully")
                else:
                    print(f"Failed to download {pdb_id}")
                    
            except Exception as e:
                print(f"Error downloading {pdb_id}: {e}")
                
            # Avoid overwhelming the server
            time.sleep(0.5)
            
    def extract_features(self):
        """
        Extract features from downloaded protein structures
        """
        print("Extracting structural features from proteins...")
        
        for pdb_id in self.pdb_ids:
            # Determine the path to the PDB file
            pdb_filename = os.path.join(self.pdb_dir, f"pdb{pdb_id.lower()}.ent")
            
            # Skip if file doesn't exist
            if not os.path.exists(pdb_filename):
                print(f"Skipping {pdb_id} - file not found")
                continue
                
            try:
                # Load the structure
                structure = self.pdb_parser.get_structure(pdb_id, pdb_filename)
                
                # Extract CA atoms for the first model
                model = structure[0]
                ca_atoms = [atom for atom in model.get_atoms() if atom.get_name() == 'CA']
                
                if len(ca_atoms) < 3:
                    print(f"Skipping {pdb_id} - too few CA atoms found")
                    continue
                    
                # Calculate the pairwise distances between CA atoms
                coords = np.array([atom.get_coord() for atom in ca_atoms])
                
                # Store the features
                self.proteins[pdb_id] = {
                    'coords': coords,
                    'num_residues': len(ca_atoms),
                    'centroid': np.mean(coords, axis=0)
                }
                
                print(f"Processed {pdb_id}: {len(ca_atoms)} residues")
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                
        print(f"Successfully extracted features from {len(self.proteins)} proteins")
        
        # Create feature vector for each protein
        feature_list = []
        valid_pdb_ids = []
        
        for pdb_id in self.pdb_ids:
            if pdb_id in self.proteins:
                # For now, use simple features: number of residues and centroid position
                protein = self.proteins[pdb_id]
                features = np.concatenate([[protein['num_residues']], protein['centroid']])
                feature_list.append(features)
                valid_pdb_ids.append(pdb_id)
                
        self.pdb_ids = valid_pdb_ids
        self.feature_matrix = np.array(feature_list)
        
        # Calculate distance matrix
        if len(feature_list) > 1:
            self.distance_matrix = squareform(pdist(self.feature_matrix, 'euclidean'))
            
    def cluster_proteins(self, method="hierarchical", n_clusters=3):
        """
        Cluster the proteins based on their features
        
        Args:
            method (str): Clustering method: 'hierarchical', 'kmeans', or 'dbscan'
            n_clusters (int): Number of clusters for hierarchical and kmeans
        
        Returns:
            dict: Clustering results
        """
        if len(self.proteins) < 2:
            print("Need at least 2 proteins for clustering")
            return None
            
        print(f"Clustering {len(self.proteins)} proteins using {method} method...")
        
        # Check if feature matrix exists
        if self.feature_matrix is None or len(self.feature_matrix) == 0:
            print("No feature matrix available. Run extract_features first.")
            return None
            
        # Normalize features
        normalized_features = (self.feature_matrix - np.mean(self.feature_matrix, axis=0)) / np.std(self.feature_matrix, axis=0)
        
        # Apply the selected clustering method
        if method == "hierarchical":
            # Try different parameter combinations based on scikit-learn version
            try:
                # Newer versions of scikit-learn
                clustering = AgglomerativeClustering(
                    n_clusters=min(n_clusters, len(self.proteins)),
                    affinity='euclidean',
                    linkage='ward'
                )
            except TypeError:
                # Older versions of scikit-learn
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=min(n_clusters, len(self.proteins)),
                        linkage='ward'
                    )
                except TypeError:
                    # Very old versions or other configurations
                    clustering = AgglomerativeClustering(
                        n_clusters=min(n_clusters, len(self.proteins))
                    )
            
            labels = clustering.fit_predict(normalized_features)
            
        elif method == "kmeans":
            clustering = KMeans(
                n_clusters=min(n_clusters, len(self.proteins)),
                random_state=42
            )
            labels = clustering.fit_predict(normalized_features)
            
        elif method == "dbscan":
            clustering = DBSCAN(
                eps=0.5,
                min_samples=2
            )
            labels = clustering.fit_predict(normalized_features)
            
        else:
            print(f"Unknown clustering method: {method}")
            return None
            
        # Create a dictionary mapping PDB IDs to cluster labels
        cluster_results = {
            'method': method,
            'n_clusters': len(set(labels)),
            'clusters': {}
        }
        
        # Group proteins by cluster
        for i, pdb_id in enumerate(self.pdb_ids):
            cluster_id = int(labels[i])
            if cluster_id not in cluster_results['clusters']:
                cluster_results['clusters'][cluster_id] = []
            cluster_results['clusters'][cluster_id].append(pdb_id)
            
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"clusters_{method}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(cluster_results, f, indent=2)
            
        print(f"Clustering results saved to {results_file}")
        
        # Visualize clusters
        self.visualize_clusters(normalized_features, labels, method, timestamp)
        
        return cluster_results
        
    def visualize_clusters(self, features, labels, method, timestamp):
        """
        Create visualizations for the clustering results
        
        Args:
            features (np.array): Feature matrix
            labels (np.array): Cluster labels
            method (str): Clustering method used
            timestamp (str): Timestamp for filenames
        """
        print("Creating cluster visualizations...")
        
        # Create a 2D representation using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        # Create a 2D representation using t-SNE if we have enough samples
        if len(features) >= 5:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            tsne_result = tsne.fit_transform(features)
        else:
            tsne_result = None
            
        # Plot PCA results
        plt.figure(figsize=(12, 10))
        
        # Get unique cluster labels
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=[colors[i]],
                label=f'Cluster {label}',
                alpha=0.7,
                s=100
            )
            
        # Add labels for each point
        for i, pdb_id in enumerate(self.pdb_ids):
            plt.annotate(
                pdb_id,
                (pca_result[i, 0], pca_result[i, 1]),
                fontsize=8,
                alpha=0.8
            )
            
        plt.title(f'PCA Visualization of Protein Clusters ({method})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the PCA plot
        pca_filename = os.path.join(self.figures_dir, f"pca_clusters_{method}_{timestamp}.png")
        plt.savefig(pca_filename, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to {pca_filename}")
        
        # Plot t-SNE results if available
        if tsne_result is not None:
            plt.figure(figsize=(12, 10))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(
                    tsne_result[mask, 0],
                    tsne_result[mask, 1],
                    c=[colors[i]],
                    label=f'Cluster {label}',
                    alpha=0.7,
                    s=100
                )
                
            # Add labels for each point
            for i, pdb_id in enumerate(self.pdb_ids):
                plt.annotate(
                    pdb_id,
                    (tsne_result[i, 0], tsne_result[i, 1]),
                    fontsize=8,
                    alpha=0.8
                )
                
            plt.title(f't-SNE Visualization of Protein Clusters ({method})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save the t-SNE plot
            tsne_filename = os.path.join(self.figures_dir, f"tsne_clusters_{method}_{timestamp}.png")
            plt.savefig(tsne_filename, dpi=300, bbox_inches='tight')
            print(f"t-SNE visualization saved to {tsne_filename}")
            
    def print_cluster_summary(self, cluster_results):
        """Print a summary of the clustering results"""
        if not cluster_results:
            print("No clustering results available")
            return
            
        print("\n" + "="*50)
        print(f"CLUSTERING SUMMARY ({cluster_results['method'].upper()})")
        print("="*50)
        print(f"Total proteins analyzed: {len(self.pdb_ids)}")
        print(f"Number of clusters found: {cluster_results['n_clusters']}")
        print("\nClusters:")
        
        for cluster_id, pdb_ids in cluster_results['clusters'].items():
            if cluster_id == -1:
                print(f"\nNOISE/OUTLIERS: {len(pdb_ids)} proteins")
            else:
                print(f"\nCluster {cluster_id}: {len(pdb_ids)} proteins")
                
            # Print the PDB IDs in this cluster
            for pdb_id in pdb_ids:
                if pdb_id in self.proteins:
                    num_residues = self.proteins[pdb_id]['num_residues']
                    print(f"  - {pdb_id}: {num_residues} residues")
        
        print("="*50)
        
    def cleanup(self):
        """Optional cleanup of downloaded files"""
        response = input("Do you want to clean up downloaded PDB files? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(self.pdb_dir)
            print(f"Removed directory: {self.pdb_dir}")
            os.makedirs(self.pdb_dir)
            print(f"Created empty directory: {self.pdb_dir}")


def main():
    parser = argparse.ArgumentParser(description="Protein Structure Clustering Tool")
    
    # Create a mutually exclusive group for the input method
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--search", "-s",
        help="Search term to find proteins in PDB"
    )
    input_group.add_argument(
        "--file", "-f",
        help="File containing PDB IDs (one per line)"
    )
    
    # Add optional arguments
    parser.add_argument(
        "--max-results", "-m",
        type=int,
        default=20,
        help="Maximum number of search results (default: 20)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="protein_clusters",
        help="Output directory for results (default: protein_clusters)"
    )
    parser.add_argument(
        "--cluster-method", "-c",
        choices=["hierarchical", "kmeans", "dbscan"],
        default="hierarchical",
        help="Clustering method to use (default: hierarchical)"
    )
    parser.add_argument(
        "--num-clusters", "-n",
        type=int,
        default=3,
        help="Number of clusters for hierarchical and kmeans methods (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Initialize the clustering tool
    clustering_tool = ProteinStructureClustering(output_dir=args.output_dir)
    
    # Get PDB IDs either from search or file
    pdb_ids = []
    if args.search:
        pdb_ids = clustering_tool.search_pdb(args.search, max_results=args.max_results)
    elif args.file:
        pdb_ids = clustering_tool.load_pdb_ids_from_file(args.file)
        
    if not pdb_ids:
        print("No PDB IDs found. Exiting.")
        sys.exit(1)
        
    # Download protein structures
    clustering_tool.download_proteins(pdb_ids)
    
    # Extract features from the proteins
    clustering_tool.extract_features()
    
    # Perform clustering
    cluster_results = clustering_tool.cluster_proteins(
        method=args.cluster_method,
        n_clusters=args.num_clusters
    )
    
    # Print cluster summary
    clustering_tool.print_cluster_summary(cluster_results)
    
    # Optional cleanup
    clustering_tool.cleanup()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()