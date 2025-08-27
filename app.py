"""
shinyUMAP - UMAP Parameter Optimization Platform for Single-Cell Visualization

This Shiny for Python application provides an interactive platform for optimizing
UMAP parameters on single-cell data. It offers comprehensive visualization
and analysis tools for exploring cellular data through dimensionality reduction.

Key Features:
- Interactive UMAP parameter tuning with real-time visualization
- Support for AnnData file format (H5AD)
- Automated data preprocessing and quality control
- Gene expression and metadata-based coloring
- Hierarchical clustering with interactive dendrograms  
- Data export capabilities (AnnData, UMAP coordinates, parameters)

Author: Rohan Misra
License: MIT
Version: 1.0
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, module, req
import tempfile
import warnings
import os
from pathlib import Path
import json
import logging
from contextlib import contextmanager
import gc
import traceback
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import seaborn as sns
import shutil
import time
import uuid
import asyncio


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("shinyUMAP")

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# Configuration - shinyapps.io compatible
class Config:
    MAX_CELLS_WITHOUT_PROMPT = 10000
    DEFAULT_DOWNSAMPLE_SIZE = 10000
    FILE_SIZE_THRESHOLD_MB = 100

# Improved Temp File Manager
class TempFileManager:
    """Manage temporary files with automatic cleanup"""
    def __init__(self):
        self.session_dir = None
        self.files = set()
        
    def init_session(self, session_id):
        """Initialize session-specific temp directory"""
        self.session_dir = os.path.join(".", "temp_sessions", session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        return self.session_dir
        
    def add_file(self, filename):
        """Track a temp file"""
        if self.session_dir:
            filepath = os.path.join(self.session_dir, filename)
            self.files.add(filepath)
            return filepath
        return filename
        
    def get_temp_path(self, suffix=""):
        """Get a new temp file path"""
        unique_id = str(uuid.uuid4())[:8]
        filename = f"temp_{unique_id}{suffix}"
        return self.add_file(filename)
        
    def cleanup(self):
        """Clean up all temp files for this session"""
        if self.session_dir and os.path.exists(self.session_dir):
            try:
                shutil.rmtree(self.session_dir)
                logger.info(f"Cleaned up session directory: {self.session_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up session directory: {e}")
        self.files.clear()
    
    def cleanup_old_sessions(self, max_age_hours=24):
        """Clean up old session directories"""
        temp_root = os.path.join(".", "temp_sessions")
        if not os.path.exists(temp_root):
            return
            
        current_time = time.time()
        for session_dir in os.listdir(temp_root):
            dir_path = os.path.join(temp_root, session_dir)
            if os.path.isdir(dir_path):
                # Check directory age
                try:
                    dir_age = current_time - os.path.getmtime(dir_path)
                    if dir_age > max_age_hours * 3600:
                        shutil.rmtree(dir_path)
                        logger.info(f"Cleaned up old session: {session_dir}")
                except:
                    pass

# Helper function to create info buttons
def info_button(param_id, title, content):
    """Create an info button with popover"""
    return ui.span(
        ui.HTML(f'''
        <button type="button" class="btn btn-link btn-sm p-0 ml-1" 
                data-bs-toggle="popover" 
                data-bs-trigger="hover focus" 
                data-bs-placement="right"
                data-bs-html="true"
                title="{title}"
                data-bs-content="{content}">
            <i class="fas fa-info-circle text-info"></i>
        </button>
        '''),
        id=f"info_{param_id}"
    )

def status_badge(status, text):
    """Create status badges with icons"""
    icons = {
        'success': 'fas fa-check-circle text-success',
        'warning': 'fas fa-exclamation-triangle text-warning', 
        'error': 'fas fa-times-circle text-danger',
        'info': 'fas fa-info-circle text-info',
        'disabled': 'fas fa-lock text-muted'
    }
    
    return ui.span(
        ui.tags.i(class_=icons.get(status, 'fas fa-circle')),
        f" {text}",
        class_=f"badge badge-{status} ml-2"
    )

def get_file_info(file_path):
    """Get comprehensive information about the h5ad file"""
    logger.info(f"Getting file info for: {file_path}")
    logger.info(f"File exists: {os.path.exists(file_path)}")
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return {
                'format': 'h5ad',
                'size_mb': 0,
                'n_cells': None,
                'n_genes': None,
                'is_valid': False,
                'error': f'File not found: {file_path}',
                'has_raw': False,
                'has_reductions': False,
                'reductions': [],
                'processing_status': 'error'
            }
            
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        info = {
            'format': 'h5ad',
            'size_mb': file_size_mb,
            'n_cells': None,
            'n_genes': None,
            'is_valid': True,
            'error': None,
            'has_raw': False,
            'has_reductions': False,
            'reductions': [],
            'processing_status': 'unknown'
        }

        # Try to read detailed info
        try:
            import h5py
            logger.info("Attempting to read h5ad file structure...")
            with h5py.File(file_path, 'r') as f:
                if 'obs' in f and '_index' in f['obs']:
                    info['n_cells'] = f['obs']['_index'].shape[0]
                if 'var' in f and '_index' in f['var']:
                    info['n_genes'] = f['var']['_index'].shape[0]
                
                # Check for raw data
                if 'raw' in f:
                    info['has_raw'] = True
                
                # Check for dimensionality reductions
                if 'obsm' in f:
                    reductions = [key for key in f['obsm'].keys() if key.startswith('X_')]
                    info['reductions'] = reductions
                    info['has_reductions'] = len(reductions) > 0
            
            logger.info(f"File info extracted successfully: {info['n_cells']} cells, {info['n_genes']} genes")

        except Exception as e:
            logger.warning(f"Could not read detailed file info with h5py: {e}")

        return info

    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {
            'format': 'h5ad',
            'size_mb': 0,
            'n_cells': None,
            'n_genes': None,
            'is_valid': False,
            'error': str(e),
            'has_raw': False,
            'has_reductions': False,
            'reductions': [],
            'processing_status': 'error'
        }

def analyze_dataset_quality(adata):
    """Analyze dataset to provide technical state information"""
    try:
        # Sample data for efficiency
        n_sample = min(100, adata.n_obs)
        if hasattr(adata.X, 'toarray'):
            X_sample = adata.X[:n_sample, :100].toarray()
        else:
            X_sample = adata.X[:n_sample, :100]

        max_val = np.max(X_sample)
        min_val = np.min(X_sample)
        mean_val = np.mean(X_sample)
        
        # Check for existing dimensionality reductions
        has_reductions = any(key.startswith('X_') and key != 'X_umap' for key in adata.obsm.keys())
        has_hvg = 'highly_variable' in adata.var.columns
        has_raw = adata.raw is not None
        
        analysis = {
            'has_reductions': has_reductions,
            'has_hvg': has_hvg,
            'has_raw': has_raw,
            'should_remove_raw': False,
            'data_state': 'unknown',
            'recommendations': []
        }
        
        # Determine technical data state
        if has_reductions and min_val < 0 and abs(mean_val) < 1:
            analysis['data_state'] = 'normalized & scaled'
        elif has_reductions:
            analysis['data_state'] = 'normalized with reductions'
        elif max_val > 100 and np.sum(X_sample % 1 == 0) > X_sample.size * 0.7:
            analysis['data_state'] = 'raw counts'
            analysis['recommendations'].append("Raw count data detected - preprocessing recommended")
        elif min_val >= 0 and max_val < 50:
            analysis['data_state'] = 'normalized'
        elif min_val < 0:
            analysis['data_state'] = 'scaled'
        else:
            analysis['data_state'] = 'processed'
        
        # Aggressive auto-optimization: remove raw if ANY processing detected
        if has_raw and (has_hvg or has_reductions or analysis['data_state'] in ['normalized', 'scaled', 'normalized & scaled']):
            analysis['should_remove_raw'] = True
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return {
            'has_reductions': False,
            'has_hvg': False,
            'has_raw': False,
            'should_remove_raw': False,
            'data_state': 'error',
            'recommendations': [f"Error analyzing dataset: {str(e)}"]
        }

def detect_available_reductions(adata):
    """Detect available dimensionality reductions suitable for UMAP input"""
    available = {}
    if adata is None:
        return available

    # Reductions suitable as input for UMAP
    reduction_names = ['X_pca', 'X_pls', 'X_lsi', 'X_ica', 'X_nmf']

    for name in reduction_names:
        if name in adata.obsm:
            n_components = adata.obsm[name].shape[1]
            display_name = name.replace('X_', '').upper()
            available[name] = f"{display_name} ({n_components} components)"

    return available

def is_numeric_column(adata, column_name):
    """
    Simple check if a metadata column is truly continuous (for dendrogram purposes)
    """
    if adata is None or column_name not in adata.obs.columns:
        return False

    try:
        col_data = adata.obs[column_name]
        
        # Not numeric at all
        if not pd.api.types.is_numeric_dtype(col_data):
            return False
        
        # Already categorical
        if pd.api.types.is_categorical_dtype(col_data):
            return False
        
        # Remove NaN values
        clean_data = col_data.dropna()
        if len(clean_data) == 0:
            return False
        
        n_unique = clean_data.nunique()
        
        # Simple rule: if more than 50 unique values, probably continuous
        # Otherwise, probably categorical (clusters, conditions, etc.)
        return n_unique > 50
        
    except Exception as e:
        logger.error(f"Error checking column {column_name}: {e}")
        return False


def truncate_filename(filename, max_length=25):
    """Truncate filename intelligently, keeping start and end"""
    if not filename or len(filename) <= max_length:
        return filename
    
    # Split name and extension
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
        ext = '.' + ext
    else:
        name, ext = filename, ''
    
    # Calculate space for name part
    available = max_length - len(ext) - 3  # 3 for "..."
    
    if available <= 5:  # Too short, just truncate
        return filename[:max_length-3] + "..."
    
    # Keep start and end of name
    start_len = available // 2
    end_len = available - start_len
    
    if len(name) <= available:
        return filename
    
    truncated_name = name[:start_len] + "..." + name[-end_len:] if end_len > 0 else name[:start_len] + "..."
    return truncated_name + ext

def auto_optimize_memory(adata):
    """Aggressively optimize memory by removing redundant data"""
    if adata is None:
        return adata
    
    try:
        optimizations = []
        
        # Get dataset analysis
        analysis = analyze_dataset_quality(adata)
        
        # Remove raw if any processing is detected
        if analysis['should_remove_raw'] and adata.raw is not None:
            adata.raw = None
            optimizations.append("raw count matrix")
        
        # Remove large unused layers if present
        if hasattr(adata, 'layers') and len(adata.layers) > 0:
            # Keep only essential layers
            essential_layers = ['counts', 'logcounts', 'data']
            layers_to_remove = [layer for layer in adata.layers.keys() if layer not in essential_layers]
            for layer in layers_to_remove:
                del adata.layers[layer]
                optimizations.append(f"layer '{layer}'")
        
        if optimizations:
            logger.info(f"Memory optimized: removed {', '.join(optimizations)}")
            
    except Exception as e:
        logger.error(f"Error in auto memory optimization: {e}")
    
    return adata

def compute_category_dendrogram_cached(adata, category_column, umap_params=None):
    """
    Compute hierarchical clustering dendrogram using the EXACT same embedding as UMAP
    """
    try:
        if category_column not in adata.obs.columns:
            logger.warning(f"Category column '{category_column}' not found in metadata")
            return None
        
        # STRICT LIMITS for categorical data
        if is_numeric_column(adata, category_column):
            logger.info(f"Skipping dendrogram for continuous variable: {category_column}")
            return None
            
        categories = adata.obs[category_column].astype('category')
        category_names = categories.cat.categories.tolist()
        
        # Enhanced limits with clear thresholds
        n_categories = len(category_names)
        if n_categories < 2:
            logger.info(f"Too few categories for dendrogram: {n_categories} < 2")
            return None
        if n_categories > 25:  # Match UMAP plotting limit
            logger.info(f"Too many categories for dendrogram: {n_categories} > 25 (limit)")
            return None
        
        # FIXED: Check minimum cells per category - only skip if too few valid categories remain
        min_cells_per_category = 5  # Require at least 5 cells per category
        category_counts = categories.value_counts()
        total_valid_categories = len([count for count in category_counts if count >= min_cells_per_category])
        if total_valid_categories < 2:
            logger.info(f"Skipping dendrogram: Only {total_valid_categories} categories have ≥ {min_cells_per_category} cells")
            return None
        
        # ENHANCED: Determine which embedding to use - MUST match UMAP exactly
        use_rep = None
        n_components = None
        representation_used = "unknown"
        
        logger.info(f"UMAP params for dendrogram: {umap_params}")
        
        # Priority 1: Use exact same embedding as UMAP if available
        if umap_params and 'reduction_used' in umap_params:
            reduction_name = umap_params['reduction_used']
            use_rep_key = f'X_{reduction_name.lower()}'
            
            logger.info(f"Trying to match UMAP embedding: {reduction_name} -> {use_rep_key}")
            
            # Check if this embedding exists
            if use_rep_key in adata.obsm:
                use_rep = use_rep_key
                representation_used = reduction_name
                
                # Use exact same number of components as UMAP
                if 'n_embedding_components' in umap_params:
                    available_components = adata.obsm[use_rep].shape[1]
                    n_components = min(umap_params['n_embedding_components'], available_components)
                else:
                    n_components = min(50, adata.obsm[use_rep].shape[1])
                    
                logger.info(f"✅ Using UMAP embedding: {use_rep} with {n_components} components (matches UMAP)")
            else:
                logger.warning(f"UMAP embedding {use_rep_key} not found in obsm keys: {list(adata.obsm.keys())}")
        
        # Priority 2: If UMAP params missing or embedding not found, match what UMAP would use
        if use_rep is None:
            logger.info("No UMAP embedding match - trying to infer what UMAP used...")
            
            # Check if there's a subset embedding created by UMAP computation
            subset_embeddings = [key for key in adata.obsm.keys() if key.endswith('_subset')]
            if subset_embeddings:
                # Use the subset embedding (this is what UMAP actually used)
                use_rep = subset_embeddings[0]  # Should be something like 'X_pca_subset'
                n_components = adata.obsm[use_rep].shape[1]
                representation_used = use_rep.replace('X_', '').replace('_subset', '').upper()
                logger.info(f"✅ Using UMAP subset embedding: {use_rep} with {n_components} components")
            else:
                # Fallback: use standard embeddings in priority order
                for reduction in ['X_pca', 'X_lsi', 'X_pls', 'X_ica', 'X_nmf']:
                    if reduction in adata.obsm:
                        use_rep = reduction
                        n_components = min(50, adata.obsm[reduction].shape[1])
                        representation_used = reduction.replace('X_', '').upper()
                        logger.info(f"⚠️ Fallback embedding: {use_rep} with {n_components} components")
                        break
        
        if use_rep is None:
            logger.error("No suitable embedding found for dendrogram computation")
            return None
        
        # Enhanced logging for debugging
        logger.info(f"Final dendrogram embedding: {use_rep} ({representation_used}) with {n_components} components")
        if umap_params:
            logger.info(f"UMAP used: {umap_params.get('reduction_used', 'unknown')} with {umap_params.get('n_embedding_components', 'unknown')} components")
        
        # Compute category means using the EXACT same embedding as UMAP
        category_matrix = []
        valid_categories = []
        
        logger.info(f"Computing dendrogram using {use_rep} with {n_components} components")
        
        for cat in category_names:
            mask = (categories == cat).values
            n_cells_in_cat = np.sum(mask)
            
            if n_cells_in_cat < min_cells_per_category:
                logger.debug(f"Skipping category '{cat}': {n_cells_in_cat} cells < {min_cells_per_category}")
                continue
            
            # Compute mean of embedding for this category (exact same approach as UMAP)
            cat_embedding = adata.obsm[use_rep][mask, :n_components]
            cat_mean = np.mean(cat_embedding, axis=0)
            
            # Validate the computed mean
            has_nan = np.isnan(cat_mean).any()
            has_inf = np.isinf(cat_mean).any()
            
            if has_nan or has_inf:
                logger.error(f"Category '{cat}' has invalid values in embedding mean: NaN={has_nan}, Inf={has_inf}")
                continue
            
            category_matrix.append(cat_mean)
            valid_categories.append(cat)
            
            logger.debug(f"Category '{cat}': {n_cells_in_cat} cells, mean range [{np.min(cat_mean):.3f}, {np.max(cat_mean):.3f}]")
            
        if len(valid_categories) < 2:
            logger.info(f"Too few valid categories after filtering: {len(valid_categories)}")
            return None
            
        # Convert to numpy array and validate
        category_matrix = np.array(category_matrix)
        
        logger.info(f"Category matrix shape: {category_matrix.shape}")
        logger.info(f"Valid categories: {valid_categories}")
        
        # Final validation of category matrix
        if np.isnan(category_matrix).any() or np.isinf(category_matrix).any():
            logger.error("Category matrix contains NaN or Inf values")
            return None
        
        # Compute distance matrix using correlation distance (scanpy default)
        try:
            distances = pdist(category_matrix, metric='correlation')
            logger.info(f"Distance computation successful: {distances.shape[0]} pairwise distances")
        except Exception as e:
            logger.error(f"Distance computation failed: {e}")
            return None
        
        # Compute linkage using complete linkage (scanpy default)
        try:
            linkage_matrix = linkage(distances, method='complete')
            logger.info(f"Linkage computation successful: {linkage_matrix.shape}")
        except Exception as e:
            logger.error(f"Linkage computation failed: {e}")
            return None
        
        result = {
            'linkage': linkage_matrix,
            'labels': valid_categories,
            'category': category_column,
            'n_categories': len(valid_categories),
            'representation': representation_used,
            'n_components': n_components,
            'embedding_used': use_rep,  # Store exact embedding used
            'metric': 'correlation',
            'linkage_method': 'complete',
            'computed_at': time.time(),
            'matched_umap': umap_params is not None  # Flag whether it matched UMAP
        }
        
        logger.info(f"✅ Computed dendrogram for {category_column} using {representation_used} ({len(valid_categories)} categories)")
        return result
        
    except Exception as e:
        logger.error(f"Error computing dendrogram: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# Enhanced UI definition
app_ui = ui.page_fluid(
    # Add title and Font Awesome for icons
    ui.tags.head(
        ui.tags.title("shinyUMAP"),
        ui.tags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
        ),
        ui.tags.style("""
        /* Enhanced styling for academic interface */
        .main-content-row {
            display: flex;
            min-height: 100vh;
        }
        
        .parameters-column {
            flex: 0 0 33.333333%;
            max-width: 33.333333%;
            padding-right: 15px;
            padding-left: 15px;
            overflow-y: auto;
            max-height: 100vh;
        }
        
        .plot-column {
            flex: 0 0 66.666667%;
            max-width: 66.666667%;
            position: sticky;
            top: 0;
            height: 100vh;
            overflow: auto;
            padding-left: 15px;
            padding-right: 15px;
        }
        
        .plot-container {
            width: 100%;
            height: auto;
            min-width: 600px;
            max-width: 800px;
            min-height: 600px;
            padding: 20px;
            box-sizing: border-box;
        }
        
        /* Data loading panel styling */
        .data-loading-panel {
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        .data-loading-panel.loaded {
            border-color: #28a745;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }
        
        /* Status badges */
        .badge {
            display: inline-block;
            padding: 0.25em 0.6em;
            font-size: 0.75em;
            font-weight: 700;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.25rem;
        }
        
        .badge-success { background-color: #28a745; color: white; }
        .badge-warning { background-color: #ffc107; color: black; }
        .badge-error { background-color: #dc3545; color: white; }
        .badge-info { background-color: #17a2b8; color: white; }
        .badge-disabled { background-color: #6c757d; color: white; }
        
        /* Section headers with status indicators */
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        /* Getting started panel */
        .getting-started {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        /* Validation messages */
        .validation-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
            border-left: 4px solid;
        }
        
        .validation-success {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        
        .validation-warning {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        
        .validation-error {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        
        .validation-info {
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
        
        /* Progress bars */
        .progress-container {
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #dee2e6;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: linear-gradient(90deg, #007bff, #28a745, #ffc107, #dc3545);
            background-size: 400% 400%;
            border-radius: 4px;
            animation: gradient-shift 2s ease infinite;
        }
        
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .progress-text {
            text-align: center;
            margin-top: 5px;
            font-size: 0.9em;
            color: #495057;
        }
        
        /* Disabled section styling */
        .section-disabled {
            opacity: 0.6;
            pointer-events: none;
        }
        
        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }
        
        /* Dynamic plot dimensions */
        #umap_plot {
            width: auto !important;
            height: 600px !important;
            max-width: none !important;
            max-height: none !important;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main-content-row {
                flex-direction: column;
            }
            .parameters-column, .plot-column {
                flex: 1 1 auto;
                max-width: 100%;
                position: relative;
                height: auto;
                max-height: none;
                padding: 10px;
            }
            #umap_plot {
                width: 100% !important;
                height: 500px !important;
                max-width: 100% !important;
            }
            .plot-container {
                min-width: auto;
                max-width: 100%;
                min-height: 500px;
                padding: 10px;
            }
            .btn-block {
                margin-bottom: 10px;
            }
            .section-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 5px;
            }
        }
        
        /* Clickable links in validation messages */
        .validation-message a {
            color: #007bff;
            text-decoration: underline;
            cursor: pointer;
        }
        
        .validation-message a:hover {
            color: #0056b3;
            text-decoration: none;
        }
        """),
        ui.tags.script("""
        $(document).ready(function(){
            function initPopovers() {
                try {
                    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
                    popoverTriggerList.forEach(function(popoverTriggerEl) {
                        var existingPopover = bootstrap.Popover.getInstance(popoverTriggerEl);
                        if (existingPopover) {
                            existingPopover.dispose();
                        }
                        new bootstrap.Popover(popoverTriggerEl);
                    });
                } catch (e) {
                    console.error('Error initializing popovers:', e);
                }
            }

            // Initial setup
            initPopovers();

            // Re-initialize popovers on content updates
            $(document).on('shiny:value', function(event) {
                setTimeout(initPopovers, 100);
            });
        });
        """)
    ),

    ui.div(
        ui.h1("shinyUMAP", style="margin-bottom: 0.25rem;"),
        ui.p("UMAP Parameter Optimization Platform for Single-Cell Visualization", 
             style="font-size: 1.1rem; color: #495057; margin-bottom: 0.25rem;"),
        style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #dee2e6; margin-bottom: 1rem;"
    ),

    ui.navset_tab(
        ui.nav_panel("Analysis",
            ui.div(
                ui.div(
                    # Left column - Parameters (scrollable)
                    ui.div(
                        # Getting Started Guide
                        ui.output_ui("getting_started_ui"),
                        
                        # Data Loading Panel
                        ui.output_ui("data_loading_panel_ui"),
                        
                        # Progress indicator
                        ui.output_ui("progress_ui"),
                        
                        # Validation Messages
                        ui.output_ui("validation_messages_ui"),
                        
                        # Analysis sections
                        ui.output_ui("analysis_sections_ui"),
                        
                        class_="parameters-column"
                    ),
                    
                    # Right column - Plot (sticky)
                    ui.div(
                        ui.output_ui("status_message"),
                        ui.h4("UMAP Visualization"),
                        ui.output_plot("umap_plot", width="100%", height="auto"),
                        ui.output_ui("umap_parameters_display"),
                        ui.hr(),
                        
                        # Dendrogram section
                        ui.output_ui("dendrogram_ui"),
                        
                        class_="plot-column"
                    ),
                    
                    class_="main-content-row"
                )
            )
        ),

        ui.nav_panel("User Guide",
            ui.div(
                # Platform Overview
                ui.div(
                    ui.h3("UMAP Parameter Optimization Platform"),
                    ui.p("A tool for systematic exploration of UMAP embeddings in single-cell visualization. "
                         "UMAP (Uniform Manifold Approximation and Projection) creates low-dimensional representations "
                         "that preserve both local neighborhoods and global structure of high-dimensional data."),
                    
                    ui.h4("Biological Context", class_="mt-4"),
                    ui.p("UMAP embeddings help visualize relationships between cells based on their expression profiles. "
                         "The algorithm balances preservation of local structure (cells with similar expression) and "
                         "global structure (relationships between cell populations). Parameter choices affect whether "
                         "the embedding emphasizes fine-scale differences or broad population structure."),
                    
                    ui.div(
                        ui.p(ui.strong("Key biological applications:")),
                        ui.tags.ul(
                            ui.tags.li("Cell type identification: Distinct populations should form separate clusters"),
                            ui.tags.li("Developmental trajectories: Continuous transitions between states"),
                            ui.tags.li("Subpopulation discovery: Fine structure within major cell types"),
                            ui.tags.li("Quality control: Identifying outliers or batch effects")
                        ),
                        class_="alert alert-info"
                    ),
                    class_="mb-4"
                ),

                # Dataset Requirements  
                ui.h4("Data Requirements"),
                ui.div(
                    ui.p("UMAP performs best on properly preprocessed single-cell data:"),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Normalized:"), " Library size correction and log-transformation applied"),
                        ui.tags.li(ui.strong("Feature-selected:"), " Highly variable genes identified (typically 2000-5000)"),
                        ui.tags.li(ui.strong("Scaled (optional):"), " Zero-centered and unit variance per gene"),
                        ui.tags.li(ui.strong("Reduced:"), " PCA or other dimensionality reduction computed")
                    ),
                    ui.p("The platform accepts H5AD files (AnnData format) commonly used in scanpy workflows. "
                         "Raw count matrices can be processed using the preprocessing panel."),
                    class_="alert alert-success"
                ),

                # Parameter Selection Guide
                ui.h4("Parameter Selection Guide", class_="text-primary mt-4"),
                
                ui.h5("Why Default Parameters Work"),
                ui.p("The default parameters (n_neighbors=15, min_dist=0.5) are empirically validated across "
                     "diverse single-cell datasets. They provide a balanced view suitable for initial exploration:"),
                ui.tags.ul(
                    ui.tags.li("n_neighbors=15: Captures local structure while maintaining global relationships"),
                    ui.tags.li("min_dist=0.5: Prevents artificial cluster separation while showing genuine structure"),
                    ui.tags.li("40 PCA components: Retains most biological variation while filtering noise")
                ),
                
                ui.h5("When to Adjust Parameters"),
                ui.div(
                    ui.p(ui.strong("Clusters appear fragmented:")),
                    ui.tags.ul(
                        ui.tags.li("Increase n_neighbors (try 30-50) to emphasize global structure"),
                        ui.tags.li("Increase min_dist (try 0.3-0.5) to reduce artificial separation")
                    ),
                    ui.p(ui.strong("Missing fine structure:")),
                    ui.tags.ul(
                        ui.tags.li("Decrease n_neighbors (try 5-10) to preserve local details"),
                        ui.tags.li("Decrease min_dist (try 0.01-0.1) to allow tighter clustering")
                    ),
                    ui.p(ui.strong("Trajectory appears discontinuous:")),
                    ui.tags.ul(
                        ui.tags.li("Increase both n_neighbors and min_dist for smoother transitions"),
                        ui.tags.li("Consider using more PCA components to capture gradual changes")
                    ),
                    class_="alert alert-light"
                ),

                # Troubleshooting Section
                ui.h4("Troubleshooting Common Issues", class_="mt-4"),
                ui.div(
                    ui.h6("UMAP produces unexpected 'blob' with no structure"),
                    ui.p("Likely causes: Data not properly normalized, too few variable genes selected, or batch effects dominating."),
                    ui.p(ui.em("Solution: Verify preprocessing, especially normalization and HVG selection. Check for batch effects.")),
                    
                    ui.h6("Cells of same type split into multiple clusters", class_="mt-3"),
                    ui.p("Likely causes: n_neighbors too small, technical batch effects, or cell cycle effects."),
                    ui.p(ui.em("Solution: Increase n_neighbors to 30-50. Consider batch correction if samples were processed separately.")),
                    
                    ui.h6("Known distinct cell types merge together", class_="mt-3"),
                    ui.p("Likely causes: n_neighbors too large, insufficient genes retained, or populations genuinely similar."),
                    ui.p(ui.em("Solution: Decrease n_neighbors to 5-15. Verify marker genes are included in variable genes.")),
                    
                    ui.h6("UMAP looks different each run with same parameters", class_="mt-3"),
                    ui.p("Likely causes: Random seed not set, or using spectral initialization with sparse data."),
                    ui.p(ui.em("Solution: Set random_state to any integer for reproducibility. Different seeds produce similar topology.")),
                    
                    ui.h6("Computation takes too long", class_="mt-3"),
                    ui.p("Likely causes: Too many cells, using too many PCA components, or max iterations too high."),
                    ui.p(ui.em("Solution: Downsample to 10-20k cells for exploration. Use 30-50 PCA components. Leave maxiter empty for auto.")),
                    
                    class_="alert alert-warning"
                ),

                # Interpretation Guidelines
                ui.h4("Interpreting UMAP Results", class_="mt-4"),
                ui.p("Important considerations when interpreting UMAP embeddings:"),
                ui.tags.ul(
                    ui.tags.li(ui.strong("Distances:"), " Only local distances are meaningful. Distance between far clusters is not interpretable."),
                    ui.tags.li(ui.strong("Density:"), " Point density doesn't reflect cell abundance due to UMAP's uniform distribution tendency."),
                    ui.tags.li(ui.strong("Continuity:"), " Connections between clusters may indicate trajectories but require validation."),
                    ui.tags.li(ui.strong("Validation:"), " Always verify structure using known marker genes and biological knowledge.")
                ),

                # Export and Reproducibility
                ui.h4("Reproducibility", class_="mt-4"),
                ui.p("Download the parameters JSON to document your analysis. This includes all settings needed "
                     "to reproduce the embedding. When reporting UMAP visualizations, always include:"),
                ui.tags.ul(
                    ui.tags.li("n_neighbors and min_dist values"),
                    ui.tags.li("Number of input dimensions (PCs or other)"),
                    ui.tags.li("Preprocessing steps applied"),
                    ui.tags.li("Random seed if reproducibility is critical")
                ),
                
                # Seurat Conversion
                ui.h4("Converting from Seurat", class_="mt-4"),
                ui.p("For Seurat users, we provide an R script to convert .rds objects to H5AD format:"),
                
                ui.div(
                    ui.download_button("download_seurat_converter_docs", 
                                     "Download Conversion Script", 
                                     class_="btn-primary mb-3"),
                    ui.p("The script uses the zellconverter package to preserve all embeddings, metadata, and expression data."),
                    class_="alert alert-light"
                )
            )
        )
    )
)


def server(input, output, session):
    """Enhanced server function with robust color selection and progress bars"""
    
    # Initialize temp file manager
    session_id = str(uuid.uuid4())
    temp_manager = TempFileManager()
    temp_manager.init_session(session_id)
    
    # Clean up old sessions on startup
    temp_manager.cleanup_old_sessions()
    
    # Register cleanup on session end
    session.on_ended(lambda: temp_manager.cleanup())

    # Session data storage using reactive values
    session_data = reactive.Value({
        'original_adata': None,
        'preprocessed_adata': None,
        'umap_adata': None,
        'test_adata': None,
        'uploaded_file_path': None
    })

    # Analysis state tracking
    analysis_state = reactive.Value({
        'has_umap': False,
        'preprocessing_done': False,
        'new_umap_computed': False,
        'data_loaded': False,
        'data_loading': False,
        'has_pca': False,
        'available_reductions': {},
        'processing_state': None,
        'is_test_data': False,
        'downsampled': False,
        'dataset_quality': None
    })

    # Simplified reactive values
    data_version = reactive.Value(0)
    file_info_reactive = reactive.Value(None)
    downsample_settings = reactive.Value({'apply': None, 'max_cells': None})
    data_source = reactive.Value({'type': None, 'path': None, 'is_test': False})
    current_dataset_id = reactive.Value(None)
    is_computing = reactive.Value(False)
    umap_computation_params = reactive.Value(None)
    
    # Simple individual reactive values for UI state
    show_getting_started = reactive.Value(True)
    data_panel_expanded = reactive.Value(True)
    analysis_sections_enabled = reactive.Value(False)
    upload_mode = reactive.Value(False)  # Fixed: proper reactive value for upload mode
    
    # Progress tracking
    current_progress = reactive.Value(None)
    
    # FIXED DENDROGRAM STATE - NO INFINITE LOOPS
    current_dendrogram = reactive.Value(None)
    dendrogram_collapsed = reactive.Value(True)
    dendrogram_cache = reactive.Value({})  # Cache computed dendrograms
    dendrogram_computation_in_progress = reactive.Value(False)  # Prevent concurrent computation
    last_dendrogram_request = reactive.Value(None)  # Track last request to prevent duplicates
    
    # Color selection state - for persistence
    color_selection_state = reactive.Value({
        "color_mode": "none",
        "metadata_column": None,
        "gene": None
    })

    @render.ui
    def progress_ui():
        """Show progress indicator when operations are running"""
        progress_info = current_progress()
        if not progress_info:
            return None
            
        return ui.div(
            ui.div(
                ui.h6(progress_info.get('title', 'Processing...'), class_="mb-2"),
                ui.div(class_="progress-bar"),
                ui.div(progress_info.get('message', ''), class_="progress-text"),
                class_="progress-container"
            )
        )

    @render.ui
    def getting_started_ui():
        """Show simplified getting started guide when requested"""
        if show_getting_started():
            return ui.div(
                ui.h4("UMAP Analysis for Single-Cell Data"),
                ui.p("This platform enables systematic optimization of UMAP parameters to visualize single-cell populations. "
                     "UMAP preserves both local cell-cell relationships and global population structure, making it valuable "
                     "for identifying cell types, states, and trajectories."),
                ui.p("Begin by loading your preprocessed dataset (normalized, log-transformed) to explore how different "
                     "parameter settings affect cluster visualization and biological interpretation."),
                ui.div(
                    ui.input_action_button("dismiss_getting_started", "Begin", class_="btn-sm btn-outline-secondary"),
                    class_="mt-2"
                ),
                class_="getting-started"
            )
        return None

    @reactive.Effect
    @reactive.event(input.dismiss_getting_started)
    def dismiss_getting_started():
        """Dismiss the getting started guide"""
        show_getting_started.set(False)

    @render.ui
    def data_loading_panel_ui():
        """Enhanced data loading panel with always-available upload option"""
        data_version()  # React to data changes
        state = analysis_state()
        
        # Check if we're in "upload new" mode
        upload_new_mode = upload_mode()
        
        # Always show upload option, but style differently based on state
        if not state.get('data_loaded', False) or upload_new_mode:
            # No data loaded OR user clicked "Upload New Dataset" - show full panel
            panel_class = "data-loading-panel"
            
            header_content = [
                ui.h4("Data Input"),
                status_badge('info', 'No Dataset' if not state.get('data_loaded', False) else 'Replace Dataset')
            ]
            
            content = [
                ui.div(
                    *header_content,
                    class_="section-header"
                ),
                ui.div(
                    ui.h5("Load Dataset"),
                    ui.input_file("file", "Select H5AD File",
                                accept=[".h5ad"],
                                button_label="Browse",
                                placeholder="AnnData format (.h5ad)"),
                    ui.div(
                        ui.p("Or use reference dataset for testing:", class_="text-muted small mt-2 mb-1"),
                        ui.input_action_button("load_test_data",
                                            "Load PBMC 10k Dataset",
                                            class_="btn-outline-primary btn-sm"),
                        ui.span(
                            info_button("test_data",
                                    "PBMC 10k Reference Dataset",
                                    "10,902 PBMCs and granulocytes from 10x Genomics Multiome v1.0. Pre-processed with standard workflow including normalization, HVG selection (n=3,026), PCA, and existing UMAP embedding."),
                            class_="ml-2"
                        )
                    )
                )
            ]

            # Add cancel button if we have existing data
            if state.get('data_loaded', False) and upload_new_mode:
                content.append(
                    ui.div(
                        ui.input_action_button("cancel_upload_new", "Cancel", 
                                             class_="btn-sm btn-secondary mt-2"),
                        class_="mt-2"
                    )
                )
            
        elif data_panel_expanded():
            # Data loaded and expanded
            panel_class = "data-loading-panel loaded"
            
            adata = get_current_adata()
            source = data_source()
            dataset_analysis = state.get('dataset_quality') or {}
            
            # Dataset information
            if source and source.get('is_test', False):
                dataset_name = "PBMC 10k Reference Dataset"
            else:
                dataset_name = source.get('filename', 'H5AD Dataset') if source else 'H5AD Dataset'
            
            if adata:
                current_cells = adata.n_obs
                current_genes = adata.n_vars
                
                settings = downsample_settings()
                if state.get('downsampled', False) and settings.get('original_n_cells'):
                    original_cells = settings['original_n_cells']
                    dimension_text = f"{current_cells:,}/{original_cells:,} cells × {current_genes:,} genes (downsampled)"
                else:
                    dimension_text = f"{current_cells:,} cells × {current_genes:,} genes"
            else:
                dimension_text = "Dimensions: Loading..."
            
            header_content = [
                ui.h4("Dataset Information"),
                status_badge('success', 'Loaded & Ready')
            ]
            
            content = [
                ui.div(
                    *header_content,
                    class_="section-header"
                ),
                ui.div(
                    ui.p(ui.strong("Dataset: "), dataset_name, class_="mb-2"),
                    ui.p(ui.strong("Dimensions: "), dimension_text, class_="mb-2"),
                    class_="mt-2"
                ),
                ui.div(
                    ui.input_action_button("collapse_data_panel", "Minimize", 
                                         class_="btn-sm btn-outline-secondary mr-2"),
                    ui.input_action_button("upload_new_dataset", "Replace Dataset", 
                                         class_="btn-sm btn-primary"),
                    class_="mt-2"
                )
            ]
            
        else:
            # Data loaded but minimized
            adata = get_current_adata()
            source = data_source()
            
            compact_info = "Dataset loaded"
            if adata:
                current_cells = adata.n_obs
                current_genes = adata.n_vars
                
                if source and source.get('is_test', False):
                    settings = downsample_settings()
                    if state.get('downsampled', False) and settings.get('original_n_cells'):
                        original_cells = settings['original_n_cells']
                        compact_info = f"{current_cells:,}/{original_cells:,} cells | PBMC 10k Reference"
                    else:
                        compact_info = f"{current_cells:,} cells | PBMC 10k Reference"
                else:
                    filename = source.get('filename', 'Dataset') if source else 'Dataset'
                    truncated_filename = truncate_filename(filename)
                    
                    settings = downsample_settings()
                    if state.get('downsampled', False) and settings.get('original_n_cells'):
                        original_cells = settings['original_n_cells']
                        compact_info = f"{current_cells:,}/{original_cells:,} cells × {current_genes:,} genes | {truncated_filename}"
                    else:
                        compact_info = f"{current_cells:,} cells × {current_genes:,} genes | {truncated_filename}"
            
            return ui.div(
                ui.div(
                    ui.span(ui.tags.i(class_="fas fa-database text-success"), f" {compact_info}"),
                    ui.div(
                        ui.input_action_button("expand_data_panel", "Details", 
                                             class_="btn-sm btn-outline-primary mr-2"),
                        ui.input_action_button("upload_new_dataset", "Replace Dataset", 
                                             class_="btn-sm btn-primary"),
                        class_="d-flex"
                    ),
                    class_="d-flex justify-content-between align-items-center"
                ),
                class_="alert alert-success"
            )
        
        return ui.div(*content, class_=panel_class)

    @reactive.Effect
    @reactive.event(input.collapse_data_panel)
    def collapse_data_panel():
        """Collapse the data loading panel"""
        data_panel_expanded.set(False)

    @reactive.Effect
    @reactive.event(input.expand_data_panel)
    def expand_data_panel():
        """Expand the data loading panel"""
        data_panel_expanded.set(True)

    @reactive.Effect
    @reactive.event(input.upload_new_dataset)
    def upload_new_dataset():
        """Show upload interface for new dataset"""
        upload_mode.set(True)
        data_panel_expanded.set(True)
        data_version.set(data_version() + 1)
        ui.notification_show("Select H5AD file. Current analysis will be replaced upon loading.", type="info", duration=3)

    @reactive.Effect
    @reactive.event(input.cancel_upload_new)
    def cancel_upload_new():
        """Cancel the upload new dataset operation"""
        upload_mode.set(False)
        data_panel_expanded.set(False)
        data_version.set(data_version() + 1)
        ui.notification_show("Upload cancelled - current dataset retained", type="info", duration=3)

    @render.ui
    def validation_messages_ui():
        """Show dataset validation and processing information with smart warnings"""
        state = analysis_state()
        if not state['data_loaded']:
            return None
            
        dataset_analysis = state.get('dataset_quality') or {}
        if not dataset_analysis:
            return None
            
        messages = []
        
        # Enhanced data state detection
        data_state = dataset_analysis.get('data_state', 'unknown')
        
        # Check if preprocessing is recommended
        needs_preprocessing = False
        preprocessing_reason = []
        
        adata = get_current_adata()
        if adata:
            # Check for raw counts
            if data_state == 'raw counts':
                needs_preprocessing = True
                preprocessing_reason.append("Dataset appears to contain raw counts")
            
            # Check for missing PCA
            if 'X_pca' not in adata.obsm and adata.n_vars > 1000:
                needs_preprocessing = True
                preprocessing_reason.append("No PCA computed")
            
            # Check for missing HVG
            if 'highly_variable' not in adata.var.columns and adata.n_vars > 5000:
                needs_preprocessing = True
                preprocessing_reason.append("No highly variable genes selected")
        
        # Create appropriate message
        if needs_preprocessing:
            messages.append(
                ui.div(
                    ui.tags.i(class_="fas fa-exclamation-triangle mr-2"),
                    ui.strong("Preprocessing recommended: "),
                    "For optimal UMAP results, apply normalization (account for sequencing depth), "
                    "log-transformation (reduce impact of highly expressed genes), and "
                    "variable gene selection (focus on informative features). ",
                    ui.tags.a(
                        "Open preprocessing panel",
                        href="#",
                        onclick="""
                        // Use Bootstrap Collapse API to open preprocessing panel
                        try {
                            // First try to find the preprocessing panel button in the main accordion
                            const accordion = document.getElementById('main_accordion');
                            if (accordion) {
                                // Look for buttons with preprocessing in their target
                                const buttons = accordion.querySelectorAll('button[data-bs-toggle="collapse"]');
                                for (const button of buttons) {
                                    const target = button.getAttribute('data-bs-target') || button.getAttribute('aria-controls');
                                    if (target && target.includes('preprocessing')) {
                                        console.log('Opening preprocessing panel');
                                        button.click();
                                        return false;
                                    }
                                }
                                
                                // Fallback: look for the first accordion item (which should be preprocessing)
                                const firstButton = accordion.querySelector('button[data-bs-toggle="collapse"]');
                                if (firstButton) {
                                    console.log('Opening first panel (preprocessing)');
                                    firstButton.click();
                                }
                            }
                        } catch (e) {
                            console.error('Error opening preprocessing panel:', e);
                        }
                        return false;
                        """
                    ),
                    class_="validation-warning"
                )
            )
        elif data_state in ['normalized & scaled', 'normalized with reductions', 'scaled', 'processed']:
            messages.append(
                ui.div(
                    ui.tags.i(class_="fas fa-check-circle mr-2"),
                    ui.strong("Data ready: "),
                    "Dataset is preprocessed and suitable for UMAP analysis",
                    class_="validation-success"
                )
            )
        else:
            state_descriptions = {
                'normalized': "Data appears normalized - suitable for analysis",
                'unknown': "Dataset preprocessing state unclear - review if UMAP results appear suboptimal",
                'error': "Error analyzing data state"
            }
            state_desc = state_descriptions.get(data_state, f"Data state: {data_state}")
            messages.append(
                ui.div(
                    ui.tags.i(class_="fas fa-info-circle mr-2"),
                    ui.strong("Data status: "),
                    state_desc,
                    class_="validation-info"
                )
            )
        
        return ui.div(*messages) if messages else None

    @render.ui 
    def analysis_sections_ui():
        """Render analysis sections with proper enabling/disabling"""
        state = analysis_state()
        
        sections_enabled = state['data_loaded'] and not state.get('data_loading', False)
        section_class = "" if sections_enabled else "section-disabled"
        
        # Create sections with status badges
        preprocessing_status = "success" if state['preprocessing_done'] else ("info" if sections_enabled else "disabled")
        preprocessing_badge = status_badge(preprocessing_status, 
                                         "Applied" if state['preprocessing_done'] else 
                                         ("Available" if sections_enabled else "Disabled"))
        
        viz_status = "success" if sections_enabled else "disabled"
        viz_badge = status_badge(viz_status, "Available" if sections_enabled else "Disabled")
        
        umap_status = ("success" if state['has_umap'] else "info") if sections_enabled else "disabled"
        umap_badge = status_badge(umap_status, 
                                "Computed" if state['has_umap'] else 
                                ("Ready" if sections_enabled else "Disabled"))
        
        # Create accordion sections in new logical order
        accordion_sections = []
        
        # Preprocessing Pipeline (collapsed by default since most won't need it)
        accordion_sections.append(
            ui.accordion_panel(
                ui.div(
                    "Preprocessing Pipeline (Optional)",
                    preprocessing_badge,
                    class_="section-header"
                ),
                ui.p("Apply standard single-cell preprocessing steps if your data contains raw counts:", class_="text-muted"),
                ui.input_numeric("min_genes", "Minimum genes per cell (filter low-quality cells):", 
                               value=200, min=0, step=50),
                ui.input_numeric("min_cells", "Minimum cells per gene (remove rarely detected genes):", 
                               value=3, min=0, step=1),
                ui.input_checkbox("normalize", "Normalize and log-transform expression values", 
                                value=True),
                ui.input_numeric("n_top_genes", "Number of highly variable genes to retain:", 
                               value=2000, min=100, max=10000, step=100),
                ui.input_action_button("run_preprocessing", 
                                     "Apply Preprocessing", 
                                     class_="btn-warning btn-block"),
                value="preprocessing"
            )
        )
        
        # Visualization & Exploration (EARLY - check existing results first!)
        accordion_sections.append(
            ui.accordion_panel(
                ui.div(
                    "Visualization & Exploration",
                    viz_badge,
                    class_="section-header"
                ),
                ui.p("Color cells by metadata or gene expression to assess biological structure. "
                     "Well-separated clusters with consistent metadata annotation suggest appropriate parameters. "
                     "Mixed or fragmented populations may indicate parameter adjustment is needed.", class_="text-muted"),
                ui.output_ui("color_selector_ui"),
                ui.br(),
                ui.input_checkbox("show_legend", "Display legend", value=True),
                ui.input_slider("point_size", "Point size:", 
                              min=1, max=100, value=30, step=5),
                ui.output_ui("color_map_ui"),
                value="visualization"
            )
        )

        # Dataset Analysis & Input Configuration  
        accordion_sections.append(
            ui.accordion_panel(
                ui.div(
                    "Dataset Analysis & Input Configuration",
                    status_badge("info" if sections_enabled else "disabled", 
                               "Available" if sections_enabled else "Disabled"),
                    class_="section-header"
                ),
                ui.p("Configure which dimensionality reduction to use as UMAP input. "
                     "PCA is standard for scRNA-seq. Use more components to capture more variation, "
                     "fewer for faster computation and noise reduction.", class_="text-muted"),
                ui.output_ui("dimensionality_reduction_info"),
                ui.output_ui("reduction_selector_ui"),
                
                ui.h6("Input Data Configuration", class_="mt-3"),
                ui.div(
                    ui.span("Components from selected embedding (default 40 balances information and speed):"),
                    info_button("n_embedding_components", 
                              "Components from Embedding",
                              "Number of dimensions to use from the selected reduction. More components preserve more variation but increase computation time. 30-50 typical for most datasets."),
                    ui.input_numeric("n_embedding_components", "", 
                                   value=40, min=5, max=100, step=5)
                ),
                
                ui.div(
                    ui.span("Components for neighbor graph (default 50 for comprehensive neighborhoods):"),
                    info_button("n_pcs", 
                              "Components for Neighbors",
                              "Dimensions used to find nearest neighbors. Should be ≥ n_neighbors to capture sufficient variation. 50 works well for most single-cell datasets."),
                    ui.input_numeric("n_pcs", "", 
                                   value=50, min=10, max=100, step=5)
                ),
                
                ui.div(
                    ui.span("Distance metric:"),
                    info_button("distance_metric", 
                              "Distance Metric",
                              "How to measure cell-cell similarity. Euclidean: standard for normalized data. Cosine: emphasizes relative expression patterns. Manhattan: less sensitive to outliers."),
                    ui.input_select("distance_metric", "", 
                                   choices={
                                       "euclidean": "Euclidean (standard)",
                                       "cosine": "Cosine (relative patterns)", 
                                       "manhattan": "Manhattan (robust)"
                                   },
                                   selected="euclidean")
                ),
                value="input_config"
            )
        )
        
        # UMAP Parameter Tuning (after seeing visualization)
        accordion_sections.append(
            ui.accordion_panel(
                ui.div(
                    "UMAP Parameter Tuning",
                    umap_badge,
                    class_="section-header"
                ),
                ui.p("Adjust parameters to optimize embedding for your biological question. "
                     "Default values (n_neighbors=15, min_dist=0.5) provide balanced local/global structure "
                     "suitable for initial exploration of most single-cell datasets.", class_="text-info"),
                
                ui.h6("Core Parameters"),
                ui.div(
                    ui.span("Number of neighbors:"),
                    info_button("n_neighbors", 
                              "Number of Neighbors",
                              "Number of neighboring cells used to learn data structure. Smaller values (5-15) preserve fine detail and may separate similar cell types. Larger values (30-50) emphasize major populations and trajectories. Default of 15 balances local and global structure."),
                    ui.input_numeric("n_neighbors", "", 
                                   value=15, min=2, max=200, step=1)
                ),

                ui.div(
                    ui.span("Embedding dimensions:"),
                    info_button("n_components", 
                              "Embedding Dimensions",
                              "Number of UMAP dimensions. Use 2 for standard visualization. 3 Dimensions can be downloaded but won't be visualized."),
                    ui.input_numeric("n_components", "", 
                                   value=2, min=2, max=3, step=1)
                ),

                ui.div(
                    ui.span("Minimum distance:"),
                    info_button("min_dist", 
                              "Minimum Distance",
                              "Controls how tightly cells cluster. Lower values (0.01-0.3) create compact, separated clusters - useful for identifying distinct cell types. Higher values (0.3-0.5) show more continuous transitions - better for trajectories. Default of 0.5 prevents artificial separation."),
                    ui.input_numeric("min_dist", "", 
                                   value=0.5, min=0.0, max=1.0, step=0.01)
                ),

                ui.div(
                    ui.span("Embedding spread:"),
                    info_button("spread", 
                              "Spread",
                              "Controls overall scale of the embedding. Usually kept at 1.0. Reduce (<1.0) to bring clusters closer; increase (>1.0) to spread them apart. Adjust if visualization appears too cramped or dispersed."),
                    ui.input_numeric("spread", "", 
                                   value=1.0, min=0.1, max=5.0, step=0.1)
                ),

                ui.hr(),
                ui.h6("Advanced Parameters"),

                ui.div(
                    ui.span("Random seed:"),
                    info_button("random_state", 
                              "Random State",
                              "Seed for random number generation. Set to any integer for reproducible embeddings across runs. Different seeds may produce rotated/reflected but topologically similar embeddings."),
                    ui.input_numeric("random_state", "", 
                                   value=0, min=0, step=1)
                ),

                ui.div(
                    ui.span("Maximum iterations:"),
                    info_button("maxiter", 
                              "Maximum Iterations",
                              "Number of embedding optimization steps. Leave empty for automatic selection based on dataset size. Increase if embedding appears incomplete; decrease for faster computation."),
                    ui.input_numeric("maxiter", "(Auto if empty)", 
                                   value=None, min=10, max=1000, step=50)
                ),

                ui.div(
                    ui.span("Learning rate (α):"),
                    info_button("alpha", 
                              "Alpha (Learning Rate)",
                              "Initial learning rate for embedding optimization. Default 1.0 works well. Reduce if embedding is unstable; increase for faster convergence on large datasets."),
                    ui.input_numeric("alpha", "", 
                                   value=1.0, min=0.01, max=10.0, step=0.1)
                ),

                ui.div(
                    ui.span("Repulsion strength (γ):"),
                    info_button("gamma", 
                              "Gamma (Repulsion Strength)",
                              "Weight of repulsive forces between distant cells. Default 1.0 balances attraction and repulsion. Reduce to allow more overlap; increase for stronger separation."),
                    ui.input_numeric("gamma", "", 
                                   value=1.0, min=0.0, max=10.0, step=0.1)
                ),

                ui.div(
                    info_button("init_pos", 
                              "Initialization Position",
                              "Use existing UMAP coordinates as starting positions. Can speed convergence when fine-tuning parameters. May bias toward previous structure."),
                    ui.input_checkbox("init_pos", 
                                    "Initialize from existing embedding", 
                                    value=False)
                ),

                ui.hr(),
                ui.input_action_button("reset_umap_params", 
                                     "Reset to Defaults", 
                                     class_="btn-outline-secondary btn-sm btn-block"),
                value="umap_params"
            )
        )
        
        # Determine which panels to open by default
        open_panels = False
        
        # Main content with accordion
        content = [
            ui.div(
                ui.accordion(
                    *accordion_sections,
                    open=open_panels,
                    id="main_accordion"  # Add ID for JavaScript targeting
                ),
                class_=section_class
            ),
            ui.br(),
            ui.output_ui("umap_button_ui"),
            ui.br(),
            ui.div(
                ui.h6("Export Options"),
                ui.download_button("download_umap", "Download UMAP Coordinates", 
                                 class_="btn-info btn-block mb-2"),
                ui.p("CSV file with UMAP positions and cell metadata for downstream analysis", 
                     class_="text-muted small mb-2"),
                
                ui.download_button("download_anndata", "Download Complete Dataset", 
                                 class_="btn-info btn-block mb-2"),
                ui.p("H5AD file with computed UMAP for use in scanpy/Seurat workflows", 
                     class_="text-muted small mb-2"),
                
                ui.download_button("download_params", "Download Analysis Parameters", 
                                 class_="btn-info btn-block mb-2"),
                ui.p("JSON file with all settings for methods reporting and reproducibility", 
                     class_="text-muted small mb-3"),
                
                style_="padding-bottom: 50px;"  # Extra space for scrolling past buttons
            )
        ]
        
        return ui.div(*content)

    # Load test dataset
    @reactive.Effect
    @reactive.event(input.load_test_data)
    def load_test_dataset():
        """Load the PBMC10k reference dataset with progress indication"""
        try:
            current_progress.set({
                'title': 'Load PBMC 10k Test data',
                'message': 'Loading 10,902 cells with pre-computed embeddings...'
            })

            # Load the dataset
            adata = sc.read_h5ad("pbmc10k_rna.h5ad")

            # Memory optimization
            adata = auto_optimize_memory(adata)

            # Generate unique dataset ID for this session
            dataset_id = f"test_{time.time()}"
            current_dataset_id.set(dataset_id)

            # Clear old data and set test data
            current_session_data = session_data()
            current_session_data['test_adata'] = adata
            current_session_data['original_adata'] = None
            current_session_data['preprocessed_adata'] = None
            current_session_data['umap_adata'] = None
            session_data.set(current_session_data)

            # Set data source
            data_source.set({
                'type': 'test', 
                'path': None,
                'is_test': True
            })

            # Skip downsampling for test dataset
            downsample_settings.set({'apply': False, 'max_cells': None, 'original_n_cells': None})

            # Analyze dataset
            dataset_analysis = analyze_dataset_quality(adata)
            available_reductions = detect_available_reductions(adata)

            # Update analysis state
            state = {
                'has_umap': 'X_umap' in adata.obsm,
                'preprocessing_done': False,
                'new_umap_computed': False,
                'data_loaded': True,
                'data_loading': False,
                'downsampled': False,
                'has_pca': 'X_pca' in adata.obsm,
                'is_test_data': True,
                'available_reductions': available_reductions,
                'dataset_quality': dataset_analysis
            }
            analysis_state.set(state)

            # Clear UMAP parameters since this is pre-existing
            umap_computation_params.set(None)

            # Update UI state
            analysis_sections_enabled.set(True)
            data_panel_expanded.set(False)  # Auto-minimize after loading
            show_getting_started.set(False)

            # Reset color selection state for new dataset
            color_selection_state.set({
                "color_mode": "none",
                "metadata_column": None,
                "gene": None
            })
            
            # Reset dendrogram state for new dataset
            current_dendrogram.set(None)
            dendrogram_collapsed.set(True)
            last_dendrogram_request.set(None)

            # Clear progress
            current_progress.set(None)

            ui.notification_show("PBMC 10k reference dataset loaded (10,902 cells × 3,026 genes)", type="success", duration=3)  # Updated counts

            # Force UI refresh
            data_version.set(data_version() + 1)

        except Exception as e:
            current_progress.set(None)
            logger.error(f"Error loading reference dataset: {e}")
            ui.notification_show(f"Error loading reference dataset: {str(e)}", type="error", duration=10)

    # File upload handling with progress and enhanced debugging
    def load_uploaded_file():
        """Load uploaded file with progress indication and system reset"""
        source = data_source()
        if not source or source['type'] != 'upload' or not source['path']:
            logger.error("Invalid data source for file loading")
            return
            
        file_path = source['path']
        logger.info(f"=== STARTING FILE LOAD ===")
        logger.info(f"Attempting to load file: {file_path}")
        logger.info(f"File exists check: {os.path.exists(file_path)}")
        
        # Check if file exists before proceeding
        if not os.path.exists(file_path):
            logger.error(f"File does not exist at path: {file_path}")
            ui.notification_show(f"Uploaded file not found at: {file_path}", type="error")
            return
            
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size} bytes")
        except Exception as e:
            logger.error(f"Cannot get file size: {e}")
            
        # RESET SYSTEM when actually loading new file
        logger.info("Resetting system for new dataset upload")
        
        # Reset all reactive values
        session_data.set({
            'original_adata': None,
            'preprocessed_adata': None,
            'umap_adata': None,
            'test_adata': None,
            'uploaded_file_path': file_path  # Keep the current file path
        })
        
        analysis_state.set({
            'has_umap': False,
            'preprocessing_done': False,
            'new_umap_computed': False,
            'data_loaded': False,
            'data_loading': True,  # Set loading state
            'has_pca': False,
            'available_reductions': {},
            'processing_state': None,
            'is_test_data': False,
            'downsampled': False,
            'dataset_quality': None
        })
        
        # Reset UI state
        data_panel_expanded.set(False)
        show_getting_started.set(False)
        current_dataset_id.set(None)
        current_progress.set(None)
        
        # Reset color selection and dendrogram state for new dataset
        color_selection_state.set({
            "color_mode": "none",
            "metadata_column": None,
            "gene": None
        })
        current_dendrogram.set(None)
        dendrogram_collapsed.set(True)
        last_dendrogram_request.set(None)
        
        settings = downsample_settings()
        
        try:
            current_progress.set({
                'title': 'Loading Dataset',
                'message': 'Reading H5AD file and validating structure...'
            })

            # Load h5ad file with explicit error handling
            try:
                adata = sc.read_h5ad(file_path)
                logger.info(f"Successfully loaded h5ad file: {adata.n_obs} cells x {adata.n_vars} genes")
            except Exception as h5ad_error:
                logger.error(f"Error loading h5ad file: {h5ad_error}")
                raise h5ad_error

            if adata.n_obs == 0 or adata.n_vars == 0:
                raise ValueError("Dataset contains no cells or genes")

            # Apply downsampling if requested
            original_n_cells = adata.n_obs
            if settings.get('apply', False) and settings.get('max_cells') and adata.n_obs > settings['max_cells']:
                current_progress.set({
                    'title': 'Processing Dataset',
                    'message': f'Downsampling from {adata.n_obs:,} to {settings["max_cells"]:,} cells...'
                })
                
                sc.pp.subsample(adata, n_obs=settings['max_cells'], random_state=42)
                settings['original_n_cells'] = original_n_cells
                downsample_settings.set(settings)

            # Memory optimization
            adata = auto_optimize_memory(adata)

            # Generate unique dataset ID
            dataset_id = f"upload_{time.time()}"
            current_dataset_id.set(dataset_id)

            # Comprehensive dataset analysis
            dataset_analysis = analyze_dataset_quality(adata)
            available_reductions = detect_available_reductions(adata)

            # Update analysis state
            state = {
                'has_umap': 'X_umap' in adata.obsm,
                'preprocessing_done': False,
                'new_umap_computed': False,
                'data_loaded': True,
                'data_loading': False,
                'downsampled': settings.get('apply', False),
                'has_pca': 'X_pca' in adata.obsm,
                'is_test_data': False,
                'available_reductions': available_reductions,
                'dataset_quality': dataset_analysis
            }
            analysis_state.set(state)

            # Clear UMAP parameters for uploaded data
            umap_computation_params.set(None)

            # Update UI state
            analysis_sections_enabled.set(True)
            data_panel_expanded.set(False)
            show_getting_started.set(False)

            # Reset color selection state for new dataset
            color_selection_state.set({
                "color_mode": "none",
                "metadata_column": None,
                "gene": None
            })

            # Store in session data
            current_session = session_data()
            current_session['original_adata'] = adata
            current_session['preprocessed_adata'] = None
            current_session['umap_adata'] = None
            current_session['test_adata'] = None
            current_session['uploaded_file_path'] = file_path  # Keep track of the file
            session_data.set(current_session)

            # Clear progress
            current_progress.set(None)

            logger.info("=== FILE LOAD COMPLETED SUCCESSFULLY ===")
            cells_text = f"{adata.n_obs:,} cells"
            if state.get('downsampled', False) and settings.get('original_n_cells'):
                cells_text = f"{adata.n_obs:,}/{settings['original_n_cells']:,} cells (downsampled)"
            ui.notification_show(f"Dataset loaded: {cells_text} × {adata.n_vars:,} genes", type="success", duration=3)
            
            # Force UI refresh
            data_version.set(data_version() + 1)
            gc.collect()

        except Exception as e:
            current_progress.set(None)
            logger.error(f"Error loading dataset: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            ui.notification_show(f"Error loading dataset: {str(e)}", type="error")
        finally:
            # Reset loading state
            state = analysis_state()
            state['data_loading'] = False
            analysis_state.set(state)

    # File upload event handling
    @reactive.Effect
    @reactive.event(input.file)
    def check_file_and_show_modal():
        """Check uploaded file and show downsampling modal if needed"""
        # Reset upload mode when file is actually selected
        upload_mode.set(False)
        
        request = input.file()
        if not request:
            return

        file_path = request[0]["datapath"]
        logger.info(f"Original uploaded file path: {file_path}")
        logger.info(f"File exists at original path: {os.path.exists(file_path)}")
        
        # Use temp manager for persistent storage
        try:
            import shutil
            
            # Create persistent file path using temp manager
            persistent_file_path = temp_manager.get_temp_path(".h5ad")
            logger.info(f"Target persistent file path: {persistent_file_path}")
            
            # Verify original file exists and get its size
            if not os.path.exists(file_path):
                raise Exception(f"Original uploaded file does not exist: {file_path}")
            
            original_size = os.path.getsize(file_path)
            logger.info(f"Original file size: {original_size} bytes")
            
            # Copy the uploaded file to our managed location
            shutil.copy2(file_path, persistent_file_path)
            logger.info(f"File copy completed")
            
            # Verify the copy was successful
            if not os.path.exists(persistent_file_path):
                raise Exception(f"File copy failed - destination file does not exist: {persistent_file_path}")
            
            copied_size = os.path.getsize(persistent_file_path)
            logger.info(f"Copied file size: {copied_size} bytes")
            
            if copied_size != original_size:
                raise Exception(f"File copy incomplete - size mismatch: {original_size} vs {copied_size}")
                
            logger.info(f"Successfully copied uploaded file to: {persistent_file_path}")
            
        except Exception as e:
            logger.error(f"Error copying uploaded file: {e}")
            ui.notification_show(f"Error processing uploaded file: {str(e)}", type="error")
            return

        # Set data source with the persistent file path
        data_source.set({
            'type': 'upload', 
            'path': persistent_file_path,
            'is_test': False,
            'filename': request[0]["name"]
        })

        # Update session data with the persistent file path
        current_session = session_data()
        current_session['uploaded_file_path'] = persistent_file_path
        session_data.set(current_session)

        # Get file info from the persistent copy
        logger.info(f"Getting file info for: {persistent_file_path}")
        info = get_file_info(persistent_file_path)
        file_info_reactive.set(info)

        if not info['is_valid']:
            logger.error(f"File validation failed: {info['error']}")
            ui.notification_show(f"Invalid dataset file: {info['error']}", type="error", duration=10)
            return

        logger.info(f"File validation successful - Size: {info['size_mb']:.1f}MB, Cells: {info.get('n_cells', 'unknown')}")

        # Check if downsampling modal should be shown
        should_show_modal = (info['size_mb'] > Config.FILE_SIZE_THRESHOLD_MB or 
                           (info['n_cells'] and info['n_cells'] > Config.MAX_CELLS_WITHOUT_PROMPT))

        if should_show_modal:
            recommended_cells = min(Config.DEFAULT_DOWNSAMPLE_SIZE, info['n_cells']) if info['n_cells'] else Config.DEFAULT_DOWNSAMPLE_SIZE

            m = ui.modal(
                ui.div(
                    ui.p(f"File Size: {info['size_mb']:.1f} MB"),
                    ui.p(f"Cells: {info['n_cells']:,}") if info['n_cells'] else None,
                    ui.p(f"Genes: {info['n_genes']:,}") if info['n_genes'] else None,
                    class_="alert alert-info"
                ),
                ui.input_checkbox("apply_downsampling", "Apply downsampling for performance optimization", value=False),
                ui.input_numeric("downsample_max_cells", "Maximum cells to retain:", 
                               value=recommended_cells, min=1000, max=50000, step=1000),
                title="Dataset Processing Options",
                easy_close=False,
                footer=ui.TagList(
                    ui.input_action_button("confirm_downsample", "Load Dataset", class_="btn-primary"),
                    ui.modal_button("Cancel", class_="btn-secondary"),
                ),
            )
            ui.modal_show(m)
        else:
            # Proceed directly
            downsample_settings.set({'apply': False, 'max_cells': None, 'original_n_cells': None})
            load_uploaded_file()

    @reactive.Effect
    @reactive.event(input.confirm_downsample)
    def handle_downsample_confirm():
        """Handle the downsampling confirmation"""
        settings = {
            'apply': input.apply_downsampling() if hasattr(input, 'apply_downsampling') else False,
            'max_cells': input.downsample_max_cells() if hasattr(input, 'downsample_max_cells') and input.apply_downsampling() else None,
            'original_n_cells': None
        }
        downsample_settings.set(settings)
        ui.modal_remove()
        load_uploaded_file()

    # Data access functions
    @reactive.Calc
    def data():
        """Get data from either uploaded file or test dataset"""
        source = data_source()
        if not source or source['type'] is None:
            return None

        current_session = session_data()

        if source['is_test']:
            return current_session.get('test_adata', None)

        return current_session.get('original_adata', None)

    @reactive.Calc
    def get_current_adata():
        """Get the most current version of adata"""
        data_version()  # React to data changes
        
        current_session = session_data()
        
        # Priority: UMAP data > preprocessed data > original/test data
        if current_session.get('umap_adata') is not None:
            return current_session['umap_adata']
        elif current_session.get('preprocessed_adata') is not None:
            return current_session['preprocessed_adata']
        elif current_session.get('test_adata') is not None:
            return current_session['test_adata']
        elif current_session.get('original_adata') is not None:
            return current_session['original_adata']

        return data()

    # Preprocessing with progress indication
    @reactive.Effect
    @reactive.event(input.run_preprocessing)
    def preprocess_data():
        """Apply preprocessing with progress indication"""
        adata = data()
        if adata is None:
            ui.notification_show("Please load dataset first", type="warning")
            return

        if is_computing():
            ui.notification_show("Another computation is in progress", type="warning")
            return

        try:
            is_computing.set(True)
            current_progress.set({
                'title': 'Preprocessing Dataset',
                'message': 'Applying quality control filters and normalization...'
            })

            # Create a copy to avoid modifying original
            adata_processed = adata.copy()

            # Basic filtering
            initial_cells = adata_processed.n_obs
            initial_genes = adata_processed.n_vars

            if input.min_genes() > 0:
                sc.pp.filter_cells(adata_processed, min_genes=input.min_genes())
            if input.min_cells() > 0:
                sc.pp.filter_genes(adata_processed, min_cells=input.min_cells())

            if adata_processed.n_obs == 0:
                raise ValueError("All cells were filtered out. Adjust filtering criteria.")

            # Store raw if needed
            should_store_raw = (adata_processed.raw is None and 
                              not ('highly_variable' in adata_processed.var.columns))
            if should_store_raw:
                adata_processed.raw = adata_processed.copy()

            current_progress.set({
                'title': 'Preprocessing Dataset',
                'message': 'Computing normalization and highly variable genes...'
            })

            # Normalization
            if input.normalize():
                X_sample = adata_processed.X[:100, :100]
                if hasattr(X_sample, 'toarray'):
                    X_sample = X_sample.toarray()

                if not (np.mean(X_sample) < 20 and np.min(X_sample) >= 0):
                    sc.pp.normalize_total(adata_processed, target_sum=1e4)
                    sc.pp.log1p(adata_processed)

            # Find highly variable genes
            try:
                sc.pp.filter_genes(adata_processed, min_cells=1)
                if 'highly_variable' not in adata_processed.var.columns:
                    sc.pp.highly_variable_genes(adata_processed, n_top_genes=input.n_top_genes())
                else:
                    adata_processed.var['highly_variable'] = True
            except Exception:
                adata_processed.var['highly_variable'] = True

            current_progress.set({
                'title': 'Preprocessing Dataset',
                'message': 'Computing PCA and finalizing preprocessing...'
            })

            # Scale and compute PCA
            sc.pp.scale(adata_processed, max_value=10)
            n_comps = min(50, adata_processed.n_vars-1, adata_processed.n_obs-1)
            sc.tl.pca(adata_processed, n_comps=n_comps)

            # Memory optimization
            adata_processed = auto_optimize_memory(adata_processed)

            # Update analysis
            dataset_analysis = analyze_dataset_quality(adata_processed)
            available_reductions = detect_available_reductions(adata_processed)

            # Store the preprocessed data
            current_session = session_data()
            current_session['preprocessed_adata'] = adata_processed
            session_data.set(current_session)

            # Update state with fresh reduction information
            state = analysis_state()
            state['preprocessing_done'] = True
            state['has_pca'] = 'X_pca' in adata_processed.obsm
            state['available_reductions'] = available_reductions
            state['dataset_quality'] = dataset_analysis
            analysis_state.set(state)

            # Clear progress
            current_progress.set(None)

            # Force UI refresh
            data_version.set(data_version() + 1)

            ui.notification_show(f"Preprocessing complete: {adata_processed.n_obs} cells × {adata_processed.n_vars} genes retained, PCA computed", type="success")
            
            logger.info(f"Available reductions after preprocessing: {list(available_reductions.keys())}")
            
            gc.collect()

        except Exception as e:
            current_progress.set(None)
            logger.error(f"Preprocessing error: {e}")
            ui.notification_show(f"Preprocessing error: {str(e)}", type="error")
        finally:
            is_computing.set(False)

    # Enhanced UMAP computation with robust color persistence
    @reactive.Effect
    @reactive.event(input.run_umap)
    def compute_umap():
        """Enhanced UMAP computation with robust color persistence"""
        if is_computing():
            ui.notification_show("Another computation is in progress", type="warning")
            return

        # Store current color settings SYNCHRONOUSLY
        preserved_color = {
            "color_mode": None,
            "metadata_column": None,
            "gene": None
        }
        
        try:
            preserved_color["color_mode"] = input.color_mode()
            if preserved_color["color_mode"] == "metadata" and hasattr(input, 'metadata_column'):
                preserved_color["metadata_column"] = input.metadata_column()
            elif preserved_color["color_mode"] == "gene" and hasattr(input, 'gene'):
                preserved_color["gene"] = input.gene()
        except:
            pass
        
        logger.info(f"Preserving color selection: {preserved_color}")

        # Get current data
        current_session = session_data()
        
        if current_session.get('umap_adata') is not None:
            adata = current_session['umap_adata'].copy()
        elif current_session.get('preprocessed_adata') is not None:
            adata = current_session['preprocessed_adata'].copy()
        elif current_session.get('original_adata') is not None:
            adata = current_session['original_adata'].copy()
        elif current_session.get('test_adata') is not None:
            adata = current_session['test_adata'].copy()
        else:
            ui.notification_show("Please load dataset first", type="warning")
            return

        try:
            is_computing.set(True)
            current_progress.set({
                'title': 'Computing UMAP Embedding',
                'message': 'Preparing parameters and validating inputs...'
            })

            # ROBUST parameter extraction with defaults
            def safe_get_input(attr_name, default_value):
                try:
                    if hasattr(input, attr_name):
                        val = getattr(input, attr_name)()
                        return val if val is not None else default_value
                    return default_value
                except:
                    return default_value

            # Get all parameters with defaults
            n_neighbors = safe_get_input('n_neighbors', 15)
            n_components = safe_get_input('n_components', 2)
            min_dist = safe_get_input('min_dist', 0.5)
            spread = safe_get_input('spread', 1.0)
            random_state = safe_get_input('random_state', 0)
            maxiter = safe_get_input('maxiter', None)
            alpha = safe_get_input('alpha', 1.0)
            gamma = safe_get_input('gamma', 1.0)
            init_pos = safe_get_input('init_pos', False)
            use_reduction = safe_get_input('use_reduction', None)
            n_pcs = safe_get_input('n_pcs', 50)
            n_embedding_components = safe_get_input('n_embedding_components', 40)
            distance_metric = safe_get_input('distance_metric', 'euclidean')

            # Validate parameters
            if n_neighbors >= adata.n_obs:
                ui.notification_show(f"Number of neighbors ({n_neighbors}) exceeds number of cells ({adata.n_obs})", 
                                   type="error")
                return

            # Determine reduction to use
            state = analysis_state()
            available_reductions = detect_available_reductions(adata)
            
            reduction_used = "PCA"  # Default
            use_rep = None
            embedding_components_to_use = n_embedding_components

            if use_reduction and use_reduction in adata.obsm:
                use_rep = use_reduction
                reduction_used = use_reduction.replace('X_', '').upper()
            elif 'X_pca' in adata.obsm:
                use_rep = 'X_pca'
                reduction_used = "PCA"
            elif available_reductions:
                use_rep = list(available_reductions.keys())[0]
                reduction_used = use_rep.replace('X_', '').upper()

            current_progress.set({
                'title': 'Computing UMAP Embedding',
                'message': f'Computing using {reduction_used} embeddings...'
            })

            n_components_for_neighbors = n_pcs
            use_rep_for_neighbors = None

            if use_rep:
                # Use the selected embedding with user-specified number of components
                available_components = adata.obsm[use_rep].shape[1]
                n_components_for_neighbors = min(n_pcs, available_components)
                
                # Limit the embedding to the requested number of components
                embedding_components_to_use = min(n_embedding_components, available_components)
                if n_embedding_components > available_components:
                    ui.notification_show(f"Requested {n_embedding_components} components, but only {available_components} available in {reduction_used}. Using {available_components}.", 
                                       type="warning", duration=5)
                
                # Create a subset of the embedding if needed
                if embedding_components_to_use < available_components:
                    adata.obsm[f"{use_rep}_subset"] = adata.obsm[use_rep][:, :embedding_components_to_use]
                    use_rep_for_neighbors = f"{use_rep}_subset"
                    logger.info(f"Using {embedding_components_to_use} components from {reduction_used}")
                else:
                    use_rep_for_neighbors = use_rep
                    
                n_components_for_neighbors = min(n_pcs, embedding_components_to_use)
            else:
                # Compute PCA
                current_progress.set({
                    'title': 'Computing UMAP Embedding',
                    'message': 'Computing PCA for UMAP input...'
                })
                
                # Handle NaN values
                if hasattr(adata.X, 'toarray'):
                    if np.isnan(adata.X.data).any():
                        adata.X.data = np.nan_to_num(adata.X.data)
                else:
                    if np.isnan(adata.X).any():
                        adata.X = np.nan_to_num(adata.X)

                n_comps = min(50, adata.n_vars-1, adata.n_obs-1)
                sc.tl.pca(adata, n_comps=n_comps)
                use_rep = 'X_pca'
                use_rep_for_neighbors = 'X_pca'
                reduction_used = "PCA"
                n_components_for_neighbors = min(n_pcs, n_comps)
                embedding_components_to_use = min(n_embedding_components, n_comps)

            # Remove existing UMAP and neighbors
            if 'X_umap' in adata.obsm:
                del adata.obsm['X_umap']
            if 'neighbors' in adata.uns:
                del adata.uns['neighbors']
            if 'distances' in adata.obsp:
                del adata.obsp['distances']
            if 'connectivities' in adata.obsp:
                del adata.obsp['connectivities']

            current_progress.set({
                'title': 'Computing UMAP Embedding',
                'message': 'Computing nearest neighbor graph...'
            })

            # Compute neighbors with user-specified parameters
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_components_for_neighbors, 
                           use_rep=use_rep_for_neighbors, metric=distance_metric)

            current_progress.set({
                'title': 'Computing UMAP Embedding',
                'message': 'Running UMAP optimization...'
            })

            # Prepare UMAP parameters
            umap_params = {
                'min_dist': float(min_dist),
                'spread': float(spread),
                'n_components': int(n_components),
                'random_state': int(random_state),
                'alpha': float(alpha),
                'method': 'umap'
            }

            if maxiter is not None:
                umap_params['maxiter'] = int(maxiter)

            if init_pos and 'X_umap' in adata.obsm:
                umap_params['init_pos'] = 'X_umap'

            if gamma != 1.0:
                umap_params['gamma'] = float(gamma)

            # Execute UMAP
            sc.tl.umap(adata, **umap_params)

            # Store the actual parameters used
            actual_params = {
                'n_neighbors': n_neighbors,
                'min_dist': float(min_dist),
                'spread': float(spread),
                'n_components': int(n_components),
                'random_state': int(random_state),
                'alpha': float(alpha),
                'gamma': float(gamma),
                'maxiter': int(maxiter) if maxiter is not None else None,
                'n_pcs': n_components_for_neighbors,
                'n_embedding_components': embedding_components_to_use,
                'distance_metric': distance_metric,
                'reduction_used': reduction_used,
                'init_pos': init_pos,
                'computed_by': 'shinyUMAP',
                'timestamp': str(time.time())
            }
            umap_computation_params.set(actual_params)

            # Store results
            current_session['umap_adata'] = adata
            session_data.set(current_session)

            # Update state with fresh reduction detection
            fresh_available_reductions = detect_available_reductions(adata)
            state = analysis_state()
            state['has_umap'] = True
            state['new_umap_computed'] = True
            state['has_pca'] = 'X_pca' in adata.obsm
            state['available_reductions'] = fresh_available_reductions
            analysis_state.set(state)

            # Clear progress
            current_progress.set(None)

            ui.notification_show(f"UMAP embedding computed using {reduction_used} ({n_neighbors} neighbors, min_dist={min_dist:.2f})", type="success")
            data_version.set(data_version() + 1)
            
            # Restore color settings with a small delay to ensure UI is ready
            if preserved_color["color_mode"] and preserved_color["color_mode"] != "none":
                # Schedule restoration for next event loop iteration
                async def restore_colors():
                    await asyncio.sleep(0.1)  # Small delay for UI update
                    
                    try:
                        # Update color mode
                        ui.update_radio_buttons("color_mode", selected=preserved_color["color_mode"])
                        
                        # Update specific selector
                        if preserved_color["color_mode"] == "metadata" and preserved_color["metadata_column"]:
                            if preserved_color["metadata_column"] in adata.obs.columns:
                                ui.update_select("metadata_column", selected=preserved_color["metadata_column"])
                        elif preserved_color["color_mode"] == "gene" and preserved_color["gene"]:
                            if preserved_color["gene"] in adata.var_names:
                                ui.update_selectize("gene", selected=preserved_color["gene"])
                    except Exception as e:
                        logger.warning(f"Could not restore color settings: {e}")
                
                # Run restoration
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(restore_colors())
                    loop.close()
                except:
                    # Fallback to simple restoration
                    try:
                        ui.update_radio_buttons("color_mode", selected=preserved_color["color_mode"])
                    except:
                        pass
            
            gc.collect()

        except Exception as e:
            current_progress.set(None)
            logger.error(f"UMAP computation error: {e}")
            ui.notification_show(f"UMAP computation error: {str(e)}", type="error", duration=10)
        finally:
            is_computing.set(False)

    # ENHANCED DENDROGRAM REACTIVE LOGIC - Syncs with UMAP embedding
    @reactive.Effect
    @reactive.event(input.metadata_column, ignore_none=True, ignore_init=True)
    def update_dendrogram_on_metadata_change():
        """
        Update dendrogram when metadata column changes
        """
        try:
            # Only proceed if we have the right conditions
            state = analysis_state()
            if not state.get('data_loaded', False) or not state.get('has_umap', False):
                return
                
            # Get current selection
            try:
                color_mode = input.color_mode() if hasattr(input, 'color_mode') else "none"
                meta_col = input.metadata_column() if hasattr(input, 'metadata_column') else None
            except:
                return
            
            # Only proceed if we're in metadata mode and have a valid column
            if color_mode != "metadata" or not meta_col:
                return
                
            adata = get_current_adata()
            if not adata or meta_col not in adata.obs.columns:
                return
                
            # Skip if continuous
            if is_numeric_column(adata, meta_col):
                logger.info(f"Skipping dendrogram for continuous variable: {meta_col}")
                return
            
            # Check category count (same limit as UMAP plotting)
            n_categories = adata.obs[meta_col].nunique()
            if n_categories > 25:
                logger.info(f"Skipping dendrogram for high-cardinality column: {meta_col} ({n_categories} categories > 25 limit)")
                current_dendrogram.set(None)
                dendrogram_collapsed.set(True)
                
                # Show user notification
                try:
                    ui.notification_show(
                        f"Cannot compute dendrogram for '{meta_col}': {n_categories} categories exceeds limit of 25.",
                        type="info",
                        duration=5
                    )
                except:
                    pass
                return
            
            # Check if this is actually a NEW request
            current_request = f"{meta_col}_{adata.n_obs}_{adata.n_vars}"
            last_request = last_dendrogram_request()
            if current_request == last_request:
                logger.info(f"Dendrogram request unchanged, skipping: {meta_col}")
                return
            
            # Update request tracking
            last_dendrogram_request.set(current_request)
            
            # Check if we already have this dendrogram with current UMAP params
            current_dendrogram_data = current_dendrogram()
            umap_params = umap_computation_params()
            
            if (current_dendrogram_data and 
                current_dendrogram_data.get('category') == meta_col and
                current_dendrogram_data.get('matched_umap', False)):
                
                # Check if embedding matches current UMAP
                if (umap_params and 
                    current_dendrogram_data.get('representation', '').upper() == 
                    umap_params.get('reduction_used', '').upper()):
                    logger.info(f"Dendrogram already exists and matches current UMAP for {meta_col}")
                    return
            
            # Compute new dendrogram with current UMAP parameters
            logger.info(f"Computing new dendrogram for category: {meta_col}")
            dendrogram_data = compute_category_dendrogram_cached(adata, meta_col, umap_params)
            
            if dendrogram_data:
                current_dendrogram.set(dendrogram_data)
                dendrogram_collapsed.set(False)  # Show new dendrogram
                logger.info(f"✅ Updated dendrogram for {meta_col} using {dendrogram_data.get('representation', 'unknown')}")
            else:
                current_dendrogram.set(None)
                dendrogram_collapsed.set(True)
                logger.warning(f"Failed to compute dendrogram for category: {meta_col}")
                
        except Exception as e:
            logger.error(f"Error in dendrogram metadata update: {e}")

    # NEW: Update dendrogram when UMAP parameters change
    @reactive.Effect
    @reactive.event(umap_computation_params, ignore_none=True, ignore_init=True)
    def update_dendrogram_on_umap_change():
        """
        Update dendrogram when UMAP embedding changes to keep them synchronized
        """
        try:
            # Only proceed if we have the right conditions
            state = analysis_state()
            if not state.get('data_loaded', False) or not state.get('has_umap', False):
                return
            
            # Check if we're currently showing a dendrogram
            current_dendrogram_data = current_dendrogram()
            if not current_dendrogram_data:
                return
            
            # Get current coloring mode
            try:
                color_mode = input.color_mode() if hasattr(input, 'color_mode') else "none"
                meta_col = input.metadata_column() if hasattr(input, 'metadata_column') else None
            except:
                return
            
            # Only update if we're in metadata mode
            if color_mode != "metadata" or not meta_col:
                return
            
            # Get new UMAP parameters
            umap_params = umap_computation_params()
            if not umap_params:
                return
            
            # Check if the embedding actually changed
            current_embedding = current_dendrogram_data.get('representation', '').upper()
            new_embedding = umap_params.get('reduction_used', '').upper()
            
            if current_embedding == new_embedding:
                # Same embedding, but check component count
                current_components = current_dendrogram_data.get('n_components', 0)
                new_components = umap_params.get('n_embedding_components', 0)
                
                if current_components == new_components:
                    logger.info(f"Dendrogram embedding unchanged: {current_embedding} with {current_components} components")
                    return
            
            logger.info(f"UMAP embedding changed: {current_embedding} -> {new_embedding}, updating dendrogram")
            
            # Recompute dendrogram with new UMAP parameters
            adata = get_current_adata()
            if adata and meta_col in adata.obs.columns:
                dendrogram_data = compute_category_dendrogram_cached(adata, meta_col, umap_params)
                
                if dendrogram_data:
                    current_dendrogram.set(dendrogram_data)
                    logger.info(f"✅ Updated dendrogram to match new UMAP embedding: {dendrogram_data.get('representation', 'unknown')}")
                else:
                    logger.warning(f"Failed to recompute dendrogram with new embedding")
                    
        except Exception as e:
            logger.error(f"Error updating dendrogram for UMAP change: {e}")

    # Clear dendrogram when switching away from categorical metadata
    @reactive.Effect  
    @reactive.event(input.color_mode, ignore_none=True, ignore_init=True)
    def clear_dendrogram_on_mode_change():
        """
        Clear dendrogram when switching away from metadata coloring
        """
        try:
            color_mode = input.color_mode()
            if color_mode != "metadata":
                # Clear dendrogram and collapse it
                current_dendrogram.set(None)
                dendrogram_collapsed.set(True)
                last_dendrogram_request.set(None)
                logger.info("Cleared dendrogram due to color mode change")
        except Exception as e:
            logger.warning(f"Error clearing dendrogram: {e}")

    # ENHANCED: Dendrogram UI with embedding info
    @render.ui
    def dendrogram_ui():
        """
        Enhanced dendrogram UI showing which embedding was used
        """
        try:
            state = analysis_state()
            if not state.get('data_loaded', False) or not state.get('has_umap', False):
                return None
            
            # Check if we're in metadata mode
            try:
                color_mode = input.color_mode() if hasattr(input, 'color_mode') else "none"
                if color_mode != "metadata":
                    return None
            except:
                return None
                
            # Simply show dendrogram if we have one
            dendrogram_data = current_dendrogram()
            if not dendrogram_data:
                return None
            
            # Enhanced description with embedding info
            representation = dendrogram_data.get('representation', 'expression')
            n_categories = dendrogram_data.get('n_categories', 0)
            n_components = dendrogram_data.get('n_components', 'unknown')
            matched_umap = dendrogram_data.get('matched_umap', False)
            
            if representation == 'expression':
                using_text = "gene expression patterns"
            else:
                using_text = f"{representation} embeddings ({n_components} components)"
            
            # Add synchronization status
            sync_status = "✅ matches UMAP" if matched_umap else "⚠️ may not match UMAP"
                
            return ui.div(
                ui.div(
                    ui.h5(
                        f"Hierarchical Clustering: {dendrogram_data['category']}",
                        class_="d-inline-block"
                    ),
                    ui.input_action_button(
                        "toggle_dendrogram",
                        "Hide" if not dendrogram_collapsed() else "Show",
                        class_="btn-sm btn-outline-secondary ml-3"
                    ),
                    class_="d-flex justify-content-between align-items-center mb-2"
                ),
                ui.p(
                    f"Clustering {n_categories} categories using {using_text} ({sync_status}). "
                    "Related cell types should group together if UMAP preserves biological relationships.",
                    class_="text-muted small"
                ),
                ui.div(
                    ui.output_plot("dendrogram_plot", width="100%", height="300px"),
                    style_=f"display: {'block' if not dendrogram_collapsed() else 'none'};"
                ),
                class_="mt-3"
            )
        except Exception as e:
            logger.error(f"Error rendering dendrogram UI: {e}")
            return None


    # SIMPLIFIED: Clear dendrogram when switching away from categorical metadata
    @reactive.Effect  
    @reactive.event(input.color_mode, ignore_none=True, ignore_init=True)
    def clear_dendrogram_on_mode_change():
        """
        Clear dendrogram when switching away from metadata coloring
        """
        try:
            color_mode = input.color_mode()
            if color_mode != "metadata":
                # Clear dendrogram and collapse it
                current_dendrogram.set(None)
                dendrogram_collapsed.set(True)
                last_dendrogram_request.set(None)
                logger.info("Cleared dendrogram due to color mode change")
        except Exception as e:
            logger.warning(f"Error clearing dendrogram: {e}")

    @render.ui
    def umap_parameters_display():
        """Display parameters used for current UMAP computation"""
        state = analysis_state()
        
        if not state.get('has_umap', False):
            return None
        
        stored_params = umap_computation_params()
        
        if stored_params is None:
            return ui.div(
                ui.h6("Current UMAP", class_="mb-2"),
                ui.p("Pre-existing embedding (parameters unknown)", class_="text-muted mb-0"),
                class_="alert alert-info"
            )
        
        # Define default values for comparison
        defaults = {
            'n_neighbors': 15,
            'min_dist': 0.5,
            'spread': 1.0,
            'n_components': 2,
            'random_state': 0,
            'alpha': 1.0,
            'gamma': 1.0,
            'maxiter': None,
            'init_pos': False,
            'n_pcs': 50,
            'n_embedding_components': 40,
            'distance_metric': 'euclidean'
        }
        
        # Find parameters that differ from defaults
        modified_params = []
        
        for param, default_val in defaults.items():
            if param in stored_params:
                current_val = stored_params[param]
                
                if param == 'maxiter':
                    if current_val is not None:
                        modified_params.append(f"max_iterations={current_val}")
                elif param == 'init_pos':
                    if current_val != default_val:
                        init_text = "existing_embedding" if current_val else "random"
                        modified_params.append(f"initialization={init_text}")
                elif param == 'distance_metric':
                    if current_val != default_val:
                        modified_params.append(f"distance={current_val}")
                elif param == 'n_embedding_components':
                    if current_val != default_val:
                        modified_params.append(f"embedding_components={current_val}")
                elif current_val != default_val:
                    if isinstance(current_val, float):
                        modified_params.append(f"{param}={current_val:.3g}")
                    else:
                        modified_params.append(f"{param}={current_val}")
        
        # Add reduction used if available
        if 'reduction_used' in stored_params:
            reduction = stored_params['reduction_used']
            if reduction != 'unknown':
                modified_params.append(f"input_reduction={reduction}")
        
        if modified_params:
            params_text = " | ".join(modified_params)
            return ui.div(
                ui.h6("Current Parameters", class_="mb-2"),
                ui.p(f"Modified: {params_text}", class_="mb-0 font-monospace"),
                class_="alert alert-success"
            )
        else:
            return ui.div(
                ui.h6("Current Parameters", class_="mb-2"),
                ui.p("Using default values", class_="mb-0"),
                class_="alert alert-success"
            )

    # UPDATED: Dendrogram UI render function (NO reactive state modification)
    @render.ui
    def dendrogram_ui():
        """
        Pure render function for dendrogram section 
        NO reactive state modification - only displays existing data
        """
        try:
            state = analysis_state()
            if not state.get('data_loaded', False) or not state.get('has_umap', False):
                return None
            
            # Check if we're in metadata mode
            try:
                color_mode = input.color_mode() if hasattr(input, 'color_mode') else "none"
                if color_mode != "metadata":
                    return None
            except:
                return None
                
            # Simply show dendrogram if we have one (don't compute here)
            dendrogram_data = current_dendrogram()
            if not dendrogram_data:
                return None
            
            # Simple, clear description
            representation = dendrogram_data.get('representation', 'expression')
            n_categories = dendrogram_data.get('n_categories', 0)
            
            if representation == 'expression':
                using_text = "gene expression patterns"
            else:
                using_text = f"{representation} embeddings"
                
            return ui.div(
                ui.div(
                    ui.h5(
                        f"Hierarchical Clustering: {dendrogram_data['category']}",
                        class_="d-inline-block"
                    ),
                    ui.input_action_button(
                        "toggle_dendrogram",
                        "Hide" if not dendrogram_collapsed() else "Show",
                        class_="btn-sm btn-outline-secondary ml-3"
                    ),
                    class_="d-flex justify-content-between align-items-center mb-2"
                ),
                ui.p(
                    f"Clustering {n_categories} categories based on {using_text}. "
                    "Related cell types should group together if UMAP preserves biological relationships.",
                    class_="text-muted small"
                ),
                ui.div(
                    ui.output_plot("dendrogram_plot", width="100%", height="300px"),
                    style_=f"display: {'block' if not dendrogram_collapsed() else 'none'};"
                ),
                class_="mt-3"
            )
        except Exception as e:
            logger.error(f"Error rendering dendrogram UI: {e}")
            return None

    @reactive.Effect
    @reactive.event(input.toggle_dendrogram)
    def toggle_dendrogram():
        """Toggle dendrogram visibility"""
        dendrogram_collapsed.set(not dendrogram_collapsed())

    @render.plot
    def dendrogram_plot():
        """
        Render dendrogram plot with timeout protection and enhanced error handling
        """
        try:
            dendrogram_data = current_dendrogram()
            if not dendrogram_data:
                return None
                
            # Only render if dendrogram is not collapsed
            if dendrogram_collapsed():
                return None
            
            # Validate dendrogram data
            if 'linkage' not in dendrogram_data or 'labels' not in dendrogram_data:
                logger.error("Invalid dendrogram data structure")
                return None
            
            linkage_matrix = dendrogram_data['linkage']
            labels = dendrogram_data['labels']
            
            if linkage_matrix is None or len(labels) == 0:
                logger.error("Empty linkage matrix or labels")
                return None
            
            # Check for reasonable number of categories
            if len(labels) > 25:
                logger.warning(f"Too many categories for dendrogram plotting: {len(labels)}")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.text(0.5, 0.5, f"Too many categories to display dendrogram ({len(labels)} > 25)", 
                       ha="center", va="center", fontsize=12, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                return fig
            
            # Timeout protection for dendrogram plotting
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Dendrogram plotting timeout")
            
            try:
                # Set a 15-second timeout for dendrogram plotting
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(15)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
                
                # Plot dendrogram with timeout protection
                try:
                    dendrogram(
                        linkage_matrix,
                        labels=labels,
                        ax=ax,
                        leaf_rotation=45,
                        leaf_font_size=max(8, min(12, 120 // len(labels)))  # Scale font size
                    )
                    
                    # Clear the alarm
                    signal.alarm(0)
                    
                    # Enhanced title with embedding info
                    representation = dendrogram_data.get('representation', 'unknown')
                    n_components = dendrogram_data.get('n_components', '')
                    matched_umap = dendrogram_data.get('matched_umap', False)
                    
                    if representation == 'expression':
                        rep_text = "Expression"
                    else:
                        if n_components:
                            rep_text = f"{representation.upper()} ({n_components}D)"
                        else:
                            rep_text = representation.upper()
                    
                    # Add sync indicator to title
                    sync_indicator = "✓" if matched_umap else "⚠"
                    title = f"{dendrogram_data['category']} Clustering ({rep_text}) {sync_indicator}"
                    
                    ax.set_title(title, fontsize=12, pad=15)
                    ax.set_xlabel("Cell Type/Category")
                    ax.set_ylabel("Distance")
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    return fig
                    
                except TimeoutError:
                    signal.alarm(0)  # Clear the alarm
                    logger.error("Dendrogram plotting timed out")
                    
                    # Show timeout message
                    try:
                        ui.notification_show(
                            f"Dendrogram plotting timed out for '{dendrogram_data['category']}'. "
                            "Too many categories or complex data structure.",
                            type="warning",
                            duration=5
                        )
                    except:
                        pass
                    
                    # Return timeout plot
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.text(0.5, 0.5, f"Dendrogram plotting timed out\n({len(labels)} categories)", 
                           ha="center", va="center", fontsize=12, color='orange')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    return fig
                    
            except Exception as e:
                signal.alarm(0)  # Clear the alarm if set
                logger.error(f"Error in dendrogram plotting: {e}")
                
                # Return error plot
                fig, ax = plt.subplots(figsize=(10, 3))
                error_msg = "Unable to plot dendrogram"
                if "too many" in str(e).lower():
                    error_msg += f"\n(Too many categories: {len(labels)})"
                elif "memory" in str(e).lower():
                    error_msg += "\n(Insufficient memory)"
                else:
                    error_msg += f"\n(Error: {str(e)[:50]}...)"
                    
                ax.text(0.5, 0.5, error_msg, 
                       ha="center", va="center", fontsize=11, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                return fig
                
        except Exception as e:
            logger.error(f"Critical error in dendrogram_plot: {e}")
            
            # Fallback error plot
            try:
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.text(0.5, 0.5, f"Critical dendrogram error", 
                       ha="center", va="center", fontsize=12, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                return fig
            except:
                return None


    # Dimensionality reduction info UI
    @render.ui
    def dimensionality_reduction_info():
        """Display available dimensionality reduction information"""
        data_version()  # React to data changes
        
        state = analysis_state()
        req(state['data_loaded'])
        req(not state.get('data_loading', False))

        available = state.get('available_reductions', {})
        
        logger.info(f"Rendering dim reduction info with available reductions: {list(available.keys())}")

        if available:
            reduction_details = list(available.values())
            reduction_list = ", ".join(reduction_details)
            return ui.div(
                ui.p(f"✅ Available embeddings for UMAP input: {reduction_list}", 
                     class_="text-success"),
                class_="alert alert-success"
            )
        else:
            return ui.div(
                ui.p("⚠️ No dimensionality reduction detected", 
                     class_="text-warning"),
                ui.p("PCA required for UMAP computation. Use preprocessing panel or ensure dataset includes PCA.", 
                     class_="text-muted small"),
                class_="alert alert-warning"
            )

    # Reduction selector UI
    @render.ui
    def reduction_selector_ui():
        """Dimensionality reduction selector"""
        data_version()  # React to data changes
        
        state = analysis_state()
        if not state['data_loaded'] or state.get('data_loading', False):
            return None

        available = state.get('available_reductions', {})
        
        logger.info(f"Rendering reduction selector with available reductions: {list(available.keys())}")

        if not available:
            return ui.div(
                ui.p("⚠️ No dimensionality reduction available", class_="text-warning"),
                ui.p("PCA required for UMAP computation. Use preprocessing panel or ensure dataset includes PCA.", 
                     class_="text-muted small"),
                class_="alert alert-warning"
            )

        default_selection = 'X_pca' if 'X_pca' in available else list(available.keys())[0]

        return ui.div(
            ui.span("Dimensionality reduction input:"),
            info_button("use_reduction", 
                      "Input Dimensionality Reduction",
                      "Select which dimensionality reduction to use as input for UMAP computation."),
            ui.input_select("use_reduction", "", 
                          choices=available,
                          selected=default_selection),
            class_="mb-3"
        )

    # Color selection UI
    @render.ui
    def color_selector_ui():
        """Simplified color selector without complex preservation logic"""
        data_version()  # React to data changes
        
        state = analysis_state()
        req(state['data_loaded'])
        req(not state.get('data_loading', False))

        adata = get_current_adata()
        if adata is None:
            return ui.p("Please load dataset to configure visualization options", 
                       class_="text-muted")

        metadata_cols = list(adata.obs.columns) if len(adata.obs.columns) > 0 else []

        ui_elements = []

        ui_elements.append(
            ui.input_radio_buttons(
                "color_mode",
                "Coloring scheme:",
                choices={
                    "none": "No coloring",
                    "metadata": "Cell metadata",
                    "gene": "Gene expression"
                },
                selected="none"
            )
        )

        if metadata_cols:
            ui_elements.append(ui.output_ui("metadata_selector_ui"))

        ui_elements.append(ui.output_ui("gene_selector_ui"))

        return ui.div(*ui_elements)

    @render.ui
    def metadata_selector_ui():
        """Metadata column selector"""
        try:
            color_mode = input.color_mode()
            if color_mode != "metadata":
                return None
        except:
            return None

        state = analysis_state()
        req(state['data_loaded'])

        try:
            adata = get_current_adata()
            if adata and len(adata.obs.columns) > 0:
                metadata_cols = list(adata.obs.columns)
                
                # Simple default selection
                selected_col = metadata_cols[0]
                
                # Try to get current selection if it exists and is valid
                try:
                    current_selection = input.metadata_column()
                    if current_selection and current_selection in metadata_cols:
                        selected_col = current_selection
                except:
                    pass
                
                return ui.input_select(
                    "metadata_column",
                    "Select metadata column:",
                    choices={col: col for col in metadata_cols},
                    selected=selected_col
                )
        except Exception as e:
            logger.error(f"Error creating metadata selector: {e}")
            return ui.p("Error loading metadata columns", class_="text-danger")

        return None

    @render.ui
    def gene_selector_ui():
        """Gene expression selector"""
        try:
            color_mode = input.color_mode()
            if color_mode != "gene":
                return None
        except:
            return None

        state = analysis_state()
        req(state['data_loaded'])

        try:
            adata = get_current_adata()
            if adata and adata.n_vars > 0:
                genes = list(adata.var_names)
                
                return ui.input_selectize(
                    "gene",
                    "Select gene for expression coloring:",
                    choices={gene: gene for gene in genes},
                    selected=None,  # No default selection - user must choose
                    multiple=False,
                    options={"placeholder": "Type gene name to search..."}
                )
        except Exception as e:
            logger.error(f"Error creating gene selector: {e}")
            return ui.p("Error loading gene list", class_="text-danger")

        return None

    @render.ui
    def color_map_ui():
        """Color map selector - only for gene expression"""
        state = analysis_state()
        if state.get('data_loading', False):
            return None
            
        try:
            color_mode = input.color_mode()
        except:
            return None

        # Only show color map for gene expression
        if color_mode == "gene":
            return ui.input_select("color_map", "Color palette:", 
                                 choices=['viridis', 'plasma', 'inferno', 
                                        'magma', 'cividis', 'Reds', 
                                        'Blues', 'coolwarm', 'RdBu_r'],
                                 selected='viridis')
        else:
            return None

    @render.ui
    def umap_button_ui():
        """UMAP computation button with status"""
        state = analysis_state()

        if is_computing():
            return ui.div(
                ui.tags.i(class_="fas fa-cog fa-spin text-info"),
                " Computing...",
                class_="text-info"
            )

        if not state['data_loaded']:
            return ui.p("Load dataset to begin analysis", class_="text-muted")
        elif state['has_umap'] and not state['new_umap_computed']:
            return ui.input_action_button("run_umap", 
                                        "Compute New UMAP", 
                                        class_="btn-primary btn-lg btn-block")
        else:
            return ui.input_action_button("run_umap", 
                                        "Compute UMAP", 
                                        class_="btn-success btn-lg btn-block")

    @render.ui
    def status_message():
        """Status message display"""
        state = analysis_state()
        
        if not state.get('data_loaded', False):
            return ui.div(
                ui.p("No dataset loaded. Expected format: H5AD (AnnData) with normalized expression values.", 
                   class_="text-muted"),
                class_="alert alert-info"
            )
        elif state.get('has_umap', False):
            msg = "UMAP embedding available. Assess cluster separation and biological coherence using metadata coloring. "
            if state.get('downsampled', False):
                msg += "Dataset downsampled for computational efficiency. "
            if state.get('is_test_data', False):
                msg += "Using PBMC 10k reference dataset."
            else:
                msg += "Adjust parameters if cell populations appear incorrectly merged or fragmented."
            return ui.div(
                ui.p(msg, class_="text-success"),
                class_="alert alert-success"
            )
        else:
            msg = "Dataset loaded. UMAP will create a 2D representation preserving cell-cell relationships. "
            if state.get('downsampled', False):
                msg += "Dataset downsampled for computational efficiency. "
            if state.get('is_test_data', False):
                msg += "Using PBMC 10k reference dataset. "
            msg += "Default parameters (n_neighbors=15, min_dist=0.5) work well for initial exploration."
            return ui.div(
                ui.p(msg, class_="text-warning"),
                class_="alert alert-warning"
            )


    # UPDATED: UMAP plot render function with category limits
    @render.plot
    def umap_plot():
        """
        UMAP visualization with category limit protection
        Falls back to no coloring if too many categories to prevent hanging
        """
        state = analysis_state()
        req(state.get('data_loaded', False))
        req(not state.get('data_loading', False))
        
        try:
            adata = get_current_adata()
        except Exception as e:
            logger.warning(f"Could not get data for plot: {e}")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.text(0.5, 0.5, "Loading dataset...", ha="center", va="center", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        if adata is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.text(0.5, 0.5, "Load dataset to begin analysis", ha="center", va="center", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # Check for UMAP embedding
        if not ('X_umap' in adata.obsm and adata.obsm['X_umap'] is not None):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.text(0.5, 0.5, "UMAP embedding not computed\n\nConfigure parameters and click 'Compute UMAP'", 
                   ha="center", va="center", fontsize=14, color='#666666')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("UMAP Visualization", fontsize=16, pad=20)
            return fig

        # ROBUST parameter extraction with defaults
        def safe_get_viz_input(attr_name, default_value):
            try:
                if hasattr(input, attr_name):
                    val = getattr(input, attr_name)()
                    return val if val is not None else default_value
                return default_value
            except:
                return default_value

        color_mode = safe_get_viz_input('color_mode', 'none')
        point_size = safe_get_viz_input('point_size', 30)
        show_legend = safe_get_viz_input('show_legend', True)
        color_map = safe_get_viz_input('color_map', 'viridis')

        # Category limits to prevent hanging
        MAX_CATEGORIES_FOR_PLOTTING = 20
        
        # Determine coloring scheme with robust error handling and category limits
        color_key = None
        title = "UMAP Visualization"
        needs_legend = False
        is_continuous = False
        fallback_reason = None

        if color_mode == "gene":
            try:
                gene_name = safe_get_viz_input('gene', None)
                if gene_name and gene_name in adata.var_names:
                    color_key = gene_name
                    title = f"UMAP: {gene_name} Expression"
                    needs_legend = show_legend
                    is_continuous = True
                elif gene_name:
                    # Try case-insensitive match
                    gene_match = next((g for g in adata.var_names if g.lower() == gene_name.lower()), None)
                    if gene_match:
                        color_key = gene_match
                        title = f"UMAP: {gene_match} Expression"
                        needs_legend = show_legend
                        is_continuous = True
                    else:
                        fallback_reason = f"Gene '{gene_name}' not found in dataset"
                        logger.warning(fallback_reason)
            except Exception as e:
                fallback_reason = f"Error with gene coloring: {str(e)}"
                logger.warning(fallback_reason)

        elif color_mode == "metadata":
            try:
                meta_col = safe_get_viz_input('metadata_column', None)
                if meta_col and meta_col in adata.obs.columns:
                    # Check if this is a continuous variable
                    if is_numeric_column(adata, meta_col):
                        color_key = meta_col
                        title = f"UMAP: {meta_col}"
                        needs_legend = show_legend
                        is_continuous = True
                    else:
                        # Categorical variable - check category count
                        n_unique = adata.obs[meta_col].nunique()
                        logger.info(f"Metadata column '{meta_col}' has {n_unique} unique categories")
                        
                        if n_unique <= MAX_CATEGORIES_FOR_PLOTTING:
                            color_key = meta_col
                            title = f"UMAP: {meta_col}"
                            needs_legend = show_legend
                            is_continuous = False
                        else:
                            # Too many categories - fall back to no coloring
                            fallback_reason = f"Too many categories in '{meta_col}' ({n_unique} > {MAX_CATEGORIES_FOR_PLOTTING} limit)"
                            title = f"UMAP (too many categories in {meta_col})"
                            logger.warning(fallback_reason)
                            
                            # Show user notification about the fallback
                            try:
                                ui.notification_show(
                                    f"Cannot color by '{meta_col}': {n_unique} categories exceeds limit of {MAX_CATEGORIES_FOR_PLOTTING}. "
                                    "Choose a column with fewer categories or use gene expression coloring.",
                                    type="warning",
                                    duration=8
                                )
                            except:
                                pass  # Don't let notification errors break plotting
                elif meta_col:
                    fallback_reason = f"Metadata column '{meta_col}' not found in dataset"
                    logger.warning(fallback_reason)
            except Exception as e:
                fallback_reason = f"Error with metadata coloring: {str(e)}"
                logger.warning(fallback_reason)

        # If we had to fall back due to errors, reset to no coloring
        if fallback_reason:
            color_mode = "none"
            color_key = None
            needs_legend = False
            is_continuous = False

        # Calculate figure dimensions
        plot_size = 6.0
        fig_height = plot_size
        
        if needs_legend and color_key:
            if is_continuous:
                fig_width = plot_size + 1.5
            else:
                fig_width = plot_size + 2.0
        else:
            fig_width = plot_size
        
        # Create figure
        try:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            fig.set_dpi(100)
            
            if needs_legend and color_key:
                plt.subplots_adjust(left=0.1, right=0.15, top=0.9, bottom=0.1)
            else:
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            
        except Exception as e:
            logger.error(f"Error creating figure: {e}")
            fig, ax = plt.subplots(figsize=(6, 6))

        # Simplified plotting - trust scanpy to handle categorical vs continuous correctly
        try:
            if color_key:
                try:
                    plot_params = {
                        'color': color_key, 
                        'ax': ax, 
                        'show': False, 
                        'frameon': True, 
                        'size': point_size,
                        'legend_loc': 'right margin' if needs_legend else None
                    }

                    # Only set color map for explicitly continuous data (gene expression)
                    if color_mode == "gene":
                        plot_params['cmap'] = color_map

                    # Cross-platform timeout protection using threading
                    import threading
                    
                    plot_result = [None]
                    plot_error = [None]
                    
                    def plot_with_timeout():
                        try:
                            sc.pl.umap(adata, **plot_params)
                            plot_result[0] = True
                        except Exception as e:
                            plot_error[0] = e
                    
                    # Start plotting in a separate thread
                    plot_thread = threading.Thread(target=plot_with_timeout)
                    plot_thread.daemon = True
                    plot_thread.start()
                    
                    # Wait for completion with timeout
                    plot_thread.join(timeout=30)
                    
                    if plot_thread.is_alive():
                        # Plotting timed out
                        logger.error(f"Plotting with {color_key} timed out - falling back to no coloring")
                        ui.notification_show(
                            f"Plotting with '{color_key}' took too long and was cancelled. Using no coloring instead.",
                            type="warning", 
                            duration=5
                        )
                        # Fall back to basic plot
                        sc.pl.umap(adata, ax=ax, show=False, frameon=True, size=point_size)
                        title = "UMAP Visualization (coloring timed out)"
                    elif plot_error[0]:
                        # Plotting failed
                        raise plot_error[0]
                    # else: plotting succeeded

                except Exception as e:
                    logger.warning(f"Could not color by {color_key}: {e}")
                    # Fall back to basic plot
                    sc.pl.umap(adata, ax=ax, show=False, frameon=True, size=point_size)
                    title = "UMAP Visualization"
                    
                    # Notify user if this was due to too many categories
                    if "too many colors" in str(e).lower() or "maximum" in str(e).lower():
                        try:
                            ui.notification_show(
                                f"Could not color by '{color_key}': too many categories for visualization. "
                                "Try selecting a column with fewer categories.",
                                type="warning",
                                duration=6
                            )
                        except:
                            pass
            else:
                sc.pl.umap(adata, ax=ax, show=False, frameon=True, size=point_size)

        except Exception as e:
            logger.error(f"Error plotting UMAP: {e}")
            ax.text(0.5, 0.5, f"Error generating UMAP plot:\n{str(e)}", 
                   ha="center", va="center", fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        except Exception as e:
            logger.error(f"Error plotting UMAP: {e}")
            ax.text(0.5, 0.5, f"Error generating UMAP plot:\n{str(e)}", 
                   ha="center", va="center", fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # Set title with fallback explanation if needed
        if fallback_reason and "too many categories" in fallback_reason.lower():
            ax.set_title(title, fontsize=14, pad=20, color='#d63384')  # Bootstrap warning color
        else:
            ax.set_title(title, fontsize=16, pad=20)

        try:
            plt.tight_layout(pad=1.0)
        except Exception as e:
            logger.warning(f"tight_layout failed: {e}")
        
        return fig


    # Parameter reset function
    @reactive.Effect
    @reactive.event(input.reset_umap_params)
    def reset_umap_parameters():
        """Reset UMAP parameters to default values"""
        try:
            # Core parameters
            ui.update_numeric("n_neighbors", value=15)
            ui.update_numeric("n_components", value=2)
            ui.update_numeric("min_dist", value=0.5)
            ui.update_numeric("spread", value=1.0)
            ui.update_numeric("random_state", value=0)
            ui.update_numeric("maxiter", value=None)
            ui.update_numeric("alpha", value=1.0)
            ui.update_numeric("gamma", value=1.0)
            ui.update_checkbox("init_pos", value=False)
            
            # Input configuration parameters  
            ui.update_numeric("n_pcs", value=50)
            ui.update_numeric("n_embedding_components", value=40)
            ui.update_select("distance_metric", selected="euclidean")
            
            ui.notification_show("Parameters reset to empirically validated defaults", type="info", duration=3)
            
        except Exception as e:
            logger.error(f"Error resetting parameters: {e}")
            ui.notification_show(f"Error resetting parameters: {str(e)}", type="error")

    # Download functions
    @render.download
    def download_umap():
        """Download UMAP coordinates as CSV"""
        adata = get_current_adata()
        if adata is None or "X_umap" not in adata.obsm:
            ui.notification_show("No UMAP embedding available for download", type="warning")
            raise ValueError("No UMAP embedding available")

        try:
            temp_path = temp_manager.get_temp_path(".csv")
            
            umap_df = pd.DataFrame(
                adata.obsm["X_umap"],
                index=adata.obs_names,
                columns=[f"UMAP{i+1}" for i in range(adata.obsm["X_umap"].shape[1])]
            )

            if len(adata.obs.columns) > 0:
                umap_df = pd.concat([umap_df, adata.obs], axis=1)

            umap_df.to_csv(temp_path)
            return temp_path

        except Exception as e:
            logger.error(f"Error creating UMAP download: {e}")
            ui.notification_show(f"Error creating download: {str(e)}", type="error")
            raise

    @render.download
    def download_anndata():
        """Download current AnnData object"""
        adata = get_current_adata()
        if adata is None:
            ui.notification_show("No dataset available for download", type="warning")
            raise ValueError("No dataset available")

        try:
            temp_path = temp_manager.get_temp_path(".h5ad")
            adata.write_h5ad(temp_path)
            return temp_path

        except Exception as e:
            logger.error(f"Error creating dataset download: {e}")
            ui.notification_show(f"Error creating download: {str(e)}", type="error")
            raise

    @render.download(filename="shinyumap_seurat_to_h5ad_converter.R")
    def download_seurat_converter_docs():
        """Download R script for Seurat conversion from Documentation tab"""
        try:
            temp_path = temp_manager.get_temp_path(".R")
            
            r_script = '''#!/usr/bin/env Rscript
# Seurat to H5AD Converter for shinyUMAP
# =======================================

cat("Seurat to H5AD Converter\\n")
cat("========================\\n")
cat("Compatible with shinyUMAP\\n")
cat("Uses zellconverter Bioconductor package\\n\\n")

# Package installation and management
install_if_missing <- function(pkg, bioconductor = FALSE) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(paste("Installing", pkg, "...\\n"))
    if (bioconductor) {
      if (!require("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", repos = "https://cloud.r-project.org/")
      }
      BiocManager::install(pkg, update = FALSE, ask = FALSE)
    } else {
      install.packages(pkg, repos = "https://cloud.r-project.org/")
    }
    library(pkg, character.only = TRUE)
  }
}

cat("Verifying package dependencies...\\n")
install_if_missing("Seurat")
install_if_missing("zellconverter", bioconductor = TRUE)

convert_seurat_to_h5ad <- function(seurat_path, h5ad_path, assay_name = "RNA", 
                                   optimize_memory = TRUE, verbose = TRUE) {
  
  if (verbose) cat("=== Seurat to H5AD Conversion via zellconverter ===\\n")
  
  if (!file.exists(seurat_path)) {
    stop(paste("Input file does not exist:", seurat_path))
  }
  
  if (verbose) cat(paste("Loading Seurat object from:", seurat_path, "\\n"))
  
  tryCatch({
    seurat_obj <- readRDS(seurat_path)
  }, error = function(e) {
    stop(paste("Failed to load Seurat object:", e$message))
  })
  
  seurat_version <- packageVersion("Seurat")
  if (verbose) {
    cat(paste("Seurat version:", seurat_version, "\\n"))
    cat(paste("Object dimensions:", ncol(seurat_obj), "cells × ", nrow(seurat_obj), "genes\\n"))
    cat(paste("Available assays:", paste(names(seurat_obj@assays), collapse = ", "), "\\n"))
    
    reductions <- names(seurat_obj@reductions)
    if (length(reductions) > 0) {
      cat(paste("Dimensionality reductions:", paste(reductions, collapse = ", "), "\\n"))
    }
  }
  
  # Seurat v5 compatibility
  if (seurat_version >= "5.0.0") {
    if (verbose) cat("Applying Seurat v5 compatibility measures...\\n")
    
    if (assay_name %in% names(seurat_obj@assays)) {
      assay <- seurat_obj@assays[[assay_name]]
      
      if ("layers" %in% slotNames(assay) && length(assay@layers) > 1) {
        if (verbose) cat("Consolidating v5 assay layers...\\n")
        tryCatch({
          seurat_obj <- JoinLayers(seurat_obj, assay = assay_name)
        }, error = function(e) {
          if (verbose) cat("Note: Layer consolidation not required\\n")
        })
      }
    }
  }
  
  if (!assay_name %in% names(seurat_obj@assays)) {
    available_assays <- names(seurat_obj@assays)
    stop(paste("Assay", assay_name, "not found. Available:", paste(available_assays, collapse = ", ")))
  }
  
  DefaultAssay(seurat_obj) <- assay_name
  if (verbose) cat(paste("Using assay:", assay_name, "\\n"))
  
  # Memory optimization
  if (optimize_memory) {
    if (verbose) cat("Applying memory optimization...\\n")
    
    if (length(names(seurat_obj@assays)) > 1) {
      other_assays <- setdiff(names(seurat_obj@assays), assay_name)
      for (unused_assay in other_assays) {
        if (verbose) cat(paste("  Removing unused assay:", unused_assay, "\\n"))
        seurat_obj@assays[[unused_assay]] <- NULL
      }
    }
    
    if (length(seurat_obj@commands) > 0) {
      seurat_obj@commands <- list()
    }
  }
  
  # Execute conversion
  if (verbose) cat("Executing H5AD conversion using zellconverter...\\n")
  
  conversion_start <- Sys.time()
  
  tryCatch({
    zellconverter::writeH5AD(seurat_obj, file = h5ad_path, verbose = verbose)
  }, error = function(e) {
    stop(paste("Conversion failed:", e$message))
  })
  
  conversion_time <- difftime(Sys.time(), conversion_start, units = "secs")
  if (verbose) cat(paste("Conversion completed in", round(conversion_time, 2), "seconds\\n"))
  
  if (verbose) {
    cat("\\n=== Conversion Summary ===\\n")
    cat(paste("✅ Successfully converted to:", h5ad_path, "\\n"))
    cat("Dataset ready for shinyUMAP!\\n")
  }
  
  return(h5ad_path)
}

# Command-line interface
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("Usage: Rscript converter.R <input.rds> <output.h5ad> [assay]\\n")
    quit(status = 1)
  }
  
  input_file <- args[1]
  output_file <- args[2]
  assay_name <- if (length(args) >= 3) args[3] else "RNA"
  
  if (!file.exists(input_file)) {
    cat(paste("Error: Input file not found:", input_file, "\\n"))
    quit(status = 1)
  }
  
  tryCatch({
    convert_seurat_to_h5ad(input_file, output_file, assay_name = assay_name)
    cat("\\nâœ… Conversion completed successfully!\\n")
  }, error = function(e) {
    cat(paste("âŒ Conversion failed:", e$message, "\\n"))
    quit(status = 1)
  })
}
'''
                
            with open(temp_path, 'w') as f:
                f.write(r_script)
            
            return temp_path

        except Exception as e:
            logger.error(f"Error creating Seurat converter script from docs: {e}")
            ui.notification_show(f"Error creating conversion script: {str(e)}", type="error")
            raise

    @render.download(filename="shinyumap_analysis_parameters.json")
    def download_params():
        """Download comprehensive analysis parameters"""
        state = analysis_state()
        source = data_source()

        # Helper function to safely get input values
        def safe_input(attr_name, default_value=None):
            try:
                if hasattr(input, attr_name):
                    val = getattr(input, attr_name)()
                    return val if val is not None else default_value
                return default_value
            except:
                return default_value

        # Determine reduction used
        reduction_used = "PCA"  # Default
        try:
            use_reduction = safe_input('use_reduction')
            if use_reduction:
                reduction_used = use_reduction.replace('X_', '').upper()
        except:
            pass

        # Get dataset analysis
        dataset_analysis = state.get('dataset_quality') or {}

        # Compile comprehensive parameters
        params = {
            "analysis_metadata": {
                "timestamp": str(time.time()),
                "platform": "shinyUMAP",
                "version": "1.0",
                "session_id": current_dataset_id() if current_dataset_id() else "unknown"
            },
            "dataset_information": {
                "format": "H5AD (AnnData)",
                "source_type": source.get('type', 'unknown') if source else 'unknown',
                "is_reference_dataset": state.get('is_test_data', False),
                "downsampled": state.get('downsampled', False),
                "processing_state": dataset_analysis.get('data_state', 'unknown'),
                "has_reductions": dataset_analysis.get('has_reductions', False),
                "has_hvg": dataset_analysis.get('has_hvg', False),
                "has_raw": dataset_analysis.get('has_raw', False),
                "available_reductions": list(state.get('available_reductions', {}).keys()),
                "memory_optimized": dataset_analysis.get('should_remove_raw', False),
                "dataset_size": {
                    "n_cells": get_current_adata().n_obs if get_current_adata() else 0,
                    "n_genes": get_current_adata().n_vars if get_current_adata() else 0
                }
            },
            "preprocessing_pipeline": {
                "applied": state['preprocessing_done'],
                "parameters": {
                    "min_genes_per_cell": safe_input('min_genes'),
                    "min_cells_per_gene": safe_input('min_cells'),
                    "normalization_applied": safe_input('normalize'),
                    "highly_variable_genes": safe_input('n_top_genes'),
                    "scaling_applied": state['preprocessing_done'],
                    "pca_computed": state.get('has_pca', False)
                }
            },
            "umap_configuration": {
                "embedding_computed": state['has_umap'],
                "newly_computed": state.get('new_umap_computed', False),
                "parameters_used": umap_computation_params() if umap_computation_params() else None,
                "current_ui_parameters": {
                    "n_components": safe_input('n_components', 2),
                    "n_neighbors": safe_input('n_neighbors', 15),
                    "min_dist": safe_input('min_dist', 0.5),
                    "spread": safe_input('spread', 1.0),
                    "max_iterations": safe_input('maxiter') if safe_input('maxiter') else "automatic",
                    "random_state": safe_input('random_state', 0),
                    "learning_rate": safe_input('alpha', 1.0),
                    "repulsion_strength": safe_input('gamma', 1.0),
                    "n_pcs": safe_input('n_pcs', 50),
                    "n_embedding_components": safe_input('n_embedding_components', 40),
                    "distance_metric": safe_input('distance_metric', 'euclidean'),
                    "initialization": "existing_embedding" if safe_input('init_pos', False) else "random",
                    "input_reduction": reduction_used
                }
            },
            "visualization_settings": {
                "coloring_scheme": safe_input('color_mode', 'none'),
                "selected_metadata": safe_input('metadata_column'),
                "selected_gene": safe_input('gene'),
                "point_size": safe_input('point_size', 30),
                "legend_displayed": safe_input('show_legend', True),
                "color_palette": safe_input('color_map', 'viridis')
            },
            "dataset_analysis": dataset_analysis,
            "computational_environment": {
                "backend": "scanpy",
                "umap_implementation": "scanpy.tl.umap",
                "neighbor_method": "scanpy.pp.neighbors",
                "memory_optimization": "aggressive"
            }
        }

        try:
            temp_path = temp_manager.get_temp_path(".json")
            
            with open(temp_path, 'w') as f:
                json.dump(params, f, indent=4)
            
            return temp_path

        except Exception as e:
            logger.error(f"Error creating parameters download: {e}")
            ui.notification_show(f"Error creating parameter file: {str(e)}", type="error")
            raise

# Create the application
app = App(app_ui, server)