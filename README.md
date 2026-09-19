# giogesserCV
GeoGuessr: Hierarchical Multi-Scale Geolocation Prediction
A deep learning and spatial clustering pipeline built for the GeoGuessr: Geolocation Prediction Hackathon. The system frames coordinate regression (latitude,longitude) as a multi-task learning problem combining spatial cell classification, global 3D vector regression, and hierarchical geographic partitioning.

Key Features
Hierarchical Spatial Target Engineering:

Country/Semantic Geo-cells: Spatial joins via GeoJSON borders combined with MiniBatchKMeans (up to 768 adaptive sub-cells) for country-aware partitioning.

S2 Spherical Partitions: Dual-scale spatial discretization into Coarse (24 cells) and Fine (96 cells) geographic clusters on the unit sphere (x,y,z).

Unit Sphere Projections: Standardizes latitude/longitude into 3D Cartesian coordinates to eliminate spherical boundary discontinuity.

Vision Backbone:

Feature extraction via SigLIP 2 (google/siglip-so400m-patch14-384).

Test-Time Augmentation (TTA) via horizontal flips during offline embedding caching.

Multi-Head Spatial Modeling:

Joint loss optimization across global classification cells and continuous unit-sphere coordinate offsets.

Project Structure
Plaintext
├── data/
│   ├── train.csv              # Training metadata and coordinates
│   ├── test.csv               # Test image metadata
│   └── borders.geojson        # World boundary polygons (optional spatial join)
├── src/
│   ├── spatial_cells.py       # KMeans & GeoJSON partition generation
│   ├── dataset.py             # Image loading, PyTorch Dataset, and TTA transforms
│   ├── extract_embeddings.py  # SigLIP 2 offline caching script
│   ├── model.py               # Multi-head prediction architecture
│   └── train.py               # Multi-task training loop & validation
├── siglip2_cache/             # Offline cached visual embeddings
└── README.md
Workflow Overview
1. Spatial Discretization
Rather than regressing raw GPS coordinates directly, targets are split into multi-scale clusters:

Semantic Clusters: Groups data points by administrative boundaries and clusters local coordinate dense zones using MiniBatchKMeans.

S2 Multi-Scale Partitions: Maps points to unit sphere vectors:

x=cos(lat)⋅cos(lon),y=cos(lat)⋅sin(lon),z=sin(lat)
Clustered into NUM_S2_COARSE = 24 and NUM_S2_FINE = 96 regions.

2. Feature Extraction & TTA
Images are resized to 384×384 and normalized.

Offline embeddings are computed with both original and horizontally flipped views to enrich representation stability before head training.

3. Training & Inference
The backbone outputs pass through multi-task classification heads (coarse, fine, and semantic cells) alongside continuous coordinate regression offsets.

Setup & Execution
Requirements
Bash
pip install torch torchvision timm geopandas shapely scikit-learn numpy pandas pillow
Running on Kaggle
Attach the competition dataset (GeoGuessr: Geolocation Prediction Hackathon).

Run spatial cell generation:

Python
NUM_SEMANTIC_CLASSES, centroids_sem = create_semantic_geocells(df_all_train, max_total_cells=768)
Cache embeddings to /kaggle/working/siglip2_cache/ using GPU acceleration (cuda:0).

Train downstream prediction heads on the cached feature matrices.
