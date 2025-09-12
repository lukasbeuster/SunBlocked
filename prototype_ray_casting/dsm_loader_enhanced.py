"""
DSM/CHM Tile Loader for Point-Based Shadow Casting
================================================

Utilities to efficiently load and query DSM/CHM tiles for specific points.
Designed to work with your existing tiled structure.
"""

import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


@dataclass
class TileInfo:
    """Information about a DSM/CHM tile"""
    tile_id: str
    building_dsm_path: Path
    canopy_dsm_path: Path
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    transform: rasterio.Affine
    crs: rasterio.CRS


class DSMTileManager:
    """Manages loading and querying of DSM/CHM tiles"""
    
    def __init__(self, dsm_tiles_dir: Union[str, Path]):
        """
        Initialize tile manager
        
        Args:
            dsm_tiles_dir: Directory containing DSM tile files
        """
        self.dsm_tiles_dir = Path(dsm_tiles_dir)
        self.tiles_info: Dict[str, TileInfo] = {}
        self._loaded_tiles: Dict[str, Dict[str, np.ndarray]] = {}
        
        self._index_tiles()
    
    def _index_tiles(self):
        """Build spatial index of available tiles"""
        print("🔍 Indexing DSM tiles...")
        
        # Find all building DSM files
        building_dsm_files = list(self.dsm_tiles_dir.rglob("*building_dsm.tif"))
        
        for building_path in building_dsm_files:
            try:
                # Extract tile ID from filename
                tile_id = self._extract_tile_id(building_path.name)
                
                # Find corresponding canopy DSM
                canopy_path = building_path.parent / building_path.name.replace("building_dsm", "canopy_dsm")
                
                if not canopy_path.exists():
                    print(f"⚠️ Missing canopy DSM for {tile_id}, skipping")
                    continue
                
                # Get raster info
                with rasterio.open(building_path) as src:
                    bounds = src.bounds
                    transform = src.transform
                    crs = src.crs
                
                self.tiles_info[tile_id] = TileInfo(
                    tile_id=tile_id,
                    building_dsm_path=building_path,
                    canopy_dsm_path=canopy_path,
                    bounds=bounds,
                    transform=transform,
                    crs=crs
                )
                
            except Exception as e:
                print(f"⚠️ Error indexing {building_path}: {e}")
        
        print(f"✅ Indexed {len(self.tiles_info)} tile pairs")
    
    def _extract_tile_id(self, filename: str) -> str:
        """Extract tile ID from filename"""
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.startswith('p') and i + 1 < len(parts) and parts[i + 1].isdigit():
                return f"{part}_{parts[i + 1]}"
        return filename.split('_')[0]  # Fallback
    
    def find_tile_for_point(self, point: Point) -> Optional[str]:
        """Find which tile contains the given point"""
        x, y = point.x, point.y
        
        for tile_id, tile_info in self.tiles_info.items():
            minx, miny, maxx, maxy = tile_info.bounds
            if minx <= x <= maxx and miny <= y <= maxy:
                return tile_id
        
        return None
    
    def load_tile(self, tile_id: str) -> Dict[str, np.ndarray]:
        """Load building and canopy DSM data for a tile"""
        if tile_id in self._loaded_tiles:
            return self._loaded_tiles[tile_id]
        
        if tile_id not in self.tiles_info:
            raise ValueError(f"Tile {tile_id} not found")
        
        tile_info = self.tiles_info[tile_id]
        
        # Load building DSM
        with rasterio.open(tile_info.building_dsm_path) as src:
            building_data = src.read(1).astype(np.float32)
        
        # Load canopy DSM
        with rasterio.open(tile_info.canopy_dsm_path) as src:
            canopy_data = src.read(1).astype(np.float32)
        
        tile_data = {
            'building': building_data,
            'canopy': canopy_data,
            'transform': tile_info.transform,
            'bounds': tile_info.bounds
        }
        
        # Cache the loaded data
        self._loaded_tiles[tile_id] = tile_data
        
        return tile_data
    
    def sample_dsm_at_point(self, point: Point, tile_id: Optional[str] = None) -> Dict[str, float]:
        """Sample DSM height values at a specific point"""
        if tile_id is None:
            tile_id = self.find_tile_for_point(point)
            if tile_id is None:
                return {'building': 0.0, 'canopy': 0.0}
        
        tile_data = self.load_tile(tile_id)
        transform = tile_data['transform']
        
        # Convert world coordinates to raster coordinates
        row, col = rasterio.transform.rowcol(transform, point.x, point.y)
        
        # Check bounds
        building_array = tile_data['building']
        if 0 <= row < building_array.shape[0] and 0 <= col < building_array.shape[1]:
            building_height = float(building_array[row, col])
            canopy_height = float(tile_data['canopy'][row, col])
            
            # Handle nodata values
            if np.isnan(building_height) or building_height < 0:
                building_height = 0.0
            if np.isnan(canopy_height) or canopy_height < 0:
                canopy_height = 0.0
                
            return {
                'building': building_height,
                'canopy': canopy_height,
                'combined': max(building_height, canopy_height)
            }
        
        return {'building': 0.0, 'canopy': 0.0, 'combined': 0.0}


if __name__ == "__main__":
    print("🏗️ Testing DSM tile loading...")
    print("Note: Adjust dsm_dir path for your system")
