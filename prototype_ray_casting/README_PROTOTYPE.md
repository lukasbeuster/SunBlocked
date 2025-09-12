# 🚀 POINT-BASED SHADOW RAY CASTING PROTOTYPE

⚠️ **THIS IS A PROTOTYPE** ⚠️
This directory contains experimental point-based shadow ray casting implementation.
It is separate from the main thermal drift pipeline and serves as a proof-of-concept.

## Purpose
Demonstrate 100-1000x speedup by computing shadows only at specific GPS coordinates 
instead of generating full raster grids.

## Files
- `ray_caster.py` - Core ray casting implementation
- `dsm_loader.py` - DSM/CHM tile loading utilities  
- `sun_calculator.py` - Sun position calculations
- `prototype_demo.py` - Demo script to test the approach
- `test_with_real_data.py` - Integration test with your actual DSM tiles

## Expected Performance
- **Current pipeline**: 25 hours for full city
- **This prototype**: Minutes to hours for same accuracy
- **Memory usage**: 1-5GB vs 37GB current

## Usage
```bash
cd prototype_ray_casting
python prototype_demo.py
```

## Next Steps
1. Validate against existing shadow rasters
2. Integrate with your edge/point data
3. Add temporal optimization
4. Scale to full dataset
