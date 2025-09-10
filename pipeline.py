import click
import json
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime

# This makes sure Python can find your 'src' directory
sys.path.append(str(Path(__file__).parent / 'src'))

# --- Import the main "engine" functions from your src modules ---
from solar import check_coverage, download_data
from segmentation import run_segmentation
from raster import raster_processing_main
from processing import run_shade_processing, process_dataset_only, run_shade_simulations_only, extract_and_merge_shade_values_only, reconstruct_tile_grouped_days, process_full_dataset_combined


# --- JSON Encoder for Numpy Types ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# --- Helper Functions for State Management ---

def load_config(config_path):
    """Loads the YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_run_info(output_dir: Path):
    """Loads the run_info.json handoff file."""
    run_info_path = output_dir / 'run_info.json'
    if not run_info_path.exists():
        return {} # Return empty dict if it doesn't exist yet
    with open(run_info_path, 'r') as f:
        return json.load(f)

def save_run_info(output_dir: Path, data: dict, fresh_start: bool = False):
    """
    Saves data to the run_info.json handoff file.
    If fresh_start is True, it overwrites the file completely.
    """
    run_info_path = output_dir / 'run_info.json'

    # If it's a fresh start, begin with an empty dictionary.
    # Otherwise, load the existing data to update it.
    if fresh_start:
        existing_data = {}
    else:
        existing_data = load_run_info(output_dir)

    existing_data.update(data)
    with open(run_info_path, 'w') as f:
        json.dump(existing_data, f, indent=4, cls=NumpyEncoder)

# --- Run Management Utilities ---
from typing import List

def list_existing_runs(output_dir: Path) -> List[str]:
    """Return available run IDs (osmid) by scanning step2_solar_data subfolders."""
    base = Path(output_dir) / 'step2_solar_data'
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def get_current_run_info(cfg):
    out = Path(cfg['output_dir'])
    return load_run_info(out)


# --- Flexible Dataset Loader ---
def load_dataset_flexibly(config):
    """Loads a dataset (CSV, Parquet, Pickle, GeoJSON, etc.) into a GeoDataFrame."""
    dataset_path = config['dataset_path']
    lon_col = config['columns']['longitude']
    lat_col = config['columns']['latitude']

    # Determine CRS settings
    input_crs = config.get('input_crs', 'EPSG:4326')

    # Load data into GeoDataFrame regardless of format
    suffix = Path(dataset_path).suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    elif suffix in ('.pkl', '.pickle'):
        df = pd.read_pickle(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    elif suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    else:
        # GeoJSON, GPKG, shapefiles, etc.
        gdf = gpd.read_file(dataset_path)

    return gdf


# --- The Main CLI Group ---

@click.group()
def cli():
    """
    A command-line tool for the 'Throwing Shade' simulation pipeline.
    Run steps in order: check -> download -> segment -> process-rasters -> process-shade
    Or use 'run-all' to execute the entire pipeline.
\nUtility commands:\n  current-run     Show the selected run stored in run_info.json\n  list-runs       List available osmid folders found under output_dir\n  set-run <id>    Point the pipeline to a previous run ID\n  new-run         Clear run_info.json to start a fresh run
    """
    pass


# --- Run Management Commands ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def current_run(config):
    """Show the currently selected run (from run_info.json)."""
    cfg = load_config(config)
    info = get_current_run_info(cfg)
    if not info:
        click.secho('No current run set. run_info.json is empty or missing.', fg='yellow')
        return
    click.echo(json.dumps(info, indent=2))


@cli.command(name='list-runs')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def list_runs_cmd(config):
    """List all discovered runs (osmid directories) available under output_dir."""
    cfg = load_config(config)
    runs = list_existing_runs(Path(cfg['output_dir']))
    if not runs:
        click.secho('No runs found under step2_solar_data/.', fg='yellow')
        return
    click.echo("Available runs (osmid):")
    for r in runs:
        click.echo(f"  - {r}")


@cli.command(name='set-run')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
@click.argument('osmid')
def set_run_cmd(config, osmid):
    """Point the pipeline to an existing run ID (osmid)."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    # Validate that this osmid exists (at least step2 data present)
    step2_dir = output_dir / 'step2_solar_data' / osmid
    if not step2_dir.exists():
        click.secho(f"Run '{osmid}' not found at {step2_dir}. Use 'list-runs' to see options.", fg='red')
        return
    save_run_info(output_dir, {'osmid': osmid}, fresh_start=False)
    click.secho(f"Current run set to '{osmid}'.", fg='green')


@cli.command(name='new-run')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def new_run_cmd(config):
    """Start a fresh run (clears run_info.json). Next 'check' will populate it."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    save_run_info(output_dir, {}, fresh_start=True)
    click.secho('Cleared run_info.json. Run \"check\" to begin a new run.', fg='green')

# --- STEP 1: Interactive Coverage Check ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
@click.option('--min-points', default=1, type=int, help='Override min_points_per_tile from the config file.')
@click.option('--yes', is_flag=True, help='Skip interactive confirmation (for automation).')
def check(config, min_points, yes):
    """STEP 1: Check required Solar API tiles without downloading."""
    cfg = load_config(config)
    if min_points:
        cfg['solar_api']['min_points_per_tile'] = min_points

    click.echo("--- Running Step 1: Check Solar Coverage ---")
    tile_count, preview_path = check_coverage(cfg)

    # Save the path to the preview file for the next step
    output_dir = Path(cfg['output_dir'])

    # By setting fresh_start=True, we signal the beginning of a new run.
    save_run_info(output_dir, {'preview_path': str(preview_path)}, fresh_start=True)

    click.secho(f"\n✅ Found {tile_count} tiles to download.", fg='green')
    click.echo("A preview map has been saved to:")
    click.secho(f"   {preview_path}", fg='cyan')
    click.echo("\n=> ACTION: Open this GeoJSON file in a viewer (like vscode-geojson or QGIS) to inspect the coverage.")

    if not yes:
        if not click.confirm("\nIs this coverage acceptable to proceed?"):
            click.echo("❌ Operation cancelled. Please re-run 'check' with a different '--min-points' value.")
            sys.exit(0) # Exit gracefully

    click.echo("Confirmation received. You can now run the 'download' step.")

# --- STEP 2: Download Data ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def download(config):
    """STEP 2: Download Solar API data and generate a run ID (osmid)."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    run_info = load_run_info(output_dir)

    preview_path = run_info.get('preview_path')
    if not preview_path:
        click.secho("Error: 'preview_path' not found. Please run the 'check' step first.", fg='red')
        return

    click.echo("--- Running Step 2: Downloading Solar API Data ---")
    osmid = download_data(cfg, preview_path)

    # Save the generated osmid to our handoff file
    save_run_info(output_dir, {'osmid': osmid})

    click.secho(f"\n✅ Download complete. Run ID '{osmid}' saved to {output_dir / 'run_info.json'}.", fg='green')

# --- STEP 3: Tree Segmentation ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def segment(config):
    """STEP 3: Run tree segmentation on the downloaded RGB tiles."""
    cfg = load_config(config)
    run_info = load_run_info(Path(cfg['output_dir']))
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the 'download' step first.", fg='red')
        return

    click.echo(f"--- Running Step 3: Segmenting Trees for Run ID: {osmid} ---")
    run_segmentation(cfg, osmid)
    click.secho("\n✅ Tree segmentation complete.", fg='green')

# --- STEP 4: Raster Processing ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def process_rasters(config):
    """STEP 4: Process raw DSMs into analysis-ready DSMs."""
    cfg = load_config(config)
    run_info = load_run_info(Path(cfg['output_dir']))
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the 'download' step first.", fg='red')
        return

    click.echo(f"--- Running Step 4: Processing Raster Files for Run ID: {osmid} ---")
    raster_processing_main(cfg, osmid)
    click.secho("\n✅ Raster processing complete.", fg='green')

# --- STEP 5: Final Shade Processing ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def process_shade(config):
    """STEP 5: Run the final shade analysis and generate results."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    run_info = load_run_info(output_dir)
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the previous steps first.", fg='red')
        return

    click.echo(f"--- Running Step 5: Final Shade Processing for Run ID: {osmid} ---")

    # Load dataset flexibly based on file format
    dataset = load_dataset_flexibly(cfg)
    timestamp_col = cfg['columns']['timestamp']
    dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], errors='coerce')
    # Important: keep these as naive LOCAL timestamps so they match the Local Solar Time (LST) times in your shade filenames.
    
    n_invalid = dataset[timestamp_col].isna().sum()
    print(f"{n_invalid} rows failed to parse timestamps and became NaT")

    # Drop the bad rows before analysis
    dataset = dataset.dropna(subset=[timestamp_col])

    all_year_results = []

    for year_str, year_config in cfg['year_configs'].items():
        year = int(year_str)
        click.echo(f"-> Processing data for year {year}...")
        year_data = dataset[dataset[timestamp_col].dt.year == year].copy()

        if year_data.empty:
            click.secho(f"  No data found for year {year}, skipping.", fg='yellow')
            continue
        # TODO: Check, does this for some reason add a empty result to all_year_result when more years are present in config?
        # TODO: Add warning to add DST for regions. Some regions also don't have this at all - how to deal with that?
        # Call the engine function for this year's data
        single_year_result = run_shade_processing(cfg, osmid, year, year_data)
        # TODO: Dump to disk to get out of memory - concat after the fact. 
        all_year_results.append(single_year_result)

    # Combine and save the final result
    if all_year_results:
        final_dataset = pd.concat(all_year_results, ignore_index=True)
        final_dataset = gpd.GeoDataFrame(final_dataset, geometry='geometry')

        # Add a placeholder for your final output path in config if needed
        final_output_dir = output_dir / f"step6_final_result/{osmid}"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / "shaded_dataset.geojson"

        final_dataset.to_file(final_output_path, driver="GeoJSON")
        click.secho(f"\n✅ Pipeline complete! Final output saved to: {final_output_path}", fg='green')
    else:
        click.secho("\n❌ No data was processed. No output file created.", fg='red')

# --- NEW SPLIT PIPELINE STEPS ---

# --- STEP 5a: Dataset Processing and Binning ---

@cli.command(name="process-dataset")
@click.option("--config", default="config.yaml", type=click.Path(exists=True), help="Path to the configuration file.")
@click.option("--year", type=int, help="Specific year to process (optional)")
def process_dataset_step(config, year):
    """STEP 5a: Process and bin the dataset for shade simulation."""
    cfg = load_config(config)
    output_dir = Path(cfg["output_dir"])
    run_info = load_run_info(output_dir)
    osmid = run_info.get("osmid")
    if not osmid:
        click.secho("Error: osmid not found. Please run the previous steps first.", fg="red")
        return

    click.echo(f"--- Running Step 5a: Dataset Processing for Run ID: {osmid} ---")

    # Load dataset flexibly based on file format
    dataset = load_dataset_flexibly(cfg)
    timestamp_col = cfg["columns"]["timestamp"]
    dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], errors="coerce")
    
    n_invalid = dataset[timestamp_col].isna().sum()
    print(f"{n_invalid} rows failed to parse timestamps and became NaT")
    dataset = dataset.dropna(subset=[timestamp_col])

    # Process each year
    years_to_process = [year] if year else [int(y) for y in cfg["year_configs"].keys()]
    
    for proc_year in years_to_process:
        click.echo(f"-> Processing data for year {proc_year}...")
        year_data = dataset[dataset[timestamp_col].dt.year == proc_year].copy()
        
        if year_data.empty:
            click.secho(f"  No data found for year {proc_year}, skipping.", fg="yellow")
            continue
            
        try:
            dataset_gdf, tile_grouped_days, original_dataset, bin_output_path = process_dataset_only(
                year_data, proc_year, osmid, cfg
            )
            
            # Save processing metadata to run_info
            save_run_info(output_dir, {
                f"binned_dataset_{proc_year}": str(bin_output_path),
                f"dataset_processed_{proc_year}": True
            })
            
        except Exception as e:
            click.secho(f"❌ Error processing year {proc_year}: {e}", fg="red")
            continue
    
    click.secho("\n✅ Dataset processing complete!", fg="green")

# --- STEP 6: Shade Simulation ---

@cli.command(name="simulate-shade")
@click.option("--config", default="config.yaml", type=click.Path(exists=True), help="Path to the configuration file.")
@click.option("--year", type=int, help="Specific year to simulate (optional)")
def simulate_shade_step(config, year):
    """STEP 6: Run computationally intensive shade simulations."""
    cfg = load_config(config)
    output_dir = Path(cfg["output_dir"])
    run_info = load_run_info(output_dir)
    osmid = run_info.get("osmid")
    if not osmid:
        click.secho("Error: osmid not found. Please run the previous steps first.", fg="red")
        return

    click.echo(f"--- Running Step 6: Shade Simulations for Run ID: {osmid} ---")

    # Determine years to process
    years_to_process = [year] if year else [int(y) for y in cfg["year_configs"].keys()]
    
    for proc_year in years_to_process:
        # Check if dataset was processed for this year
        if not run_info.get(f"dataset_processed_{proc_year}"):
            click.secho(f"Warning: Dataset not processed for year {proc_year}. Run process-dataset first.", fg="yellow")
            continue
            
        # Load the binned dataset
        binned_path = run_info.get(f"binned_dataset_{proc_year}")
        if not binned_path or not Path(binned_path).exists():
            click.secho(f"Error: Binned dataset not found for year {proc_year}", fg="red")
            continue
            
        click.echo(f"-> Loading binned dataset for year {proc_year}...")
        dataset_gdf = gpd.read_file(binned_path)
        
        # Reconstruct tile_grouped_days from the binned dataset
        # This requires recreating the grouping structure
        click.echo(f"-> Reconstructing tile groupings for simulations...")
        tile_grouped_days = reconstruct_tile_grouped_days(dataset_gdf)
        
        try:
            run_shade_simulations_only(tile_grouped_days, dataset_gdf, osmid, proc_year, cfg)
            
            # Save simulation completion status
            save_run_info(output_dir, {
                f"simulations_complete_{proc_year}": True
            })
            
        except Exception as e:
            click.secho(f"❌ Error simulating shade for year {proc_year}: {e}", fg="red")
            continue
    
    click.secho("\n✅ Shade simulations complete!", fg="green")

# --- STEP 7: Extract and Merge Shade Values ---

@cli.command(name="extract-shade")
@click.option("--config", default="config.yaml", type=click.Path(exists=True), help="Path to the configuration file.")
def extract_shade_step(config):
    """STEP 7: Extract shade values from rasters and merge with dataset."""
    cfg = load_config(config)
    output_dir = Path(cfg["output_dir"])
    run_info = load_run_info(output_dir)
    osmid = run_info.get("osmid")
    if not osmid:
        click.secho("Error: osmid not found. Please run the previous steps first.", fg="red")
        return

    click.echo(f"--- Running Step 7: Shade Extraction for Run ID: {osmid} ---")

    all_year_results = []
    
    for year_str in cfg["year_configs"].keys():
        proc_year = int(year_str)
        
        # Check if simulations were completed for this year
        if not run_info.get(f"simulations_complete_{proc_year}"):
            click.secho(f"Warning: Simulations not complete for year {proc_year}. Run simulate-shade first.", fg="yellow")
            continue
            
        # Load the binned dataset and original dataset
        binned_path = run_info.get(f"binned_dataset_{proc_year}")
        if not binned_path or not Path(binned_path).exists():
            click.secho(f"Error: Binned dataset not found for year {proc_year}", fg="red")
            continue
            
        click.echo(f"-> Processing shade extraction for year {proc_year}...")
        dataset_gdf = gpd.read_file(binned_path)
        
        # For this step, we need the original dataset too - reconstruct from binned data
        # or load separately if available
        original_dataset = dataset_gdf  # Simplified for now
        
        try:
            dataset_final = extract_and_merge_shade_values_only(
                dataset_gdf, osmid, cfg, original_dataset
            )
            all_year_results.append(dataset_final)
            
        except Exception as e:
            click.secho(f"❌ Error extracting shade for year {proc_year}: {e}", fg="red")
            continue
    
    # Combine and save final result
    if all_year_results:
        final_dataset = pd.concat(all_year_results, ignore_index=True)
        final_dataset = gpd.GeoDataFrame(final_dataset, geometry="geometry")
        
        final_output_dir = output_dir / f"step6_final_result/{osmid}"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / "shaded_dataset.geojson"
        
        final_dataset.to_file(final_output_path, driver="GeoJSON")
        click.secho(f"\n✅ Shade extraction complete! Final output: {final_output_path}", fg="green")
    else:
        click.secho("\n❌ No data was processed. No output file created.", fg="red")




# =============================================================================
# SPLIT PIPELINE STEPS
# =============================================================================

# --- STEP 5: Dataset Processing and Binning ---

@cli.command(name='process-dataset')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def process_dataset_step(config):
    """STEP 5: Process and bin the dataset for shade simulation."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    run_info = load_run_info(output_dir)
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the previous steps first.", fg='red')
        return

    click.echo(f"--- Running Step 5: Dataset Processing for Run ID: {osmid} ---")

    # Load dataset flexibly based on file format
    dataset = load_dataset_flexibly(cfg)
    
    try:
        combined_dataset_gdf, combined_tile_grouped_days, combined_original_dataset, bin_output_path = process_full_dataset_combined(
            dataset, osmid, cfg
        )
        
        if combined_dataset_gdf is None:
            click.secho("❌ No processable data found", fg='red')
            return
        
        # Save processing metadata to run_info
        save_run_info(output_dir, {
            'binned_dataset_combined': str(bin_output_path),
            'dataset_processed': True,
            'processed_years': list(combined_dataset_gdf['time'].dt.year.unique()) if 'time' in combined_dataset_gdf.columns else ['unknown']
        })
        
        click.secho(f"\n✅ Dataset processing complete! Output: {bin_output_path}", fg='green')
        
    except Exception as e:
        click.secho(f"❌ Error processing dataset: {e}", fg='red')

# --- STEP 6: Shade Simulation ---

@cli.command(name='simulate-shade')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def simulate_shade_step(config):
    """STEP 6: Run computationally intensive shade simulations."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    run_info = load_run_info(output_dir)
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the previous steps first.", fg='red')
        return

    click.echo(f"--- Running Step 6: Shade Simulations for Run ID: {osmid} ---")

    # Check if dataset was processed
    if not run_info.get('dataset_processed'):
        click.secho("Warning: Dataset not processed. Run process-dataset first.", fg='yellow')
        return
        
    # Load the combined binned dataset
    binned_path = run_info.get('binned_dataset_combined')
    if not binned_path or not Path(binned_path).exists():
        click.secho("Error: Combined binned dataset not found", fg='red')
        return
        
    click.echo(f"-> Loading combined binned dataset...")
    dataset_gdf = gpd.read_file(binned_path)
    
    # Reconstruct tile groupings for simulations
    click.echo(f"-> Reconstructing tile groupings...")
    tile_grouped_days = reconstruct_tile_grouped_days(dataset_gdf)
    
    try:
        # Run simulations for all years present in the data
        processed_years = run_info.get('processed_years', [2024])  # Default to 2024
        for year in processed_years:
            click.echo(f"-> Running simulations for year {year}...")
            run_shade_simulations_only(tile_grouped_days, dataset_gdf, osmid, year, cfg)
        
        # Save simulation completion status
        save_run_info(output_dir, {
            'simulations_complete': True
        })
        
        click.secho("\n✅ Shade simulations complete!", fg='green')
        
    except Exception as e:
        click.secho(f"❌ Error during simulations: {e}", fg='red')

# --- STEP 7: Extract and Merge Shade Values ---

@cli.command(name='extract-shade')
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def extract_shade_step(config):
    """STEP 7: Extract shade values from rasters and merge with dataset."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    run_info = load_run_info(output_dir)
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the previous steps first.", fg='red')
        return

    click.echo(f"--- Running Step 7: Shade Extraction for Run ID: {osmid} ---")

    # Check if simulations were completed
    if not run_info.get('simulations_complete'):
        click.secho("Warning: Simulations not complete. Run simulate-shade first.", fg='yellow')
        return
        
    # Load the combined binned dataset
    binned_path = run_info.get('binned_dataset_combined')
    if not binned_path or not Path(binned_path).exists():
        click.secho("Error: Combined binned dataset not found", fg='red')
        return
        
    click.echo(f"-> Loading combined dataset for extraction...")
    dataset_gdf = gpd.read_file(binned_path)
    original_dataset = dataset_gdf.copy()  # Use the same dataset as original for aggregation
    
    try:
        final_result = extract_and_merge_shade_values_only(dataset_gdf, osmid, cfg, original_dataset)
        
        # Save final result
        final_output_dir = output_dir / f"step6_final_result/{osmid}"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / "shaded_dataset_split.geojson"
        
        final_result.to_file(final_output_path, driver="GeoJSON")
        
        # Update run_info
        save_run_info(output_dir, {
            'final_shaded_dataset': str(final_output_path),
            'extraction_complete': True
        })
        
        click.secho(f"\n✅ Shade extraction complete! Final output: {final_output_path}", fg='green')
        
        # Show summary stats
        shade_cols = [col for col in final_result.columns if 'shade' in col.lower()]
        click.echo(f"   - Final dataset: {len(final_result):,} rows")
        click.echo(f"   - Shade columns: {len(shade_cols)}")
        
    except Exception as e:
        click.secho(f"❌ Error during extraction: {e}", fg='red')

# --- Convenience Command to Run All Steps ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
@click.option('--min-points', type=int, help='Set min_points_per_tile for the entire run.')
@click.pass_context
def run_all(ctx, config, min_points):
    """Runs the entire pipeline sequentially."""
    click.secho("--- Running Full Pipeline ---", bold=True, fg='magenta')

    # Use ctx.invoke to call other click commands from this one
    ctx.invoke(check, config=config, min_points=min_points, yes=True)
    ctx.invoke(download, config=config)
    ctx.invoke(segment, config=config)
    ctx.invoke(process_rasters, config=config)
    ctx.invoke(process_shade, config=config)

    click.secho("\n🎉 All pipeline steps completed successfully! 🎉", bold=True, fg='magenta')


if __name__ == '__main__':
    cli()
