"""
Utility to select the best model fold based on validation metrics.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

def select_best_model(
    model_dir: Union[str, Path],
    metric: str = "best_val_f1",
    prefer_symlink: bool = True,
) -> Path:
    """
    Select best model from available folds in the directory.
    
    Args:
        model_dir: Directory containing models (e.g., models/advanced/ETHUSDT/15m)
        metric: Metric to use for selection (best_val_f1, test_f1, etc.)
        prefer_symlink: If True and model.pt exists, use it directly (legacy support)
        
    Returns:
        Path to best model file
        
    Raises:
        FileNotFoundError: If no suitable model is found
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    # 1. Check for legacy symlink/file if requested
    if prefer_symlink:
        legacy_model = model_dir / "model.pt"
        if legacy_model.exists():
            logger.debug(f"Using existing model.pt in {model_dir}")
            return legacy_model

    # 2. Try to load training results
    results_file = model_dir / "production_training_results.json"
    best_fold = None
    
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text())
            results = data.get("results", [])
            
            if results:
                # Sort by metric descending (assuming higher is better for f1/accuracy)
                # Handle nested metrics if needed (e.g. test_metrics.macro_f1)
                def get_score(entry):
                    val = entry.get(metric, 0)
                    if isinstance(val, dict):
                        return 0 # Complex metric not supported yet
                    return float(val)
                
                sorted_results = sorted(results, key=get_score, reverse=True)
                best_entry = sorted_results[0]
                best_fold = best_entry.get("fold")
                
                logger.info(
                    f"Selected fold {best_fold} for {model_dir.name} "
                    f"with {metric}={best_entry.get(metric):.4f}"
                )
        except Exception as e:
            logger.warning(f"Failed to parse training results: {e}")
    
    # 3. If we identified a best fold, try to find it
    if best_fold is not None:
        candidates = [
            model_dir / f"best_model_fold{best_fold}.pt",
            model_dir / f"model_fold{best_fold}.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
                
    # 4. Fallback: Find any .pt file, preferring 'best_model' prefix
    pt_files = list(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt model files found in {model_dir}")
        
    # Prioritize 'best_model' files
    best_models = [f for f in pt_files if "best_model" in f.name]
    if best_models:
        # Sort by modification time, newest first
        best_models.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        selected = best_models[0]
        logger.info(f"Fallback selection (newest best_model): {selected.name}")
        return selected
        
    # Fallback to any .pt file (newest first)
    pt_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    selected = pt_files[0]
    logger.info(f"Fallback selection (newest .pt): {selected.name}")
    return selected
