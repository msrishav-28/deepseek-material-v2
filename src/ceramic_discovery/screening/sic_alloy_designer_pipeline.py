"""
SiC Alloy Designer Pipeline Orchestrator.

This module orchestrates the complete SiC alloy design workflow including:
- Multi-source data loading (Materials Project, JARVIS, NIST, Literature)
- Feature engineering with composition and structure descriptors
- ML model training (Random Forest, Gradient Boosting)
- Application-specific material ranking
- Experimental protocol generation
- Comprehensive reporting

The pipeline implements graceful degradation - if any stage fails, it continues
with available data and logs the failure.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import pandas as pd
import numpy as np

from ceramic_discovery.dft.jarvis_client import JarvisClient
from ceramic_discovery.dft.nist_client import NISTClient
from ceramic_discovery.dft.literature_database import LiteratureDatabase
from ceramic_discovery.dft.data_combiner import DataCombiner
from ceramic_discovery.ml.composition_descriptors import CompositionDescriptorCalculator
from ceramic_discovery.ml.structure_descriptors import StructureDescriptorCalculator
from ceramic_discovery.ml.feature_engineering import FeatureEngineeringPipeline
from ceramic_discovery.ml.model_trainer import ModelTrainer
from ceramic_discovery.screening.application_ranker import ApplicationRanker
from ceramic_discovery.validation.experimental_planner import ExperimentalPlanner
from ceramic_discovery.reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class SiCAlloyDesignerPipeline:
    """
    Complete pipeline orchestrator for SiC alloy design.
    
    Orchestrates all stages of the workflow:
    1. Data loading from multiple sources
    2. Data combination and deduplication
    3. Feature engineering
    4. ML model training
    5. Application-specific ranking
    6. Experimental protocol generation
    7. Report generation
    
    Features:
    - Stage-by-stage logging and progress tracking
    - Error handling with graceful degradation
    - Checkpoint saving for intermediate results
    - CSV output for all major data products
    """
    
    def __init__(
        self,
        output_dir: str = "results/sic_alloy_designer",
        checkpoint_dir: Optional[str] = None,
        enable_checkpoints: bool = True
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            output_dir: Directory for output files
            checkpoint_dir: Directory for checkpoint files (defaults to output_dir/checkpoints)
            enable_checkpoints: Whether to save checkpoints
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.enable_checkpoints = enable_checkpoints
        
        # Pipeline state
        self.stage_results: Dict[str, Any] = {}
        self.stage_errors: Dict[str, str] = {}
        self.stage_timings: Dict[str, float] = {}
        
        # Data products
        self.combined_data: Optional[pd.DataFrame] = None
        self.engineered_features: Optional[pd.DataFrame] = None
        self.ml_results: Optional[Dict[str, Any]] = None
        self.application_rankings: Optional[pd.DataFrame] = None
        self.experimental_protocols: Optional[List[Dict[str, Any]]] = None
        
        logger.info(f"Initialized SiC Alloy Designer Pipeline")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def run_full_pipeline(
        self,
        jarvis_file: Optional[str] = None,
        nist_data_dir: Optional[str] = None,
        mp_data: Optional[List] = None,
        metal_elements: Optional[set] = None,
        target_property: str = "hardness",
        top_n_candidates: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete pipeline from data loading to reporting.
        
        Args:
            jarvis_file: Path to JARVIS JSON file
            nist_data_dir: Directory containing NIST data files
            mp_data: Materials Project data (optional)
            metal_elements: Set of metal elements to filter for carbides
            target_property: Target property for ML training
            top_n_candidates: Number of top candidates for experimental planning
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING SIC ALLOY DESIGNER PIPELINE")
        logger.info("=" * 80)
        
        # Default metal elements for carbides
        if metal_elements is None:
            metal_elements = {"Si", "Ti", "Zr", "Hf", "Ta", "W", "Mo", "Nb", "V", "Cr"}
        
        # Stage 1: Load data from multiple sources
        self._run_stage(
            "data_loading",
            self._load_data,
            jarvis_file=jarvis_file,
            nist_data_dir=nist_data_dir,
            mp_data=mp_data,
            metal_elements=metal_elements
        )
        
        # Check if we have any data
        if not self.stage_results.get("data_loading"):
            logger.error("No data loaded. Cannot continue pipeline.")
            return self._generate_pipeline_summary(start_time, success=False)
        
        # Stage 2: Combine data sources
        self._run_stage(
            "data_combination",
            self._combine_data_sources
        )
        
        if self.combined_data is None or len(self.combined_data) == 0:
            logger.error("No combined data available. Cannot continue pipeline.")
            return self._generate_pipeline_summary(start_time, success=False)
        
        # Save combined data
        self._save_checkpoint("combined_data", self.combined_data)
        self._save_csv(self.combined_data, "combined_materials_data.csv")
        
        # Stage 3: Feature engineering
        self._run_stage(
            "feature_engineering",
            self._engineer_features
        )
        
        if self.engineered_features is not None:
            self._save_checkpoint("engineered_features", self.engineered_features)
            self._save_csv(self.engineered_features, "engineered_features.csv")
        
        # Stage 4: ML model training
        self._run_stage(
            "ml_training",
            self._train_ml_models,
            target_property=target_property
        )
        
        if self.ml_results is not None:
            self._save_checkpoint("ml_results", self.ml_results)
        
        # Stage 5: Application ranking
        self._run_stage(
            "application_ranking",
            self._rank_materials
        )
        
        if self.application_rankings is not None:
            self._save_checkpoint("application_rankings", self.application_rankings)
            self._save_csv(self.application_rankings, "application_rankings.csv")
        
        # Stage 6: Experimental protocol generation
        self._run_stage(
            "experimental_planning",
            self._generate_experimental_protocols,
            top_n=top_n_candidates
        )
        
        if self.experimental_protocols is not None:
            self._save_checkpoint("experimental_protocols", self.experimental_protocols)
        
        # Stage 7: Report generation
        self._run_stage(
            "report_generation",
            self._generate_reports
        )
        
        # Generate pipeline summary
        summary = self._generate_pipeline_summary(start_time, success=True)
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 80)
        
        return summary
    
    def _run_stage(self, stage_name: str, stage_func, **kwargs) -> bool:
        """
        Run a pipeline stage with error handling and logging.
        
        Args:
            stage_name: Name of the stage
            stage_func: Function to execute
            **kwargs: Arguments to pass to stage function
        
        Returns:
            True if stage succeeded, False otherwise
        """
        logger.info("-" * 80)
        logger.info(f"STAGE: {stage_name.upper()}")
        logger.info("-" * 80)
        
        stage_start = datetime.now()
        
        try:
            result = stage_func(**kwargs)
            self.stage_results[stage_name] = result
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.stage_timings[stage_name] = stage_duration
            
            logger.info(f"✓ Stage '{stage_name}' completed successfully in {stage_duration:.2f}s")
            return True
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.stage_timings[stage_name] = stage_duration
            self.stage_errors[stage_name] = str(e)
            
            logger.error(f"✗ Stage '{stage_name}' failed after {stage_duration:.2f}s: {e}")
            logger.exception("Full traceback:")
            
            return False
    
    def _load_data(
        self,
        jarvis_file: Optional[str],
        nist_data_dir: Optional[str],
        mp_data: Optional[List],
        metal_elements: set
    ) -> Dict[str, Any]:
        """Load data from all available sources."""
        data_sources = {}
        
        # Load JARVIS data
        if jarvis_file:
            try:
                logger.info(f"Loading JARVIS data from {jarvis_file}")
                jarvis_client = JarvisClient(jarvis_file)
                jarvis_carbides = jarvis_client.load_carbides(metal_elements)
                data_sources["jarvis"] = jarvis_carbides
                logger.info(f"Loaded {len(jarvis_carbides)} carbides from JARVIS")
            except Exception as e:
                logger.warning(f"Failed to load JARVIS data: {e}")
                data_sources["jarvis"] = []
        else:
            logger.info("No JARVIS file provided, skipping JARVIS data")
            data_sources["jarvis"] = []
        
        # Load NIST data
        if nist_data_dir:
            try:
                logger.info(f"Loading NIST data from {nist_data_dir}")
                nist_client = NISTClient()
                # NIST data loading would go here
                # For now, placeholder
                data_sources["nist"] = {}
                logger.info("NIST data loading not yet implemented")
            except Exception as e:
                logger.warning(f"Failed to load NIST data: {e}")
                data_sources["nist"] = {}
        else:
            logger.info("No NIST directory provided, skipping NIST data")
            data_sources["nist"] = {}
        
        # Load literature data
        try:
            logger.info("Loading literature database")
            lit_db = LiteratureDatabase()
            lit_data = lit_db.get_all_data()
            data_sources["literature"] = lit_data
            logger.info(f"Loaded data for {len(lit_data)} materials from literature")
        except Exception as e:
            logger.warning(f"Failed to load literature data: {e}")
            data_sources["literature"] = {}
        
        # Materials Project data
        if mp_data:
            data_sources["mp"] = mp_data
            logger.info(f"Using {len(mp_data)} materials from Materials Project")
        else:
            logger.info("No Materials Project data provided")
            data_sources["mp"] = []
        
        return data_sources
    
    def _combine_data_sources(self) -> pd.DataFrame:
        """Combine data from all sources with deduplication."""
        data_sources = self.stage_results.get("data_loading", {})
        
        if not data_sources:
            raise ValueError("No data sources available for combination")
        
        logger.info("Combining data from multiple sources")
        
        combiner = DataCombiner()
        
        combined_df = combiner.combine_sources(
            mp_data=data_sources.get("mp"),
            jarvis_data=data_sources.get("jarvis"),
            nist_data=data_sources.get("nist"),
            literature_data=data_sources.get("literature")
        )
        
        self.combined_data = combined_df
        
        stats = combiner.get_statistics()
        logger.info(f"Combined data statistics: {stats}")
        logger.info(f"Total materials after combination: {len(combined_df)}")
        
        return combined_df
    
    def _engineer_features(self) -> pd.DataFrame:
        """Engineer features from combined data."""
        if self.combined_data is None:
            raise ValueError("No combined data available for feature engineering")
        
        logger.info("Engineering features from combined data")
        
        # Initialize feature engineering pipeline with descriptors
        pipeline = FeatureEngineeringPipeline(
            scaling_method='standard',
            handle_missing='drop',
            include_composition_descriptors=True,
            include_structure_descriptors=True
        )
        
        # Select numeric features for ML
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target-like columns and IDs
        exclude_cols = ['material_id', 'rank', 'score']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            logger.warning("No numeric features found in combined data")
            return pd.DataFrame()
        
        X = self.combined_data[feature_cols].copy()
        
        # Get formulas for composition descriptors
        formulas = self.combined_data.get('formula', pd.Series([None] * len(self.combined_data)))
        
        # Fit and transform
        try:
            X_engineered = pipeline.fit_transform(X, formulas=formulas)
            self.engineered_features = X_engineered
            
            logger.info(f"Engineered {len(X_engineered.columns)} features from {len(feature_cols)} original features")
            logger.info(f"Feature engineering complete: {len(X_engineered)} samples")
            
            return X_engineered
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            # Return original features as fallback
            self.engineered_features = X
            return X
    
    def _train_ml_models(self, target_property: str) -> Dict[str, Any]:
        """Train ML models on engineered features."""
        if self.engineered_features is None or len(self.engineered_features) == 0:
            logger.warning("No engineered features available, skipping ML training")
            return {}
        
        if self.combined_data is None:
            raise ValueError("No combined data available for ML training")
        
        logger.info(f"Training ML models to predict {target_property}")
        
        # Check if target property exists
        if target_property not in self.combined_data.columns:
            logger.warning(f"Target property '{target_property}' not found in data")
            logger.info(f"Available properties: {self.combined_data.columns.tolist()}")
            return {}
        
        # Prepare training data
        y = self.combined_data[target_property]
        
        # Align indices
        common_idx = self.engineered_features.index.intersection(y.index)
        X_train = self.engineered_features.loc[common_idx]
        y_train = y.loc[common_idx]
        
        # Remove NaN targets
        valid_mask = y_train.notna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        if len(X_train) < 10:
            logger.warning(f"Insufficient training data: {len(X_train)} samples")
            return {}
        
        logger.info(f"Training with {len(X_train)} samples and {len(X_train.columns)} features")
        
        # Train models
        results = {}
        
        try:
            # Random Forest
            logger.info("Training Random Forest model")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            rf_model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
            rf_model.fit(X_tr, y_tr)
            
            y_pred = rf_model.predict(X_te)
            
            rf_results = {
                "model_type": "Random Forest",
                "r2_score": r2_score(y_te, y_pred),
                "mae": mean_absolute_error(y_te, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
                "n_features": len(X_train.columns),
                "n_samples": len(X_train)
            }
            
            results["random_forest"] = rf_results
            logger.info(f"Random Forest R²: {rf_results['r2_score']:.4f}, MAE: {rf_results['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
        
        try:
            # Gradient Boosting
            logger.info("Training Gradient Boosting model")
            from sklearn.ensemble import GradientBoostingRegressor
            
            gb_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
            gb_model.fit(X_tr, y_tr)
            
            y_pred_gb = gb_model.predict(X_te)
            
            gb_results = {
                "model_type": "Gradient Boosting",
                "r2_score": r2_score(y_te, y_pred_gb),
                "mae": mean_absolute_error(y_te, y_pred_gb),
                "rmse": np.sqrt(mean_squared_error(y_te, y_pred_gb)),
                "n_features": len(X_train.columns),
                "n_samples": len(X_train)
            }
            
            results["gradient_boosting"] = gb_results
            logger.info(f"Gradient Boosting R²: {gb_results['r2_score']:.4f}, MAE: {gb_results['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
        
        self.ml_results = {"models": results} if results else {}
        
        return self.ml_results
    
    def _rank_materials(self) -> pd.DataFrame:
        """Rank materials for all applications."""
        if self.combined_data is None or len(self.combined_data) == 0:
            logger.warning("No combined data available for ranking")
            return pd.DataFrame()
        
        logger.info("Ranking materials for all applications")
        
        # Convert DataFrame to list of dictionaries
        materials = self.combined_data.to_dict('records')
        
        # Initialize ranker
        ranker = ApplicationRanker()
        
        # Rank for all applications
        rankings_df = ranker.rank_for_all_applications(materials)
        
        self.application_rankings = rankings_df
        
        logger.info(f"Generated rankings for {len(rankings_df)} material-application pairs")
        
        # Log top candidates for each application
        for app in ranker.list_applications():
            app_rankings = rankings_df[rankings_df['application'] == app]
            if len(app_rankings) > 0:
                top_material = app_rankings.nlargest(1, 'overall_score').iloc[0]
                logger.info(f"  {app}: Top candidate is {top_material['formula']} "
                          f"(score: {top_material['overall_score']:.3f})")
        
        return rankings_df
    
    def _generate_experimental_protocols(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Generate experimental protocols for top candidates."""
        if self.application_rankings is None or len(self.application_rankings) == 0:
            logger.warning("No rankings available for experimental planning")
            return []
        
        logger.info(f"Generating experimental protocols for top {top_n} candidates")
        
        # Get top candidates across all applications
        top_candidates = self.application_rankings.nlargest(top_n, 'overall_score')
        
        planner = ExperimentalPlanner()
        protocols = []
        
        for _, candidate in top_candidates.iterrows():
            formula = candidate.get('formula', 'Unknown')
            
            # Extract properties for protocol generation
            properties = {
                'melting_point': candidate.get('melting_point', candidate.get('property_melting_point', 2500)),
                'density': candidate.get('density', candidate.get('property_density', 3.0)),
                'hardness': candidate.get('hardness', candidate.get('property_hardness', 25.0))
            }
            
            # Target properties for characterization
            target_props = ['hardness', 'thermal_conductivity', 'fracture_toughness']
            
            try:
                protocol = planner.generate_experimental_protocol(
                    formula=formula,
                    properties=properties,
                    target_properties=target_props
                )
                
                # Convert dataclass objects to dicts for JSON serialization
                protocol_dict = {
                    'formula': protocol['formula'],
                    'synthesis_methods': [
                        {
                            'method': m.method,
                            'temperature_K': m.temperature_K,
                            'pressure_MPa': m.pressure_MPa,
                            'time_hours': m.time_hours,
                            'difficulty': m.difficulty,
                            'cost_factor': m.cost_factor
                        }
                        for m in protocol['synthesis_methods']
                    ],
                    'characterization_plan': [
                        {
                            'technique': t.technique,
                            'purpose': t.purpose,
                            'estimated_time_hours': t.estimated_time_hours,
                            'estimated_cost_dollars': t.estimated_cost_dollars
                        }
                        for t in protocol['characterization_plan']
                    ],
                    'resource_estimate': {
                        'timeline_months': protocol['resource_estimate'].timeline_months,
                        'estimated_cost_k_dollars': protocol['resource_estimate'].estimated_cost_k_dollars,
                        'required_equipment': protocol['resource_estimate'].required_equipment,
                        'required_expertise': protocol['resource_estimate'].required_expertise
                    },
                    'is_valid': protocol['is_valid'],
                    'completeness_score': protocol['completeness_score']
                }
                
                protocols.append(protocol_dict)
                
                logger.info(f"  Generated protocol for {formula}: "
                          f"{protocol['resource_estimate'].timeline_months:.1f} months, "
                          f"${protocol['resource_estimate'].estimated_cost_k_dollars:.1f}k")
                
            except Exception as e:
                logger.warning(f"Failed to generate protocol for {formula}: {e}")
        
        self.experimental_protocols = protocols
        
        logger.info(f"Generated {len(protocols)} experimental protocols")
        
        return protocols
    
    def _generate_reports(self) -> Dict[str, str]:
        """Generate comprehensive reports."""
        logger.info("Generating comprehensive reports")
        
        # Generate summary report
        report_gen = ReportGenerator(
            title="SiC Alloy Designer Analysis Report",
            authors=["SiC Alloy Designer Pipeline"]
        )
        
        summary_report = report_gen.generate_summary_report(
            data=self.combined_data,
            ml_results=self.ml_results,
            application_rankings=self.application_rankings,
            top_n=10,
            title="SiC Alloy Designer Analysis Summary"
        )
        
        # Save summary report
        report_path = self.output_dir / "analysis_summary_report.txt"
        with open(report_path, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Saved summary report to {report_path}")
        
        # Also print to console
        print("\n" + summary_report)
        
        return {"summary_report": str(report_path)}
    
    def _save_checkpoint(self, name: str, data: Any) -> None:
        """Save checkpoint data."""
        if not self.enable_checkpoints:
            return
        
        checkpoint_path = self.checkpoint_dir / f"{name}_checkpoint.json"
        
        try:
            if isinstance(data, pd.DataFrame):
                # Save DataFrame as JSON
                data.to_json(checkpoint_path, orient='records', indent=2)
            elif isinstance(data, dict):
                # Save dict as JSON
                with open(checkpoint_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif isinstance(data, list):
                # Save list as JSON
                with open(checkpoint_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                logger.warning(f"Cannot save checkpoint for type {type(data)}")
                return
            
            logger.debug(f"Saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint {name}: {e}")
    
    def _save_csv(self, data: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV."""
        csv_path = self.output_dir / filename
        
        try:
            data.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save CSV {filename}: {e}")
    
    def _generate_pipeline_summary(self, start_time: datetime, success: bool) -> Dict[str, Any]:
        """Generate pipeline execution summary."""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        summary = {
            "success": success,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": total_duration,
            "stages_completed": len(self.stage_results),
            "stages_failed": len(self.stage_errors),
            "stage_timings": self.stage_timings,
            "stage_errors": self.stage_errors,
            "output_directory": str(self.output_dir),
            "data_products": {
                "combined_data_rows": len(self.combined_data) if self.combined_data is not None else 0,
                "engineered_features_cols": len(self.engineered_features.columns) if self.engineered_features is not None else 0,
                "ml_models_trained": len(self.ml_results.get("models", {})) if self.ml_results else 0,
                "application_rankings_rows": len(self.application_rankings) if self.application_rankings is not None else 0,
                "experimental_protocols": len(self.experimental_protocols) if self.experimental_protocols else 0
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline summary saved to {summary_path}")
        logger.info(f"Total execution time: {total_duration:.2f}s")
        
        return summary
