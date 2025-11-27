"""Dopant screening engine with stability filtering and multi-objective ranking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    import warnings
    warnings.warn("Redis not available. Install with: pip install redis")

from ..dft.stability_analyzer import StabilityAnalyzer, StabilityResult
from ..ml.model_trainer import TrainedModel
from ..ceramics.ceramic_system import CeramicSystem
from .application_ranker import ApplicationRanker, RankedMaterial


logger = logging.getLogger(__name__)


class RankingCriterion(Enum):
    """Criteria for ranking materials."""
    
    STABILITY = "stability"
    PERFORMANCE = "performance"
    COST = "cost"
    COMBINED = "combined"


@dataclass
class MaterialCandidate:
    """Material candidate for screening."""
    
    material_id: str
    formula: str
    base_ceramic: str
    dopant_element: Optional[str] = None
    dopant_concentration: Optional[float] = None
    
    # DFT properties
    energy_above_hull: Optional[float] = None
    formation_energy: Optional[float] = None
    
    # Predicted properties
    predicted_v50: Optional[float] = None
    predicted_hardness: Optional[float] = None
    predicted_fracture_toughness: Optional[float] = None
    
    # Uncertainty estimates
    v50_uncertainty: Optional[float] = None
    
    # Stability analysis
    stability_result: Optional[StabilityResult] = None
    
    # Cost estimate (relative scale)
    cost_score: Optional[float] = None
    
    # Ranking scores
    stability_score: float = 0.0
    performance_score: float = 0.0
    cost_efficiency_score: float = 0.0
    combined_score: float = 0.0
    
    # Application-specific scores
    application_scores: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'material_id': self.material_id,
            'formula': self.formula,
            'base_ceramic': self.base_ceramic,
            'dopant_element': self.dopant_element,
            'dopant_concentration': self.dopant_concentration,
            'energy_above_hull': self.energy_above_hull,
            'formation_energy': self.formation_energy,
            'predicted_v50': self.predicted_v50,
            'predicted_hardness': self.predicted_hardness,
            'predicted_fracture_toughness': self.predicted_fracture_toughness,
            'v50_uncertainty': self.v50_uncertainty,
            'is_viable': self.stability_result.is_viable if self.stability_result else None,
            'cost_score': self.cost_score,
            'stability_score': self.stability_score,
            'performance_score': self.performance_score,
            'cost_efficiency_score': self.cost_efficiency_score,
            'combined_score': self.combined_score,
            'application_scores': self.application_scores,
        }


@dataclass
class ScreeningConfig:
    """Configuration for screening engine."""
    
    # Stability filtering
    stability_threshold: float = 0.1  # eV/atom
    require_viable: bool = True
    
    # Performance thresholds
    min_v50: Optional[float] = None
    min_hardness: Optional[float] = None
    min_fracture_toughness: Optional[float] = None
    
    # Ranking weights
    stability_weight: float = 0.4
    performance_weight: float = 0.4
    cost_weight: float = 0.2
    
    # Application-based ranking
    enable_application_ranking: bool = False
    selected_applications: Optional[List[str]] = None
    application_ranking_weight: float = 0.0  # Weight for application scores in combined ranking
    
    # Batch processing
    batch_size: int = 100
    
    # Caching
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.enable_application_ranking:
            # When application ranking is enabled, weights should sum to 1.0 including application weight
            total_weight = (
                self.stability_weight + 
                self.performance_weight + 
                self.cost_weight + 
                self.application_ranking_weight
            )
            if not np.isclose(total_weight, 1.0):
                raise ValueError(
                    f"Ranking weights (including application_ranking_weight) must sum to 1.0, got {total_weight}"
                )
        else:
            # Without application ranking, original weights should sum to 1.0
            total_weight = self.stability_weight + self.performance_weight + self.cost_weight
            if not np.isclose(total_weight, 1.0):
                raise ValueError(f"Ranking weights must sum to 1.0, got {total_weight}")


@dataclass
class ScreeningResults:
    """Results from screening workflow."""
    
    screening_id: str
    total_candidates: int
    viable_candidates: int
    ranked_candidates: List[MaterialCandidate]
    config: ScreeningConfig
    
    # Statistics
    avg_stability_score: float = 0.0
    avg_performance_score: float = 0.0
    avg_cost_score: float = 0.0
    
    # Application-specific rankings
    application_rankings: Optional[Dict[str, List[RankedMaterial]]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([c.to_dict() for c in self.ranked_candidates])
    
    def save(self, path: Path) -> None:
        """Save results to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        data = {
            'screening_id': self.screening_id,
            'total_candidates': self.total_candidates,
            'viable_candidates': self.viable_candidates,
            'config': {
                'stability_threshold': self.config.stability_threshold,
                'stability_weight': self.config.stability_weight,
                'performance_weight': self.config.performance_weight,
                'cost_weight': self.config.cost_weight,
            },
            'statistics': {
                'avg_stability_score': self.avg_stability_score,
                'avg_performance_score': self.avg_performance_score,
                'avg_cost_score': self.avg_cost_score,
            },
            'candidates': [c.to_dict() for c in self.ranked_candidates],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Screening results saved to {path}")


class ScreeningEngine:
    """
    High-throughput screening engine for dopant candidates.
    
    Features:
    - Thermodynamic stability filtering (ΔE_hull ≤ 0.1 eV/atom)
    - Multi-objective ranking (stability, performance, cost)
    - Batch processing for large-scale screening
    - Result caching with Redis
    """
    
    def __init__(
        self,
        stability_analyzer: StabilityAnalyzer,
        ml_model: Optional[TrainedModel] = None,
        config: Optional[ScreeningConfig] = None,
        redis_client: Optional[Any] = None,
        application_ranker: Optional[ApplicationRanker] = None
    ):
        """
        Initialize screening engine.
        
        Args:
            stability_analyzer: Analyzer for thermodynamic stability
            ml_model: Trained ML model for property prediction
            config: Screening configuration
            redis_client: Redis client for caching
            application_ranker: Application-specific ranker (optional)
        """
        self.stability_analyzer = stability_analyzer
        self.ml_model = ml_model
        self.config = config or ScreeningConfig()
        self.config.validate()
        
        self.application_ranker = application_ranker
        if self.application_ranker is None and self.config.enable_application_ranking:
            # Create default application ranker if needed
            self.application_ranker = ApplicationRanker()
            logger.info("Created default ApplicationRanker")
        
        self.redis_client = redis_client
        if self.redis_client is None and REDIS_AVAILABLE and self.config.use_cache:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
                self.redis_client = None
    
    def screen_candidates(
        self,
        candidates: List[MaterialCandidate],
        screening_id: str = "screening"
    ) -> ScreeningResults:
        """
        Screen material candidates with stability filtering and ranking.
        
        Args:
            candidates: List of material candidates
            screening_id: Unique identifier for screening run
        
        Returns:
            Screening results with ranked candidates
        """
        logger.info(f"Starting screening {screening_id} with {len(candidates)} candidates")
        
        # Step 1: Filter by stability
        viable_candidates = self._filter_by_stability(candidates)
        logger.info(f"Viable candidates after stability filtering: {len(viable_candidates)}")
        
        if not viable_candidates:
            logger.warning("No viable candidates found after stability filtering")
            return ScreeningResults(
                screening_id=screening_id,
                total_candidates=len(candidates),
                viable_candidates=0,
                ranked_candidates=[],
                config=self.config
            )
        
        # Step 2: Predict properties (if ML model available)
        if self.ml_model:
            viable_candidates = self._predict_properties(viable_candidates)
        
        # Step 3: Calculate ranking scores
        viable_candidates = self._calculate_scores(viable_candidates)
        
        # Step 4: Application-based ranking (if enabled)
        application_rankings = None
        if self.config.enable_application_ranking and self.application_ranker:
            viable_candidates, application_rankings = self._apply_application_ranking(viable_candidates)
        
        # Step 5: Rank candidates
        ranked_candidates = self._rank_candidates(viable_candidates)
        
        # Step 6: Calculate statistics
        results = ScreeningResults(
            screening_id=screening_id,
            total_candidates=len(candidates),
            viable_candidates=len(viable_candidates),
            ranked_candidates=ranked_candidates,
            config=self.config,
            application_rankings=application_rankings
        )
        
        results.avg_stability_score = np.mean([c.stability_score for c in ranked_candidates])
        results.avg_performance_score = np.mean([c.performance_score for c in ranked_candidates])
        results.avg_cost_score = np.mean([c.cost_efficiency_score for c in ranked_candidates])
        
        logger.info(f"Screening {screening_id} completed. Top candidate: {ranked_candidates[0].formula}")
        
        return results

    def _filter_by_stability(self, candidates: List[MaterialCandidate]) -> List[MaterialCandidate]:
        """
        Filter candidates by thermodynamic stability.
        
        Args:
            candidates: List of material candidates
        
        Returns:
            List of viable candidates
        """
        viable = []
        
        for candidate in candidates:
            # Check cache first
            cached_result = self._get_cached_stability(candidate.material_id)
            
            if cached_result:
                candidate.stability_result = cached_result
            elif candidate.energy_above_hull is not None and candidate.formation_energy is not None:
                # Analyze stability
                stability_result = self.stability_analyzer.analyze_stability(
                    material_id=candidate.material_id,
                    formula=candidate.formula,
                    energy_above_hull=candidate.energy_above_hull,
                    formation_energy_per_atom=candidate.formation_energy
                )
                candidate.stability_result = stability_result
                
                # Cache result
                self._cache_stability(candidate.material_id, stability_result)
            else:
                logger.warning(f"Missing stability data for {candidate.material_id}")
                continue
            
            # Filter by viability
            if self.config.require_viable and not candidate.stability_result.is_viable:
                continue
            
            viable.append(candidate)
        
        return viable
    
    def _predict_properties(self, candidates: List[MaterialCandidate]) -> List[MaterialCandidate]:
        """
        Predict material properties using ML model.
        
        Args:
            candidates: List of material candidates
        
        Returns:
            Candidates with predicted properties
        """
        if self.ml_model is None:
            logger.warning("No ML model available for property prediction")
            return candidates
        
        logger.info(f"Predicting properties for {len(candidates)} candidates")
        
        # Process in batches
        for i in range(0, len(candidates), self.config.batch_size):
            batch = candidates[i:i + self.config.batch_size]
            
            for candidate in batch:
                # Check cache
                cached_prediction = self._get_cached_prediction(candidate.material_id)
                
                if cached_prediction:
                    candidate.predicted_v50 = cached_prediction['v50']
                    candidate.v50_uncertainty = cached_prediction['uncertainty']
                else:
                    # Prepare features (placeholder - would need actual feature extraction)
                    # In production, this would extract features from the material
                    features = self._extract_features(candidate)
                    
                    if features is not None:
                        # Predict
                        prediction = self.ml_model.model.predict([features])[0]
                        candidate.predicted_v50 = prediction
                        
                        # Estimate uncertainty (simplified)
                        candidate.v50_uncertainty = prediction * 0.1  # 10% uncertainty
                        
                        # Cache prediction
                        self._cache_prediction(
                            candidate.material_id,
                            {'v50': prediction, 'uncertainty': candidate.v50_uncertainty}
                        )
        
        return candidates
    
    def _extract_features(self, candidate: MaterialCandidate) -> Optional[np.ndarray]:
        """
        Extract features for ML prediction.
        
        This is a placeholder. In production, would extract actual features
        from material properties.
        
        Args:
            candidate: Material candidate
        
        Returns:
            Feature array or None if features cannot be extracted
        """
        # Placeholder implementation
        # In production, would extract features like:
        # - Hardness, fracture toughness, density
        # - Thermal conductivity
        # - Electronic properties
        # etc.
        
        if candidate.predicted_hardness and candidate.predicted_fracture_toughness:
            return np.array([
                candidate.predicted_hardness,
                candidate.predicted_fracture_toughness,
                candidate.dopant_concentration or 0.0,
            ])
        
        return None
    
    def _apply_application_ranking(
        self,
        candidates: List[MaterialCandidate]
    ) -> Tuple[List[MaterialCandidate], Dict[str, List[RankedMaterial]]]:
        """
        Apply application-specific ranking to candidates.
        
        Args:
            candidates: List of material candidates
        
        Returns:
            Tuple of (candidates with application scores, application rankings dict)
        """
        if not self.application_ranker:
            logger.warning("Application ranker not available")
            return candidates, {}
        
        # Convert candidates to material dictionaries for ranking
        materials = []
        for candidate in candidates:
            material_dict = {
                'material_id': candidate.material_id,
                'formula': candidate.formula,
                'hardness': candidate.predicted_hardness,
                'thermal_conductivity': None,  # Would need to be added to MaterialCandidate
                'melting_point': None,
                'formation_energy': candidate.formation_energy,
                'bulk_modulus': None,
                'band_gap': None,
                'density': None,
            }
            materials.append(material_dict)
        
        # Determine which applications to rank for
        applications = self.config.selected_applications
        if applications is None:
            applications = self.application_ranker.list_applications()
        
        # Rank for each application
        application_rankings = {}
        for app in applications:
            try:
                ranked = self.application_ranker.rank_materials(materials, app)
                application_rankings[app] = ranked
                
                # Update candidates with application scores
                for i, candidate in enumerate(candidates):
                    if candidate.application_scores is None:
                        candidate.application_scores = {}
                    
                    # Find matching ranked material
                    for ranked_material in ranked:
                        if ranked_material.material_id == candidate.material_id:
                            candidate.application_scores[app] = ranked_material.overall_score
                            break
                
                logger.info(f"Ranked {len(ranked)} materials for application: {app}")
            except Exception as e:
                logger.error(f"Failed to rank for application {app}: {e}")
        
        return candidates, application_rankings
    
    def _calculate_scores(self, candidates: List[MaterialCandidate]) -> List[MaterialCandidate]:
        """
        Calculate ranking scores for candidates.
        
        Args:
            candidates: List of material candidates
        
        Returns:
            Candidates with calculated scores
        """
        # Calculate stability scores
        stability_scores = []
        for candidate in candidates:
            if candidate.stability_result:
                # Higher confidence = higher score
                score = candidate.stability_result.confidence_score
                candidate.stability_score = score
                stability_scores.append(score)
            else:
                candidate.stability_score = 0.0
                stability_scores.append(0.0)
        
        # Calculate performance scores
        performance_scores = []
        for candidate in candidates:
            if candidate.predicted_v50:
                # Normalize by max predicted V50
                score = candidate.predicted_v50
                performance_scores.append(score)
            else:
                performance_scores.append(0.0)
        
        # Normalize performance scores
        if performance_scores and max(performance_scores) > 0:
            max_performance = max(performance_scores)
            for i, candidate in enumerate(candidates):
                candidate.performance_score = performance_scores[i] / max_performance
        
        # Calculate cost efficiency scores
        for candidate in candidates:
            if candidate.cost_score:
                # Lower cost = higher efficiency score
                candidate.cost_efficiency_score = 1.0 / (1.0 + candidate.cost_score)
            else:
                # Default cost score based on dopant rarity
                candidate.cost_efficiency_score = self._estimate_cost_efficiency(candidate)
        
        # Calculate combined scores
        for candidate in candidates:
            base_score = (
                self.config.stability_weight * candidate.stability_score +
                self.config.performance_weight * candidate.performance_score +
                self.config.cost_weight * candidate.cost_efficiency_score
            )
            
            # Add application ranking component if enabled
            if self.config.enable_application_ranking and candidate.application_scores:
                # Use average of application scores
                app_scores = list(candidate.application_scores.values())
                if app_scores:
                    avg_app_score = np.mean(app_scores)
                    candidate.combined_score = (
                        base_score * (1.0 - self.config.application_ranking_weight) +
                        avg_app_score * self.config.application_ranking_weight
                    )
                else:
                    candidate.combined_score = base_score
            else:
                candidate.combined_score = base_score
        
        return candidates
    
    def _estimate_cost_efficiency(self, candidate: MaterialCandidate) -> float:
        """
        Estimate cost efficiency based on dopant element.
        
        Args:
            candidate: Material candidate
        
        Returns:
            Cost efficiency score (0-1, higher is better)
        """
        # Simplified cost model based on element rarity
        # In production, would use actual cost data
        
        common_elements = ['C', 'N', 'O', 'Si', 'Al', 'Ti', 'Fe']
        moderate_elements = ['B', 'Mg', 'Ca', 'Cr', 'Ni', 'Cu', 'Zn']
        rare_elements = ['Y', 'Zr', 'Nb', 'Mo', 'W', 'Ta', 'Re']
        
        if candidate.dopant_element is None:
            return 1.0  # Undoped is cheapest
        
        if candidate.dopant_element in common_elements:
            return 0.9
        elif candidate.dopant_element in moderate_elements:
            return 0.7
        elif candidate.dopant_element in rare_elements:
            return 0.4
        else:
            return 0.5  # Unknown element
    
    def _rank_candidates(
        self,
        candidates: List[MaterialCandidate],
        criterion: RankingCriterion = RankingCriterion.COMBINED
    ) -> List[MaterialCandidate]:
        """
        Rank candidates by specified criterion.
        
        Args:
            candidates: List of material candidates
            criterion: Ranking criterion
        
        Returns:
            Sorted list of candidates (best first)
        """
        if criterion == RankingCriterion.STABILITY:
            key = lambda c: c.stability_score
        elif criterion == RankingCriterion.PERFORMANCE:
            key = lambda c: c.performance_score
        elif criterion == RankingCriterion.COST:
            key = lambda c: c.cost_efficiency_score
        else:  # COMBINED
            key = lambda c: c.combined_score
        
        return sorted(candidates, key=key, reverse=True)
    
    def batch_screen(
        self,
        all_candidates: List[MaterialCandidate],
        screening_id: str = "batch_screening"
    ) -> List[ScreeningResults]:
        """
        Screen candidates in batches.
        
        Args:
            all_candidates: List of all material candidates
            screening_id: Base identifier for screening runs
        
        Returns:
            List of screening results for each batch
        """
        results = []
        
        num_batches = (len(all_candidates) + self.config.batch_size - 1) // self.config.batch_size
        logger.info(f"Processing {len(all_candidates)} candidates in {num_batches} batches")
        
        for i in range(0, len(all_candidates), self.config.batch_size):
            batch = all_candidates[i:i + self.config.batch_size]
            batch_id = f"{screening_id}_batch_{i // self.config.batch_size + 1}"
            
            batch_results = self.screen_candidates(batch, batch_id)
            results.append(batch_results)
        
        return results
    
    def _get_cached_stability(self, material_id: str) -> Optional[StabilityResult]:
        """Get cached stability result."""
        if not self.redis_client or not self.config.use_cache:
            return None
        
        try:
            key = f"stability:{material_id}"
            data = self.redis_client.get(key)
            
            if data:
                # Deserialize (simplified - would need proper serialization)
                return None  # Placeholder
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        
        return None
    
    def _cache_stability(self, material_id: str, result: StabilityResult) -> None:
        """Cache stability result."""
        if not self.redis_client or not self.config.use_cache:
            return
        
        try:
            key = f"stability:{material_id}"
            # Serialize (simplified - would need proper serialization)
            self.redis_client.setex(key, self.config.cache_ttl_seconds, "placeholder")
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    def _get_cached_prediction(self, material_id: str) -> Optional[Dict[str, float]]:
        """Get cached ML prediction."""
        if not self.redis_client or not self.config.use_cache:
            return None
        
        try:
            key = f"prediction:{material_id}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        
        return None
    
    def _cache_prediction(self, material_id: str, prediction: Dict[str, float]) -> None:
        """Cache ML prediction."""
        if not self.redis_client or not self.config.use_cache:
            return
        
        try:
            key = f"prediction:{material_id}"
            self.redis_client.setex(
                key,
                self.config.cache_ttl_seconds,
                json.dumps(prediction)
            )
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    def get_top_candidates(
        self,
        results: ScreeningResults,
        n: int = 10,
        criterion: RankingCriterion = RankingCriterion.COMBINED
    ) -> List[MaterialCandidate]:
        """
        Get top N candidates from screening results.
        
        Args:
            results: Screening results
            n: Number of top candidates to return
            criterion: Ranking criterion
        
        Returns:
            List of top candidates
        """
        ranked = self._rank_candidates(results.ranked_candidates, criterion)
        return ranked[:n]
    
    def compare_candidates(
        self,
        candidate1: MaterialCandidate,
        candidate2: MaterialCandidate
    ) -> Dict[str, Any]:
        """
        Compare two material candidates.
        
        Args:
            candidate1: First candidate
            candidate2: Second candidate
        
        Returns:
            Comparison dictionary
        """
        return {
            'candidate1': {
                'formula': candidate1.formula,
                'stability_score': candidate1.stability_score,
                'performance_score': candidate1.performance_score,
                'cost_score': candidate1.cost_efficiency_score,
                'combined_score': candidate1.combined_score,
                'application_scores': candidate1.application_scores,
            },
            'candidate2': {
                'formula': candidate2.formula,
                'stability_score': candidate2.stability_score,
                'performance_score': candidate2.performance_score,
                'cost_score': candidate2.cost_efficiency_score,
                'combined_score': candidate2.combined_score,
                'application_scores': candidate2.application_scores,
            },
            'winner': candidate1.formula if candidate1.combined_score > candidate2.combined_score else candidate2.formula,
        }
    
    def filter_by_application(
        self,
        results: ScreeningResults,
        application: str,
        min_score: float = 0.5
    ) -> List[MaterialCandidate]:
        """
        Filter candidates by application score threshold.
        
        Args:
            results: Screening results
            application: Application name
            min_score: Minimum application score threshold
        
        Returns:
            List of candidates meeting the threshold
        """
        if not results.application_rankings or application not in results.application_rankings:
            logger.warning(f"No rankings available for application: {application}")
            return []
        
        filtered = []
        for candidate in results.ranked_candidates:
            if candidate.application_scores and application in candidate.application_scores:
                if candidate.application_scores[application] >= min_score:
                    filtered.append(candidate)
        
        logger.info(
            f"Filtered {len(filtered)} candidates for {application} "
            f"with score >= {min_score}"
        )
        
        return filtered
    
    def rank_by_multiple_objectives(
        self,
        candidates: List[MaterialCandidate],
        objectives: Dict[str, float]
    ) -> List[MaterialCandidate]:
        """
        Rank candidates using multiple objectives with custom weights.
        
        Args:
            candidates: List of material candidates
            objectives: Dictionary of objective names to weights
                       Supported: 'stability', 'performance', 'cost', 'application:<app_name>'
        
        Returns:
            Ranked list of candidates
        """
        # Validate weights sum to 1.0
        total_weight = sum(objectives.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Objective weights must sum to 1.0, got {total_weight}")
        
        # Calculate multi-objective scores
        for candidate in candidates:
            score = 0.0
            
            for objective, weight in objectives.items():
                if objective == 'stability':
                    score += weight * candidate.stability_score
                elif objective == 'performance':
                    score += weight * candidate.performance_score
                elif objective == 'cost':
                    score += weight * candidate.cost_efficiency_score
                elif objective.startswith('application:'):
                    app_name = objective.split(':', 1)[1]
                    if candidate.application_scores and app_name in candidate.application_scores:
                        score += weight * candidate.application_scores[app_name]
                    else:
                        logger.warning(
                            f"Application score for {app_name} not available for {candidate.formula}"
                        )
                else:
                    logger.warning(f"Unknown objective: {objective}")
            
            # Store in combined_score for ranking
            candidate.combined_score = score
        
        # Sort by combined score
        ranked = sorted(candidates, key=lambda c: c.combined_score, reverse=True)
        
        logger.info(
            f"Ranked {len(ranked)} candidates using multi-objective optimization. "
            f"Top candidate: {ranked[0].formula} (score: {ranked[0].combined_score:.3f})"
        )
        
        return ranked
    
    def get_pareto_front(
        self,
        candidates: List[MaterialCandidate],
        objectives: List[str]
    ) -> List[MaterialCandidate]:
        """
        Get Pareto-optimal candidates for multiple objectives.
        
        Args:
            candidates: List of material candidates
            objectives: List of objective names to optimize
                       Supported: 'stability', 'performance', 'cost', 'application:<app_name>'
        
        Returns:
            List of Pareto-optimal candidates
        """
        if not candidates:
            return []
        
        # Extract objective values for each candidate
        objective_values = []
        for candidate in candidates:
            values = []
            for objective in objectives:
                if objective == 'stability':
                    values.append(candidate.stability_score)
                elif objective == 'performance':
                    values.append(candidate.performance_score)
                elif objective == 'cost':
                    values.append(candidate.cost_efficiency_score)
                elif objective.startswith('application:'):
                    app_name = objective.split(':', 1)[1]
                    if candidate.application_scores and app_name in candidate.application_scores:
                        values.append(candidate.application_scores[app_name])
                    else:
                        values.append(0.0)
                else:
                    values.append(0.0)
            objective_values.append(values)
        
        objective_values = np.array(objective_values)
        
        # Find Pareto front (maximize all objectives)
        pareto_front = []
        for i, candidate in enumerate(candidates):
            is_dominated = False
            for j in range(len(candidates)):
                if i == j:
                    continue
                # Check if j dominates i (all objectives >= and at least one >)
                if np.all(objective_values[j] >= objective_values[i]) and \
                   np.any(objective_values[j] > objective_values[i]):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        logger.info(
            f"Found {len(pareto_front)} Pareto-optimal candidates "
            f"from {len(candidates)} total candidates"
        )
        
        return pareto_front
