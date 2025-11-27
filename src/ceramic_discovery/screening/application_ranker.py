"""Application-specific material ranking system."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ApplicationSpec:
    """Specification for a specific application."""
    
    name: str
    description: str
    
    # Target property ranges (min, max) - None means not required
    target_hardness: Optional[Tuple[float, float]] = None
    target_thermal_cond: Optional[Tuple[float, float]] = None
    target_melting_point: Optional[Tuple[float, float]] = None
    target_formation_energy: Optional[Tuple[float, float]] = None
    target_bulk_modulus: Optional[Tuple[float, float]] = None
    target_band_gap: Optional[Tuple[float, float]] = None
    target_density: Optional[Tuple[float, float]] = None
    
    # Property weights (how important each property is for this application)
    weight_hardness: float = 0.0
    weight_thermal_cond: float = 0.0
    weight_melting_point: float = 0.0
    weight_formation_energy: float = 0.0
    weight_bulk_modulus: float = 0.0
    weight_band_gap: float = 0.0
    weight_density: float = 0.0
    
    def get_total_weight(self) -> float:
        """Get total weight of all properties."""
        return (
            self.weight_hardness +
            self.weight_thermal_cond +
            self.weight_melting_point +
            self.weight_formation_energy +
            self.weight_bulk_modulus +
            self.weight_band_gap +
            self.weight_density
        )


@dataclass
class RankedMaterial:
    """Material with application-specific ranking."""
    
    material_id: str
    formula: str
    application: str
    overall_score: float
    component_scores: Dict[str, float]
    rank: int
    confidence: float  # Based on how many properties were available
    
    # Material properties
    properties: Dict[str, Optional[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'material_id': self.material_id,
            'formula': self.formula,
            'application': self.application,
            'overall_score': self.overall_score,
            'rank': self.rank,
            'confidence': self.confidence,
            **self.component_scores,
            **{f'property_{k}': v for k, v in self.properties.items()}
        }


class ApplicationRanker:
    """
    Rank materials for specific applications based on property requirements.
    
    Features:
    - Application-specific scoring with weighted properties
    - Partial scoring for materials with missing properties
    - Batch ranking for multiple applications
    - Configurable target ranges and weights
    """
    
    def __init__(self, applications: Optional[Dict[str, ApplicationSpec]] = None):
        """
        Initialize application ranker.
        
        Args:
            applications: Dictionary of application specifications.
                         If None, uses default applications.
        """
        if applications is None:
            self.applications = self._create_default_applications()
        else:
            self.applications = applications
        
        # Validate applications
        for name, spec in self.applications.items():
            total_weight = spec.get_total_weight()
            if total_weight <= 0:
                raise ValueError(f"Application {name} has no property weights")
    
    def _create_default_applications(self) -> Dict[str, ApplicationSpec]:
        """Create default application specifications."""
        return {
            'aerospace_hypersonic': ApplicationSpec(
                name='aerospace_hypersonic',
                description='Hypersonic aerospace applications (turbine blades, leading edges)',
                target_hardness=(28.0, 35.0),  # GPa
                target_thermal_cond=(50.0, 150.0),  # W/m·K
                target_melting_point=(2500.0, 4000.0),  # K
                weight_hardness=0.35,
                weight_thermal_cond=0.35,
                weight_melting_point=0.30
            ),
            'cutting_tools': ApplicationSpec(
                name='cutting_tools',
                description='Cutting tools and machining applications',
                target_hardness=(30.0, 40.0),  # GPa
                target_formation_energy=(-4.0, -2.0),  # eV/atom (more negative = more stable)
                target_bulk_modulus=(300.0, 500.0),  # GPa
                weight_hardness=0.50,
                weight_formation_energy=0.30,
                weight_bulk_modulus=0.20
            ),
            'thermal_barriers': ApplicationSpec(
                name='thermal_barriers',
                description='Thermal barrier coatings',
                target_thermal_cond=(20.0, 60.0),  # W/m·K (lower is better for barriers)
                target_density=(1.0, 6.0),  # g/cm³
                target_melting_point=(2000.0, 3500.0),  # K
                weight_thermal_cond=0.40,
                weight_density=0.30,
                weight_melting_point=0.30
            ),
            'wear_resistant': ApplicationSpec(
                name='wear_resistant',
                description='Wear-resistant coatings and surfaces',
                target_hardness=(25.0, 40.0),  # GPa
                target_bulk_modulus=(250.0, 450.0),  # GPa
                target_formation_energy=(-5.0, -1.5),  # eV/atom
                weight_hardness=0.45,
                weight_bulk_modulus=0.35,
                weight_formation_energy=0.20
            ),
            'electronic': ApplicationSpec(
                name='electronic',
                description='Electronic and semiconductor applications',
                target_band_gap=(1.0, 4.0),  # eV
                target_thermal_cond=(100.0, 300.0),  # W/m·K
                target_hardness=(20.0, 35.0),  # GPa
                weight_band_gap=0.50,
                weight_thermal_cond=0.30,
                weight_hardness=0.20
            )
        }
    
    def calculate_application_score(
        self,
        properties: Dict[str, Optional[float]],
        application: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate compatibility score for a material and application.
        
        Args:
            properties: Dictionary of material properties
            application: Application name
        
        Returns:
            Tuple of (overall_score, component_scores)
            - overall_score: Weighted average score in [0, 1]
            - component_scores: Individual property scores
        """
        if application not in self.applications:
            raise ValueError(f"Unknown application: {application}")
        
        spec = self.applications[application]
        component_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Score each property
        property_configs = [
            ('hardness', spec.target_hardness, spec.weight_hardness),
            ('thermal_conductivity', spec.target_thermal_cond, spec.weight_thermal_cond),
            ('melting_point', spec.target_melting_point, spec.weight_melting_point),
            ('formation_energy', spec.target_formation_energy, spec.weight_formation_energy),
            ('bulk_modulus', spec.target_bulk_modulus, spec.weight_bulk_modulus),
            ('band_gap', spec.target_band_gap, spec.weight_band_gap),
            ('density', spec.target_density, spec.weight_density),
        ]
        
        for prop_name, target_range, weight in property_configs:
            if weight <= 0:
                continue  # Skip properties not used in this application
            
            prop_value = properties.get(prop_name)
            
            if prop_value is None or not np.isfinite(prop_value):
                # Missing property - skip but track for confidence
                component_scores[f'score_{prop_name}'] = None
                continue
            
            if target_range is None:
                # No target range specified - skip
                component_scores[f'score_{prop_name}'] = None
                continue
            
            # Calculate property score
            score = self._score_property(prop_value, target_range)
            component_scores[f'score_{prop_name}'] = score
            
            # Add to weighted sum
            weighted_sum += weight * score
            total_weight += weight
        
        # Calculate overall score
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.0
        
        # Ensure score is in [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))
        
        return overall_score, component_scores
    
    def _score_property(
        self,
        value: float,
        target_range: Tuple[float, float]
    ) -> float:
        """
        Score a property value against target range.
        
        Args:
            value: Property value
            target_range: (min, max) target range
        
        Returns:
            Score in [0, 1], where 1.0 means within target range
        """
        min_val, max_val = target_range
        
        if min_val <= value <= max_val:
            # Within target range - perfect score
            return 1.0
        
        # Outside target range - calculate distance penalty
        if value < min_val:
            distance = min_val - value
            threshold = min_val * 0.5  # 50% deviation threshold
        else:  # value > max_val
            distance = value - max_val
            threshold = max_val * 0.5  # 50% deviation threshold
        
        # Linear decay from 1.0 to 0.0 over threshold distance
        if threshold > 0:
            score = max(0.0, 1.0 - distance / threshold)
        else:
            score = 0.0
        
        return score
    
    def rank_materials(
        self,
        materials: List[Dict[str, Any]],
        application: str
    ) -> List[RankedMaterial]:
        """
        Rank materials for a specific application.
        
        Args:
            materials: List of material dictionaries with properties
            application: Application name
        
        Returns:
            List of ranked materials (best first)
        """
        if application not in self.applications:
            raise ValueError(f"Unknown application: {application}")
        
        ranked_materials = []
        
        for material in materials:
            # Extract properties
            properties = {
                'hardness': material.get('hardness'),
                'thermal_conductivity': material.get('thermal_conductivity'),
                'melting_point': material.get('melting_point'),
                'formation_energy': material.get('formation_energy'),
                'bulk_modulus': material.get('bulk_modulus'),
                'band_gap': material.get('band_gap'),
                'density': material.get('density'),
            }
            
            # Calculate score
            overall_score, component_scores = self.calculate_application_score(
                properties, application
            )
            
            # Calculate confidence based on available properties
            spec = self.applications[application]
            total_weight = spec.get_total_weight()
            available_weight = sum(
                weight for prop, weight in [
                    (properties['hardness'], spec.weight_hardness),
                    (properties['thermal_conductivity'], spec.weight_thermal_cond),
                    (properties['melting_point'], spec.weight_melting_point),
                    (properties['formation_energy'], spec.weight_formation_energy),
                    (properties['bulk_modulus'], spec.weight_bulk_modulus),
                    (properties['band_gap'], spec.weight_band_gap),
                    (properties['density'], spec.weight_density),
                ]
                if prop is not None and np.isfinite(prop)
            )
            
            confidence = available_weight / total_weight if total_weight > 0 else 0.0
            
            ranked_material = RankedMaterial(
                material_id=material.get('material_id', 'unknown'),
                formula=material.get('formula', 'unknown'),
                application=application,
                overall_score=overall_score,
                component_scores=component_scores,
                rank=0,  # Will be set after sorting
                confidence=confidence,
                properties=properties
            )
            
            ranked_materials.append(ranked_material)
        
        # Sort by overall score (descending)
        ranked_materials.sort(key=lambda m: m.overall_score, reverse=True)
        
        # Assign ranks
        for i, material in enumerate(ranked_materials):
            material.rank = i + 1
        
        logger.info(
            f"Ranked {len(ranked_materials)} materials for {application}. "
            f"Top score: {ranked_materials[0].overall_score:.3f} ({ranked_materials[0].formula})"
        )
        
        return ranked_materials
    
    def rank_for_all_applications(
        self,
        materials: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Rank materials for all defined applications.
        
        Args:
            materials: List of material dictionaries with properties
        
        Returns:
            DataFrame with rankings for all applications
        """
        all_rankings = []
        
        for app_name in self.applications.keys():
            ranked = self.rank_materials(materials, app_name)
            
            for material in ranked:
                row = material.to_dict()
                all_rankings.append(row)
        
        df = pd.DataFrame(all_rankings)
        
        logger.info(
            f"Ranked {len(materials)} materials for {len(self.applications)} applications. "
            f"Total rankings: {len(all_rankings)}"
        )
        
        return df
    
    def get_top_candidates(
        self,
        materials: List[Dict[str, Any]],
        application: str,
        n: int = 10,
        min_confidence: float = 0.5
    ) -> List[RankedMaterial]:
        """
        Get top N candidates for an application.
        
        Args:
            materials: List of material dictionaries
            application: Application name
            n: Number of top candidates to return
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of top candidates
        """
        ranked = self.rank_materials(materials, application)
        
        # Filter by confidence
        filtered = [m for m in ranked if m.confidence >= min_confidence]
        
        if len(filtered) < len(ranked):
            logger.info(
                f"Filtered {len(ranked) - len(filtered)} materials below "
                f"confidence threshold {min_confidence}"
            )
        
        return filtered[:n]
    
    def compare_materials(
        self,
        material1: Dict[str, Any],
        material2: Dict[str, Any],
        application: str
    ) -> Dict[str, Any]:
        """
        Compare two materials for a specific application.
        
        Args:
            material1: First material dictionary
            material2: Second material dictionary
            application: Application name
        
        Returns:
            Comparison dictionary
        """
        # Rank both materials
        ranked = self.rank_materials([material1, material2], application)
        
        return {
            'application': application,
            'material1': {
                'formula': ranked[0].formula if ranked[0].material_id == material1.get('material_id') else ranked[1].formula,
                'score': ranked[0].overall_score if ranked[0].material_id == material1.get('material_id') else ranked[1].overall_score,
                'confidence': ranked[0].confidence if ranked[0].material_id == material1.get('material_id') else ranked[1].confidence,
            },
            'material2': {
                'formula': ranked[1].formula if ranked[1].material_id == material2.get('material_id') else ranked[0].formula,
                'score': ranked[1].overall_score if ranked[1].material_id == material2.get('material_id') else ranked[0].overall_score,
                'confidence': ranked[1].confidence if ranked[1].material_id == material2.get('material_id') else ranked[0].confidence,
            },
            'winner': ranked[0].formula,
            'score_difference': abs(ranked[0].overall_score - ranked[1].overall_score)
        }
    
    def list_applications(self) -> List[str]:
        """List all available applications."""
        return list(self.applications.keys())
    
    def get_application_spec(self, application: str) -> ApplicationSpec:
        """Get specification for an application."""
        if application not in self.applications:
            raise ValueError(f"Unknown application: {application}")
        return self.applications[application]
