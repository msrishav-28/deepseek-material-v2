"""Machine learning module for ceramic armor discovery."""

from ceramic_discovery.ml.feature_engineering import (
    FeatureEngineeringPipeline,
    FeatureMetadata,
    FeatureSet,
    FeatureValidator,
)
from ceramic_discovery.ml.model_trainer import (
    ModelTrainer,
    ModelMetrics,
    TrainedModel,
)
from ceramic_discovery.ml.uncertainty_quantification import (
    UncertaintyQuantifier,
    UncertaintyPropagator,
    ConfidenceIntervalReporter,
    PredictionWithUncertainty,
    UncertaintyMetrics,
)
from ceramic_discovery.ml.composition_descriptors import (
    CompositionDescriptorCalculator,
)
from ceramic_discovery.ml.structure_descriptors import (
    StructureDescriptorCalculator,
)

__all__ = [
    'FeatureEngineeringPipeline',
    'FeatureMetadata',
    'FeatureSet',
    'FeatureValidator',
    'ModelTrainer',
    'ModelMetrics',
    'TrainedModel',
    'UncertaintyQuantifier',
    'UncertaintyPropagator',
    'ConfidenceIntervalReporter',
    'PredictionWithUncertainty',
    'UncertaintyMetrics',
    'CompositionDescriptorCalculator',
    'StructureDescriptorCalculator',
]
