"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from ceramic_discovery.ml.feature_engineering import (
    FeatureEngineeringPipeline,
    FeatureMetadata,
    FeatureValidator,
)


class TestFeatureValidator:
    """Test feature validation."""
    
    def test_tier_1_properties_are_fundamental(self):
        """Test that Tier 1 properties are correctly identified."""
        validator = FeatureValidator()
        
        tier_1_features = ['density', 'band_gap', 'formation_energy_per_atom']
        for feature in tier_1_features:
            assert feature in validator.TIER_1_PROPERTIES
            assert validator.get_feature_tier(feature) == 1
    
    def test_tier_2_properties_are_fundamental(self):
        """Test that Tier 2 properties are correctly identified."""
        validator = FeatureValidator()
        
        tier_2_features = ['hardness_vickers', 'fracture_toughness', 'thermal_conductivity_300K']
        for feature in tier_2_features:
            assert feature in validator.TIER_2_PROPERTIES
            assert validator.get_feature_tier(feature) == 2
    
    def test_tier_3_properties_are_derived(self):
        """Test that Tier 3 properties are correctly identified as derived."""
        validator = FeatureValidator()
        
        tier_3_features = ['v50_predicted', 'areal_density', 'specific_energy_absorption']
        for feature in tier_3_features:
            assert feature in validator.TIER_3_PROPERTIES
            assert validator.get_feature_tier(feature) == 3
    
    def test_validate_fundamental_feature_set(self):
        """Test validation of fundamental feature set."""
        validator = FeatureValidator()
        
        # Valid feature set (only Tier 1-2)
        valid_features = ['density', 'hardness_vickers', 'band_gap', 'thermal_conductivity_300K']
        is_valid, violations = validator.validate_feature_set(valid_features)
        
        assert is_valid
        assert len(violations) == 0
    
    def test_reject_derived_features(self):
        """Test that derived features are rejected."""
        validator = FeatureValidator()
        
        # Invalid feature set (contains Tier 3)
        invalid_features = ['density', 'hardness_vickers', 'v50_predicted']
        is_valid, violations = validator.validate_feature_set(invalid_features)
        
        assert not is_valid
        assert len(violations) > 0
        assert 'v50_predicted' in violations[0]
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        validator = FeatureValidator()
        
        # Circular dependency: using derived feature to predict derived target
        features = ['density', 'areal_density']
        target = 'v50_predicted'
        
        is_valid, violations = validator.check_circular_dependencies(features, target)
        
        assert not is_valid
        assert len(violations) > 0
        assert 'areal_density' in violations[0]


class TestFeatureMetadata:
    """Test feature metadata."""
    
    def test_valid_tier_1_metadata(self):
        """Test creation of valid Tier 1 metadata."""
        metadata = FeatureMetadata(
            name='density',
            tier=1,
            category='structural',
            unit='g/cm³',
            is_fundamental=True
        )
        
        assert metadata.name == 'density'
        assert metadata.tier == 1
        assert metadata.is_fundamental
    
    def test_valid_tier_3_metadata(self):
        """Test creation of valid Tier 3 metadata."""
        metadata = FeatureMetadata(
            name='v50_predicted',
            tier=3,
            category='ballistic',
            unit='m/s',
            is_fundamental=False
        )
        
        assert metadata.name == 'v50_predicted'
        assert metadata.tier == 3
        assert not metadata.is_fundamental
    
    def test_invalid_tier_raises_error(self):
        """Test that invalid tier raises error."""
        with pytest.raises(ValueError, match="Invalid tier"):
            FeatureMetadata(
                name='test',
                tier=4,
                category='test',
                unit='test',
                is_fundamental=False
            )
    
    def test_tier_fundamental_mismatch_raises_error(self):
        """Test that tier-fundamental mismatch raises error."""
        with pytest.raises(ValueError, match="fundamental status mismatch"):
            FeatureMetadata(
                name='density',
                tier=1,
                category='structural',
                unit='g/cm³',
                is_fundamental=False  # Should be True for Tier 1
            )


class TestFeatureEngineeringPipeline:
    """Test feature engineering pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample feature data."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'density': np.random.uniform(2.0, 5.0, n_samples),
            'hardness_vickers': np.random.uniform(20.0, 35.0, n_samples),
            'fracture_toughness': np.random.uniform(3.0, 6.0, n_samples),
            'thermal_conductivity_300K': np.random.uniform(20.0, 120.0, n_samples),
            'band_gap': np.random.uniform(0.0, 5.0, n_samples),
        })
        
        return data
    
    def test_pipeline_fit_transform(self, sample_data):
        """Test basic fit and transform."""
        pipeline = FeatureEngineeringPipeline(scaling_method='standard')
        
        X_transformed = pipeline.fit_transform(sample_data)
        
        assert X_transformed.shape == sample_data.shape
        assert list(X_transformed.columns) == list(sample_data.columns)
        
        # Check that features are standardized (mean ≈ 0, std ≈ 1)
        assert np.abs(X_transformed.mean().mean()) < 0.1
        assert np.abs(X_transformed.std().mean() - 1.0) < 0.1
    
    def test_pipeline_rejects_derived_features(self):
        """Test that pipeline rejects derived features."""
        data = pd.DataFrame({
            'density': [3.0, 3.5, 4.0],
            'v50_predicted': [1500, 1600, 1700],  # Derived feature
        })
        
        pipeline = FeatureEngineeringPipeline()
        
        with pytest.raises(ValueError, match="Invalid feature set"):
            pipeline.fit(data)
    
    def test_pipeline_detects_circular_dependencies(self, sample_data):
        """Test circular dependency detection."""
        pipeline = FeatureEngineeringPipeline()
        
        # Add derived feature
        data_with_derived = sample_data.copy()
        data_with_derived['areal_density'] = np.random.uniform(10, 20, len(sample_data))
        
        with pytest.raises(ValueError, match="Invalid feature set"):
            pipeline.fit(data_with_derived, target_name='v50_predicted')
    
    def test_missing_value_handling_drop(self, sample_data):
        """Test missing value handling with drop strategy."""
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0, 'density'] = np.nan
        data_with_missing.loc[1, 'hardness_vickers'] = np.nan
        
        pipeline = FeatureEngineeringPipeline(handle_missing='drop')
        X_transformed = pipeline.fit_transform(data_with_missing)
        
        # Should have dropped 2 rows
        assert len(X_transformed) == len(sample_data) - 2
        assert not X_transformed.isna().any().any()
    
    def test_missing_value_handling_median(self, sample_data):
        """Test missing value handling with median imputation."""
        data_with_missing = sample_data.copy()
        original_median = data_with_missing['density'].median()
        data_with_missing.loc[0, 'density'] = np.nan
        
        pipeline = FeatureEngineeringPipeline(handle_missing='median')
        X_transformed = pipeline.fit_transform(data_with_missing)
        
        # Should have same number of rows
        assert len(X_transformed) == len(sample_data)
        assert not X_transformed.isna().any().any()
    
    def test_low_variance_feature_removal(self):
        """Test removal of low-variance features."""
        data = pd.DataFrame({
            'density': [3.0, 3.1, 3.2, 3.3],
            'hardness_vickers': [25.0, 25.0, 25.0, 25.0],  # Zero variance
        })
        
        pipeline = FeatureEngineeringPipeline(min_feature_variance=1e-6)
        X_transformed = pipeline.fit_transform(data)
        
        # Should have removed hardness_vickers
        assert 'hardness_vickers' not in X_transformed.columns
        assert 'density' in X_transformed.columns
    
    def test_feature_importance_analysis(self, sample_data):
        """Test feature importance analysis."""
        pipeline = FeatureEngineeringPipeline()
        pipeline.fit(sample_data)
        
        # Simulate feature importances from a model
        importances = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        importance_df = pipeline.get_feature_importance_analysis(importances, top_n=3)
        
        assert len(importance_df) == 3
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'tier' in importance_df.columns
        
        # Should be sorted by importance
        assert importance_df['importance'].is_monotonic_decreasing
    
    def test_physical_bounds_validation(self, sample_data):
        """Test physical bounds validation."""
        pipeline = FeatureEngineeringPipeline()
        
        # Valid data
        is_valid, violations = pipeline.validate_physical_bounds(sample_data)
        assert is_valid
        assert len(violations) == 0
        
        # Invalid data (density out of bounds)
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'density'] = 100.0  # Way too high
        
        is_valid, violations = pipeline.validate_physical_bounds(invalid_data)
        assert not is_valid
        assert len(violations) > 0
        assert 'density' in violations[0]
    
    def test_minmax_scaling(self, sample_data):
        """Test MinMax scaling."""
        pipeline = FeatureEngineeringPipeline(scaling_method='minmax')
        X_transformed = pipeline.fit_transform(sample_data)
        
        # Check that features are in [0, 1] range (with small tolerance for floating point)
        assert X_transformed.min().min() >= -1e-10
        assert X_transformed.max().max() <= 1.0 + 1e-10
    
    def test_feature_statistics(self, sample_data):
        """Test feature statistics retrieval."""
        pipeline = FeatureEngineeringPipeline()
        pipeline.fit(sample_data)
        
        stats = pipeline.get_feature_statistics()
        
        assert len(stats) == len(sample_data.columns)
        assert 'feature' in stats.columns
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'tier' in stats.columns
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises error."""
        pipeline = FeatureEngineeringPipeline()
        
        with pytest.raises(ValueError, match="must be fitted"):
            pipeline.transform(sample_data)
    
    def test_pipeline_preserves_feature_order(self, sample_data):
        """Test that pipeline preserves feature order."""
        pipeline = FeatureEngineeringPipeline()
        
        X_train = sample_data.iloc[:80]
        X_test = sample_data.iloc[80:]
        
        pipeline.fit(X_train)
        X_test_transformed = pipeline.transform(X_test)
        
        assert list(X_test_transformed.columns) == pipeline.selected_features


class TestEnhancedFeatureEngineeringPipeline:
    """Test enhanced feature engineering pipeline with composition and structure descriptors."""
    
    @pytest.fixture
    def sample_data_with_formulas(self):
        """Create sample data with formulas and properties."""
        np.random.seed(42)
        n_samples = 50
        
        formulas = pd.Series([
            'SiC', 'TiC', 'B4C', 'ZrC', 'HfC',
            'TaC', 'NbC', 'WC', 'VC', 'Cr3C2'
        ] * 5)
        
        data = pd.DataFrame({
            'density': np.random.uniform(2.0, 5.0, n_samples),
            'hardness_vickers': np.random.uniform(20.0, 35.0, n_samples),
            'bulk_modulus_vrh': np.random.uniform(200.0, 400.0, n_samples),
            'shear_modulus_vrh': np.random.uniform(150.0, 250.0, n_samples),
            'formation_energy_per_atom': np.random.uniform(-3.0, -1.0, n_samples),
            'band_gap': np.random.uniform(0.0, 5.0, n_samples),
        })
        
        return data, formulas
    
    def test_composition_descriptors_integration(self, sample_data_with_formulas):
        """Test end-to-end feature extraction with composition descriptors."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_composition_descriptors=True,
            scaling_method='standard'
        )
        
        X_transformed = pipeline.fit_transform(data, formulas=formulas)
        
        # Check that composition descriptors were added
        # Note: num_elements may be removed due to low variance in carbides
        has_composition = any('composition' in f or 'num_atoms' in f or 'mean_' in f 
                             for f in pipeline.selected_features)
        assert has_composition, "No composition descriptors found in selected features"
        
        # Check that original features are still present
        assert any('density' in f or 'hardness' in f for f in pipeline.selected_features)
        
        # Check that output is properly scaled
        assert not X_transformed.isna().any().any()
    
    def test_structure_descriptors_integration(self, sample_data_with_formulas):
        """Test end-to-end feature extraction with structure descriptors."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_structure_descriptors=True,
            scaling_method='standard'
        )
        
        X_transformed = pipeline.fit_transform(data, formulas=formulas)
        
        # Check that structure descriptors were added
        assert 'pugh_ratio' in pipeline.selected_features or 'pugh_ratio' in X_transformed.columns
        assert 'youngs_modulus' in pipeline.selected_features or 'youngs_modulus' in X_transformed.columns
        
        # Check that original features are still present
        assert any('density' in f or 'hardness' in f for f in pipeline.selected_features)
        
        # Check that output is properly scaled
        assert not X_transformed.isna().any().any()
    
    def test_both_descriptors_integration(self, sample_data_with_formulas):
        """Test end-to-end feature extraction with both composition and structure descriptors."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_composition_descriptors=True,
            include_structure_descriptors=True,
            scaling_method='standard'
        )
        
        X_transformed = pipeline.fit_transform(data, formulas=formulas)
        
        # Check that both types of descriptors were added
        has_composition = any('composition' in f or 'num_elements' in f or 'mean_' in f 
                             for f in pipeline.selected_features)
        has_structure = any('pugh' in f or 'youngs' in f or 'poisson' in f 
                           for f in pipeline.selected_features)
        
        assert has_composition or has_structure
        
        # Check that output has more features than input
        assert len(pipeline.selected_features) >= len(data.columns)
        
        # Check that output is properly scaled
        assert not X_transformed.isna().any().any()
    
    def test_feature_validation_with_new_descriptors(self, sample_data_with_formulas):
        """Test feature validation with composition and structure features."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_composition_descriptors=True,
            include_structure_descriptors=True
        )
        
        # Should not raise validation errors
        X_transformed = pipeline.fit_transform(data, formulas=formulas)
        
        # Validate that all features are fundamental (Tier 1-2)
        for feature in pipeline.selected_features:
            if feature in pipeline.feature_metadata:
                assert pipeline.feature_metadata[feature].is_fundamental
    
    def test_backward_compatibility_without_descriptors(self, sample_data_with_formulas):
        """Test backward compatibility with existing features when descriptors disabled."""
        data, formulas = sample_data_with_formulas
        
        # Pipeline without new descriptors
        pipeline_old = FeatureEngineeringPipeline(
            include_composition_descriptors=False,
            include_structure_descriptors=False
        )
        
        X_old = pipeline_old.fit_transform(data)
        
        # Should work exactly as before
        assert set(X_old.columns) == set(pipeline_old.selected_features)
        assert len(X_old.columns) <= len(data.columns)
        
        # Check that no new descriptors were added
        assert not any('composition' in f for f in X_old.columns)
        assert not any('pugh' in f for f in X_old.columns)
    
    def test_composition_descriptors_without_formulas(self, sample_data_with_formulas):
        """Test that composition descriptors are skipped when formulas not provided."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_composition_descriptors=True
        )
        
        # Fit without formulas - should warn and skip composition descriptors
        with pytest.warns(UserWarning, match="no formulas provided"):
            X_transformed = pipeline.fit_transform(data)
        
        # Should not have composition descriptors
        assert not any('composition' in f for f in pipeline.selected_features)
        assert not any('num_elements' in f for f in pipeline.selected_features)
    
    def test_transform_consistency_with_descriptors(self, sample_data_with_formulas):
        """Test that transform produces consistent results with descriptors."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_composition_descriptors=True,
            include_structure_descriptors=True
        )
        
        # Split data
        X_train = data.iloc[:40]
        X_test = data.iloc[40:]
        formulas_train = formulas.iloc[:40]
        formulas_test = formulas.iloc[40:]
        
        # Fit on train
        pipeline.fit(X_train, formulas=formulas_train)
        
        # Transform test
        X_test_transformed = pipeline.transform(X_test, formulas=formulas_test)
        
        # Check that same features are present
        assert list(X_test_transformed.columns) == pipeline.selected_features
        
        # Check that output is properly scaled
        assert not X_test_transformed.isna().any().any()
    
    def test_feature_metadata_tracking_with_descriptors(self, sample_data_with_formulas):
        """Test that feature metadata is properly tracked for new descriptors."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_composition_descriptors=True,
            include_structure_descriptors=True
        )
        
        pipeline.fit(data, formulas=formulas)
        
        # Check that metadata exists for all selected features
        for feature in pipeline.selected_features:
            assert feature in pipeline.feature_metadata
            metadata = pipeline.feature_metadata[feature]
            
            # Check that category is set for new descriptors
            if 'composition' in feature or 'num_' in feature or 'mean_' in feature:
                assert metadata.category == 'composition'
            elif feature in ['pugh_ratio', 'youngs_modulus', 'poisson_ratio', 'energy_density']:
                assert metadata.category == 'structure'
    
    def test_missing_formulas_in_data(self, sample_data_with_formulas):
        """Test handling of missing formulas in dataset."""
        data, formulas = sample_data_with_formulas
        
        # Add some missing formulas
        formulas_with_missing = formulas.copy()
        formulas_with_missing.iloc[0] = np.nan
        formulas_with_missing.iloc[5] = None
        
        pipeline = FeatureEngineeringPipeline(
            include_composition_descriptors=True,
            handle_missing='drop'
        )
        
        # Should handle missing formulas gracefully
        X_transformed = pipeline.fit_transform(data, formulas=formulas_with_missing)
        
        # Should have dropped rows with missing formulas
        assert len(X_transformed) < len(data)
    
    def test_physical_bounds_with_structure_descriptors(self, sample_data_with_formulas):
        """Test physical bounds validation with structure descriptors."""
        data, formulas = sample_data_with_formulas
        
        pipeline = FeatureEngineeringPipeline(
            include_structure_descriptors=True
        )
        
        pipeline.fit(data, formulas=formulas)
        
        # Get the enhanced data before scaling
        data_enhanced = data.copy()
        data_enhanced = pipeline._add_structure_descriptors(data_enhanced)
        
        # Validate physical bounds
        custom_bounds = {
            'pugh_ratio': (0.5, 5.0),
            'poisson_ratio': (-1.0, 0.5),
            'youngs_modulus': (0.0, 1000.0),
        }
        
        is_valid, violations = pipeline.validate_physical_bounds(data_enhanced, custom_bounds)
        
        # Should be valid or have specific violations
        if not is_valid:
            # Check that violations are reasonable
            for violation in violations:
                assert any(prop in violation for prop in custom_bounds.keys())
