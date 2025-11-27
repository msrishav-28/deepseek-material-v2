"""Property-based tests for report generation robustness."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume

from ceramic_discovery.reporting import ReportGenerator


# Strategies for generating test data
@st.composite
def dataframe_strategy(draw, min_rows=0, max_rows=100, min_cols=0, max_cols=20):
    """Generate random DataFrames with various shapes and content."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    if n_rows == 0 or n_cols == 0:
        return pd.DataFrame()
    
    # Generate column names
    col_names = [f"col_{i}" for i in range(n_cols)]
    
    # Generate data with mix of numeric and non-numeric
    data = {}
    for col in col_names:
        # Randomly choose column type
        col_type = draw(st.sampled_from(['numeric', 'string', 'mixed']))
        
        if col_type == 'numeric':
            # Numeric column with possible NaN values
            data[col] = draw(st.lists(
                st.one_of(
                    st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False),
                    st.just(np.nan),
                    st.none()
                ),
                min_size=n_rows,
                max_size=n_rows
            ))
        elif col_type == 'string':
            # String column
            data[col] = draw(st.lists(
                st.one_of(
                    st.text(min_size=0, max_size=20),
                    st.none()
                ),
                min_size=n_rows,
                max_size=n_rows
            ))
        else:
            # Mixed column
            data[col] = draw(st.lists(
                st.one_of(
                    st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False),
                    st.just(np.nan),
                    st.text(min_size=0, max_size=20),
                    st.none()
                ),
                min_size=n_rows,
                max_size=n_rows
            ))
    
    return pd.DataFrame(data)


@st.composite
def ml_results_strategy(draw):
    """Generate random ML results dictionaries."""
    # Randomly choose to return None, empty dict, or populated dict
    result_type = draw(st.sampled_from(['none', 'empty', 'single', 'multiple']))
    
    if result_type == 'none':
        return None
    elif result_type == 'empty':
        return {}
    elif result_type == 'single':
        return {
            'model_type': draw(st.sampled_from(['RandomForest', 'GradientBoosting', 'LinearRegression'])),
            'r2_score': draw(st.one_of(st.floats(min_value=-1.0, max_value=1.0), st.just(np.nan))),
            'mae': draw(st.one_of(st.floats(min_value=0.0, max_value=1000.0), st.just(np.nan))),
            'rmse': draw(st.one_of(st.floats(min_value=0.0, max_value=1000.0), st.just(np.nan))),
            'n_features': draw(st.one_of(st.integers(min_value=1, max_value=100), st.none())),
            'n_samples': draw(st.one_of(st.integers(min_value=10, max_value=10000), st.none())),
        }
    else:  # multiple
        n_models = draw(st.integers(min_value=1, max_value=5))
        models = {}
        for i in range(n_models):
            model_name = f"model_{i}"
            models[model_name] = {
                'r2_score': draw(st.one_of(st.floats(min_value=-1.0, max_value=1.0), st.just(np.nan))),
                'mae': draw(st.one_of(st.floats(min_value=0.0, max_value=1000.0), st.just(np.nan))),
                'rmse': draw(st.one_of(st.floats(min_value=0.0, max_value=1000.0), st.just(np.nan))),
                'n_features': draw(st.one_of(st.integers(min_value=1, max_value=100), st.none())),
                'n_samples': draw(st.one_of(st.integers(min_value=10, max_value=10000), st.none())),
            }
        return {'models': models}


@st.composite
def application_rankings_strategy(draw, min_rows=0, max_rows=50):
    """Generate random application rankings DataFrames."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    if n_rows == 0:
        return pd.DataFrame()
    
    applications = ['aerospace_hypersonic', 'cutting_tools', 'thermal_barriers', 'wear_resistant', 'electronic']
    
    data = {
        'application': draw(st.lists(
            st.sampled_from(applications),
            min_size=n_rows,
            max_size=n_rows
        )),
        'formula': draw(st.lists(
            st.text(min_size=1, max_size=10),
            min_size=n_rows,
            max_size=n_rows
        )),
        'rank': draw(st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=n_rows,
            max_size=n_rows
        )),
        'overall_score': draw(st.lists(
            st.one_of(st.floats(min_value=0.0, max_value=1.0), st.just(np.nan)),
            min_size=n_rows,
            max_size=n_rows
        )),
        'confidence': draw(st.lists(
            st.one_of(st.floats(min_value=0.0, max_value=1.0), st.just(np.nan)),
            min_size=n_rows,
            max_size=n_rows
        )),
    }
    
    return pd.DataFrame(data)


class TestReportGenerationRobustness:
    """
    Property-based tests for report generation robustness.
    
    **Feature: sic-alloy-integration, Property 11: Report generation robustness**
    
    For any dataset (including empty or incomplete), the report generation should
    complete without exceptions and indicate missing data sections.
    """
    
    @given(
        data=dataframe_strategy(),
        ml_results=ml_results_strategy(),
        application_rankings=application_rankings_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_report_generation_never_crashes(
        self,
        data,
        ml_results,
        application_rankings
    ):
        """
        Property 11: Report generation robustness.
        
        For any combination of data, ML results, and application rankings
        (including None, empty, or incomplete), generate_summary_report should:
        1. Complete without raising exceptions
        2. Return a non-empty string
        3. Include section headers
        """
        generator = ReportGenerator(title="Test Report")
        
        # This should never crash regardless of input
        try:
            report = generator.generate_summary_report(
                data=data,
                ml_results=ml_results,
                application_rankings=application_rankings,
                top_n=10
            )
            
            # Report should be a string
            assert isinstance(report, str), "Report should be a string"
            
            # Report should not be empty
            assert len(report) > 0, "Report should not be empty"
            
            # Report should contain key section headers
            assert "DATASET STATISTICS" in report, "Report should have dataset statistics section"
            assert "MACHINE LEARNING MODEL PERFORMANCE" in report, "Report should have ML performance section"
            assert "TOP CANDIDATE RECOMMENDATIONS" in report, "Report should have recommendations section"
            
            # Report should indicate when data is missing
            if data is None or len(data) == 0:
                assert "No dataset provided" in report or "dataset is empty" in report
            
            if ml_results is None or len(ml_results) == 0:
                assert "No ML results provided" in report
            
            if application_rankings is None or len(application_rankings) == 0:
                assert "No application rankings provided" in report
            
        except Exception as e:
            pytest.fail(f"Report generation crashed with exception: {e}")
    
    @given(data=dataframe_strategy(min_rows=1, max_rows=50, min_cols=1, max_cols=10))
    @settings(max_examples=50, deadline=None)
    def test_property_report_handles_all_numeric_data(self, data):
        """
        Property 11 variant: Report handles numeric data correctly.
        
        For any DataFrame with numeric columns, the report should include
        property statistics without crashing.
        """
        generator = ReportGenerator(title="Test Report")
        
        try:
            report = generator.generate_summary_report(data=data)
            
            # Should complete successfully
            assert isinstance(report, str)
            assert len(report) > 0
            
            # Should report number of materials
            assert f"Total materials: {len(data)}" in report
            
        except Exception as e:
            pytest.fail(f"Report generation crashed with numeric data: {e}")
    
    @given(
        ml_results=ml_results_strategy()
    )
    @settings(max_examples=50, deadline=None)
    def test_property_report_handles_all_ml_results(self, ml_results):
        """
        Property 11 variant: Report handles all ML result formats.
        
        For any ML results dictionary (None, empty, single model, multiple models),
        the report should handle it gracefully.
        """
        generator = ReportGenerator(title="Test Report")
        
        try:
            report = generator.generate_summary_report(ml_results=ml_results)
            
            # Should complete successfully
            assert isinstance(report, str)
            assert len(report) > 0
            
            # Should have ML section
            assert "MACHINE LEARNING MODEL PERFORMANCE" in report
            
        except Exception as e:
            pytest.fail(f"Report generation crashed with ML results: {e}")
    
    @given(
        application_rankings=application_rankings_strategy(min_rows=1, max_rows=30)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_report_handles_all_rankings(self, application_rankings):
        """
        Property 11 variant: Report handles all ranking formats.
        
        For any application rankings DataFrame, the report should handle it
        gracefully and display top candidates.
        """
        generator = ReportGenerator(title="Test Report")
        
        try:
            report = generator.generate_summary_report(
                application_rankings=application_rankings,
                top_n=5
            )
            
            # Should complete successfully
            assert isinstance(report, str)
            assert len(report) > 0
            
            # Should have recommendations section
            assert "TOP CANDIDATE RECOMMENDATIONS" in report
            
        except Exception as e:
            pytest.fail(f"Report generation crashed with rankings: {e}")
    
    @given(
        top_n=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_report_handles_various_top_n(self, top_n):
        """
        Property 11 variant: Report handles various top_n values.
        
        For any positive top_n value, the report should handle it correctly.
        """
        generator = ReportGenerator(title="Test Report")
        
        # Create simple test data
        data = pd.DataFrame({
            'formula': ['SiC', 'TiC', 'B4C'],
            'hardness': [28.0, 30.0, 32.0]
        })
        
        try:
            report = generator.generate_summary_report(data=data, top_n=top_n)
            
            # Should complete successfully
            assert isinstance(report, str)
            assert len(report) > 0
            
        except Exception as e:
            pytest.fail(f"Report generation crashed with top_n={top_n}: {e}")
