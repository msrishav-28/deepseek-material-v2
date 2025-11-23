"""
Property-based tests for data conversion utilities.

**Feature: sic-alloy-integration, Property 3: Safe conversion robustness**
**Validates: Requirements 6.1, 6.4**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays

from ceramic_discovery.utils.data_conversion import (
    safe_float,
    safe_int,
    extract_numeric_from_dict,
    extract_numeric_from_list,
)


# Custom strategies for generating test data
@st.composite
def valid_numeric_values(draw):
    """Generate valid numeric values that should convert successfully."""
    choice = draw(st.integers(min_value=0, max_value=6))
    
    if choice == 0:
        # Regular integers
        return draw(st.integers(min_value=-1000000, max_value=1000000))
    elif choice == 1:
        # Regular floats (excluding inf and nan)
        return draw(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False))
    elif choice == 2:
        # String representations of numbers
        num = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
        return str(num)
    elif choice == 3:
        # Dictionary with "value" key
        num = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
        return {"value": num}
    elif choice == 4:
        # List of numbers
        nums = draw(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=10))
        return nums
    elif choice == 5:
        # Tuple of numbers
        nums = draw(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=10))
        return tuple(nums)
    else:
        # Numpy array
        arr = draw(arrays(dtype=np.float64, shape=st.integers(min_value=1, max_value=10)))
        # Filter out inf and nan
        assume(np.all(np.isfinite(arr)))
        return arr


@st.composite
def invalid_values(draw):
    """Generate values that should return None."""
    choice = draw(st.integers(min_value=0, max_value=5))
    
    if choice == 0:
        return None
    elif choice == 1:
        return float('nan')
    elif choice == 2:
        return float('inf')
    elif choice == 3:
        return float('-inf')
    elif choice == 4:
        # Non-numeric string
        return draw(st.text(alphabet=st.characters(blacklist_characters='0123456789.-+eE'), min_size=1))
    else:
        # Empty list
        return []


class TestSafeConversionRobustness:
    """
    Property 3: Safe conversion robustness
    
    For any input value of supported types (None, NaN, int, float, string, dict, 
    list, tuple, array), the safe_float function should either return a valid float 
    or None without raising exceptions.
    """
    
    @given(value=st.one_of(
        st.none(),
        st.integers(),
        st.floats(),
        st.text(),
        st.dictionaries(st.text(), st.one_of(st.integers(), st.floats(), st.text())),
        st.lists(st.one_of(st.integers(), st.floats())),
        st.tuples(st.one_of(st.integers(), st.floats())),
    ))
    def test_safe_float_never_raises_exception(self, value):
        """
        Property test: safe_float should never raise an exception.
        
        This is the core robustness property - no matter what input we provide,
        the function should handle it gracefully.
        """
        try:
            result = safe_float(value)
            # Result must be either a float or None
            assert result is None or isinstance(result, float)
            # If result is a float, it must be finite
            if result is not None:
                assert np.isfinite(result), f"Result {result} is not finite"
        except Exception as e:
            pytest.fail(f"safe_float raised exception {type(e).__name__}: {e} for input {value}")
    
    @given(value=st.one_of(
        st.none(),
        st.integers(),
        st.floats(),
        st.text(),
        st.dictionaries(st.text(), st.one_of(st.integers(), st.floats(), st.text())),
        st.lists(st.one_of(st.integers(), st.floats())),
    ))
    def test_safe_int_never_raises_exception(self, value):
        """
        Property test: safe_int should never raise an exception.
        """
        try:
            result = safe_int(value)
            # Result must be either an int or None
            assert result is None or isinstance(result, int)
        except Exception as e:
            pytest.fail(f"safe_int raised exception {type(e).__name__}: {e} for input {value}")
    
    @given(value=valid_numeric_values())
    def test_safe_float_returns_finite_for_valid_inputs(self, value):
        """
        Property test: safe_float should return finite floats for valid numeric inputs.
        """
        result = safe_float(value)
        if result is not None:
            assert isinstance(result, float)
            assert np.isfinite(result)
    
    @given(value=invalid_values())
    def test_safe_float_returns_none_for_invalid_inputs(self, value):
        """
        Property test: safe_float should return None for invalid inputs.
        """
        result = safe_float(value)
        assert result is None
    
    @given(num=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False))
    def test_safe_float_preserves_numeric_values(self, num):
        """
        Property test: safe_float should preserve the value of valid numbers.
        """
        result = safe_float(num)
        assert result is not None
        assert abs(result - num) < 1e-10 or abs((result - num) / num) < 1e-10
    
    @given(num=st.integers(min_value=-1000000, max_value=1000000))
    def test_safe_int_preserves_integer_values(self, num):
        """
        Property test: safe_int should preserve the value of integers.
        """
        result = safe_int(num)
        assert result is not None
        assert result == num
    
    @given(data=st.dictionaries(
        st.sampled_from(["value", "magnitude", "mean", "other"]),
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=1
    ))
    def test_extract_numeric_from_dict_never_raises(self, data):
        """
        Property test: extract_numeric_from_dict should never raise an exception.
        """
        try:
            result = extract_numeric_from_dict(data)
            assert result is None or isinstance(result, float)
            if result is not None:
                assert np.isfinite(result)
        except Exception as e:
            pytest.fail(f"extract_numeric_from_dict raised exception {type(e).__name__}: {e}")
    
    @given(data=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    ))
    def test_extract_numeric_from_list_computes_mean(self, data):
        """
        Property test: extract_numeric_from_list should compute mean of numeric values.
        """
        result = extract_numeric_from_list(data)
        assert result is not None
        expected_mean = np.mean(data)
        assert abs(result - expected_mean) < 1e-10
    
    @given(data=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=20
    ))
    def test_extract_numeric_from_list_never_raises(self, data):
        """
        Property test: extract_numeric_from_list should never raise an exception.
        """
        try:
            result = extract_numeric_from_list(data)
            assert result is None or isinstance(result, float)
            if result is not None:
                assert np.isfinite(result)
        except Exception as e:
            pytest.fail(f"extract_numeric_from_list raised exception {type(e).__name__}: {e}")


class TestSafeConversionUnitTests:
    """Unit tests for specific edge cases and examples."""
    
    def test_safe_float_with_none(self):
        """Test that None returns None."""
        assert safe_float(None) is None
    
    def test_safe_float_with_nan(self):
        """Test that NaN returns None."""
        assert safe_float(float('nan')) is None
    
    def test_safe_float_with_inf(self):
        """Test that infinity returns None."""
        assert safe_float(float('inf')) is None
        assert safe_float(float('-inf')) is None
    
    def test_safe_float_with_integer(self):
        """Test integer conversion."""
        assert safe_float(42) == 42.0
        assert safe_float(-17) == -17.0
    
    def test_safe_float_with_float(self):
        """Test float passthrough."""
        assert safe_float(3.14) == 3.14
        assert safe_float(-2.5) == -2.5
    
    def test_safe_float_with_string(self):
        """Test string conversion."""
        assert safe_float("42") == 42.0
        assert safe_float("3.14") == 3.14
        assert safe_float("-2.5") == -2.5
        assert safe_float("invalid") is None
    
    def test_safe_float_with_dict_value_key(self):
        """Test dictionary with 'value' key."""
        assert safe_float({"value": 3.14}) == 3.14
    
    def test_safe_float_with_dict_magnitude_key(self):
        """Test dictionary with 'magnitude' key."""
        assert safe_float({"magnitude": 2.5, "unit": "eV"}) == 2.5
    
    def test_safe_float_with_dict_mean_key(self):
        """Test dictionary with 'mean' key."""
        assert safe_float({"mean": 1.5, "std": 0.2}) == 1.5
    
    def test_safe_float_with_list(self):
        """Test list mean calculation."""
        assert safe_float([1, 2, 3]) == 2.0
        assert safe_float([1.5, 2.5, 3.5]) == 2.5
    
    def test_safe_float_with_tuple(self):
        """Test tuple mean calculation."""
        assert safe_float((1, 2, 3)) == 2.0
    
    def test_safe_float_with_numpy_array(self):
        """Test numpy array mean calculation."""
        arr = np.array([1.0, 2.0, 3.0])
        assert safe_float(arr) == 2.0
    
    def test_safe_float_with_empty_list(self):
        """Test empty list returns None."""
        assert safe_float([]) is None
    
    def test_safe_int_with_integer(self):
        """Test integer passthrough."""
        assert safe_int(42) == 42
        assert safe_int(-17) == -17
    
    def test_safe_int_with_float(self):
        """Test float to int conversion."""
        assert safe_int(3.7) == 3
        assert safe_int(3.2) == 3
    
    def test_safe_int_with_string(self):
        """Test string to int conversion."""
        assert safe_int("42") == 42
        assert safe_int("3.7") == 3
    
    def test_safe_int_with_none(self):
        """Test None returns None."""
        assert safe_int(None) is None
    
    def test_extract_numeric_from_dict_with_custom_keys(self):
        """Test custom key priority."""
        data = {"custom": 5.0, "value": 3.0}
        result = extract_numeric_from_dict(data, keys=["custom"])
        assert result == 5.0
    
    def test_extract_numeric_from_dict_with_numeric_key(self):
        """Test numeric keys."""
        data = {0: 3.14, 1: 2.71}
        result = extract_numeric_from_dict(data)
        assert result in [3.14, 2.71]
    
    def test_extract_numeric_from_list_with_mixed_types(self):
        """Test list with mixed valid and invalid values."""
        result = extract_numeric_from_list([1, "invalid", 3])
        assert result == 2.0
    
    def test_extract_numeric_from_list_empty(self):
        """Test empty list returns None."""
        assert extract_numeric_from_list([]) is None
