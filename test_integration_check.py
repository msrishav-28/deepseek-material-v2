"""Quick integration check for SiC Alloy Designer components."""

import sys
import traceback

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from ceramic_discovery.dft import JarvisClient
        print("✓ JarvisClient imported")
    except Exception as e:
        print(f"✗ JarvisClient import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.dft import NISTClient
        print("✓ NISTClient imported")
    except Exception as e:
        print(f"✗ NISTClient import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.dft import LiteratureDatabase
        print("✓ LiteratureDatabase imported")
    except Exception as e:
        print(f"✗ LiteratureDatabase import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.dft import DataCombiner
        print("✓ DataCombiner imported")
    except Exception as e:
        print(f"✗ DataCombiner import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.ml import CompositionDescriptorCalculator
        print("✓ CompositionDescriptorCalculator imported")
    except Exception as e:
        print(f"✗ CompositionDescriptorCalculator import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.ml import StructureDescriptorCalculator
        print("✓ StructureDescriptorCalculator imported")
    except Exception as e:
        print(f"✗ StructureDescriptorCalculator import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.screening import ApplicationRanker
        print("✓ ApplicationRanker imported")
    except Exception as e:
        print(f"✗ ApplicationRanker import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.validation import ExperimentalPlanner
        print("✓ ExperimentalPlanner imported")
    except Exception as e:
        print(f"✗ ExperimentalPlanner import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.utils import safe_float, safe_int
        print("✓ Safe conversion utilities imported")
    except Exception as e:
        print(f"✗ Safe conversion utilities import failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.screening import SiCAlloyDesignerPipeline
        print("✓ SiCAlloyDesignerPipeline imported")
    except Exception as e:
        print(f"✗ SiCAlloyDesignerPipeline import failed: {e}")
        traceback.print_exc()

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        from ceramic_discovery.utils import safe_float
        assert safe_float(42) == 42.0
        assert safe_float("3.14") == 3.14
        assert safe_float(None) is None
        print("✓ safe_float works correctly")
    except Exception as e:
        print(f"✗ safe_float test failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.ml import CompositionDescriptorCalculator
        calc = CompositionDescriptorCalculator()
        descriptors = calc.calculate_descriptors("SiC")
        assert "n_elements" in descriptors
        assert descriptors["n_elements"] == 2
        print("✓ CompositionDescriptorCalculator works correctly")
    except Exception as e:
        print(f"✗ CompositionDescriptorCalculator test failed: {e}")
        traceback.print_exc()
        
    try:
        from ceramic_discovery.screening import ApplicationRanker
        ranker = ApplicationRanker()
        apps = ranker.list_applications()
        assert len(apps) > 0
        print(f"✓ ApplicationRanker works correctly ({len(apps)} applications)")
    except Exception as e:
        print(f"✗ ApplicationRanker test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("SiC Alloy Designer Integration Check")
    print("=" * 60)
    
    test_imports()
    test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("Integration check complete!")
    print("=" * 60)
