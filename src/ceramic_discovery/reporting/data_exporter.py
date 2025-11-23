"""Data export system for research outputs."""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ..validation.reproducibility import (
    ExperimentSnapshot,
    ReproducibilityFramework
)


class DataExporter:
    """
    Data export system for research outputs.
    
    Provides standardized data export with:
    - Multiple format support (CSV, JSON, HDF5, Excel)
    - Data anonymization for public release
    - Reproducibility packages with complete environment
    - Automated documentation generation
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Base directory for exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dataframe(
        self,
        data: pd.DataFrame,
        filename: str,
        format: str = 'csv',
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export DataFrame to file.
        
        Args:
            data: DataFrame to export
            filename: Output filename (without extension)
            format: Export format ('csv', 'json', 'excel', 'hdf5', 'parquet')
            metadata: Optional metadata to include
        
        Returns:
            Path to exported file
        """
        if format == 'csv':
            output_path = self.output_dir / f"{filename}.csv"
            data.to_csv(output_path, index=False)
            
            # Save metadata separately if provided
            if metadata:
                meta_path = self.output_dir / f"{filename}_metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
        
        elif format == 'json':
            output_path = self.output_dir / f"{filename}.json"
            export_data = {
                'data': data.to_dict(orient='records'),
                'metadata': metadata or {}
            }
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == 'excel':
            output_path = self.output_dir / f"{filename}.xlsx"
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Data', index=False)
                
                # Add metadata sheet if provided
                if metadata:
                    meta_df = pd.DataFrame([metadata])
                    meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        elif format == 'hdf5':
            output_path = self.output_dir / f"{filename}.h5"
            data.to_hdf(output_path, key='data', mode='w')
            
            # Save metadata separately
            if metadata:
                meta_path = self.output_dir / f"{filename}_metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
        
        elif format == 'parquet':
            output_path = self.output_dir / f"{filename}.parquet"
            data.to_parquet(output_path, index=False)
            
            # Save metadata separately
            if metadata:
                meta_path = self.output_dir / f"{filename}_metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path
    
    def export_multiple_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        filename: str,
        format: str = 'excel',
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export multiple datasets to a single file.
        
        Args:
            datasets: Dictionary mapping sheet/table names to DataFrames
            filename: Output filename (without extension)
            format: Export format ('excel', 'hdf5')
            metadata: Optional metadata to include
        
        Returns:
            Path to exported file
        """
        if format == 'excel':
            output_path = self.output_dir / f"{filename}.xlsx"
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, data in datasets.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add metadata sheet if provided
                if metadata:
                    meta_df = pd.DataFrame([metadata])
                    meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        elif format == 'hdf5':
            output_path = self.output_dir / f"{filename}.h5"
            for key, data in datasets.items():
                data.to_hdf(output_path, key=key, mode='a')
            
            # Save metadata separately
            if metadata:
                meta_path = self.output_dir / f"{filename}_metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Format {format} not supported for multiple datasets")
        
        return output_path
    
    def anonymize_data(
        self,
        data: pd.DataFrame,
        sensitive_columns: List[str],
        method: str = 'remove'
    ) -> pd.DataFrame:
        """
        Anonymize sensitive data for public release.
        
        Args:
            data: DataFrame to anonymize
            sensitive_columns: List of column names containing sensitive data
            method: Anonymization method ('remove', 'hash', 'mask')
        
        Returns:
            Anonymized DataFrame
        """
        anonymized = data.copy()
        
        for col in sensitive_columns:
            if col not in anonymized.columns:
                continue
            
            if method == 'remove':
                anonymized = anonymized.drop(columns=[col])
            
            elif method == 'hash':
                # Hash values for anonymization
                anonymized[col] = anonymized[col].apply(
                    lambda x: self._hash_value(str(x))
                )
            
            elif method == 'mask':
                # Mask with generic placeholder
                anonymized[col] = '[REDACTED]'
            
            else:
                raise ValueError(f"Unknown anonymization method: {method}")
        
        return anonymized
    
    def _hash_value(self, value: str) -> str:
        """Hash a value for anonymization."""
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def create_reproducibility_package(
        self,
        experiment_id: str,
        experiment_snapshot: ExperimentSnapshot,
        data_files: Optional[List[Path]] = None,
        code_files: Optional[List[Path]] = None,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Create complete reproducibility package.
        
        Args:
            experiment_id: Experiment identifier
            experiment_snapshot: Experiment snapshot with full provenance
            data_files: List of data files to include
            code_files: List of code files to include
            output_filename: Output filename (default: experiment_id.zip)
        
        Returns:
            Path to reproducibility package (ZIP file)
        """
        if output_filename is None:
            output_filename = f"{experiment_id}_reproducibility_package.zip"
        
        package_path = self.output_dir / output_filename
        
        # Create temporary directory for package contents
        temp_dir = self.output_dir / f"temp_{experiment_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save experiment snapshot
            snapshot_path = temp_dir / "experiment_snapshot.json"
            experiment_snapshot.save(snapshot_path)
            
            # Create requirements.txt
            requirements_path = temp_dir / "requirements.txt"
            with open(requirements_path, 'w') as f:
                for dep in experiment_snapshot.environment.dependencies:
                    f.write(f"{dep}\n")
            
            # Create environment.yml for conda
            env_path = temp_dir / "environment.yml"
            env_spec = {
                'name': f'ceramic_discovery_{experiment_id}',
                'channels': ['conda-forge', 'defaults'],
                'dependencies': [
                    f'python={experiment_snapshot.environment.python_version.split()[0]}',
                    'pip',
                    {'pip': [str(dep) for dep in experiment_snapshot.environment.dependencies[:20]]}
                ]
            }
            with open(env_path, 'w') as f:
                yaml.dump(env_spec, f)
            
            # Copy data files
            if data_files:
                data_dir = temp_dir / "data"
                data_dir.mkdir(exist_ok=True)
                for data_file in data_files:
                    if Path(data_file).exists():
                        shutil.copy2(data_file, data_dir / Path(data_file).name)
            
            # Copy code files
            if code_files:
                code_dir = temp_dir / "code"
                code_dir.mkdir(exist_ok=True)
                for code_file in code_files:
                    if Path(code_file).exists():
                        shutil.copy2(code_file, code_dir / Path(code_file).name)
            
            # Create README
            readme_path = temp_dir / "README.md"
            self._create_reproducibility_readme(
                readme_path,
                experiment_id,
                experiment_snapshot
            )
            
            # Create ZIP archive
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)
        
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        return package_path

    
    def _create_reproducibility_readme(
        self,
        readme_path: Path,
        experiment_id: str,
        snapshot: ExperimentSnapshot
    ) -> None:
        """Create README for reproducibility package."""
        content = [
            f"# Reproducibility Package: {experiment_id}",
            "",
            "This package contains all necessary information to reproduce the computational experiment.",
            "",
            "## Experiment Information",
            "",
            f"- **Experiment ID**: {experiment_id}",
            f"- **Type**: {snapshot.parameters.experiment_type}",
            f"- **Timestamp**: {snapshot.parameters.timestamp}",
            f"- **Random Seed**: {snapshot.parameters.random_seed}",
            "",
            "## Environment",
            "",
            f"- **Python Version**: {snapshot.environment.python_version}",
            f"- **Platform**: {snapshot.environment.platform} {snapshot.environment.platform_version}",
            f"- **Processor**: {snapshot.environment.processor}",
            "",
            "## Setup Instructions",
            "",
            "### Option 1: Using Conda (Recommended)",
            "",
            "```bash",
            "# Create environment from specification",
            "conda env create -f environment.yml",
            "",
            "# Activate environment",
            f"conda activate ceramic_discovery_{experiment_id}",
            "```",
            "",
            "### Option 2: Using pip",
            "",
            "```bash",
            "# Create virtual environment",
            "python -m venv venv",
            "",
            "# Activate virtual environment",
            "# On Windows:",
            "venv\\Scripts\\activate",
            "# On Unix/MacOS:",
            "source venv/bin/activate",
            "",
            "# Install dependencies",
            "pip install -r requirements.txt",
            "```",
            "",
            "## Reproduction Instructions",
            "",
            "1. Set up the environment using one of the methods above",
            "2. Load the experiment snapshot:",
            "",
            "```python",
            "from ceramic_discovery.validation.reproducibility import ExperimentSnapshot",
            "",
            "# Load snapshot",
            "snapshot = ExperimentSnapshot.load('experiment_snapshot.json')",
            "",
            "# Access parameters",
            "parameters = snapshot.parameters.parameters",
            "random_seed = snapshot.parameters.random_seed",
            "```",
            "",
            "3. Rerun the experiment using the same parameters and random seed",
            "4. Compare results with those in the snapshot",
            "",
            "## Experiment Parameters",
            "",
            "```json",
            json.dumps(snapshot.parameters.parameters, indent=2),
            "```",
            "",
            "## Data Provenance",
            "",
            f"- **Source Data**: {snapshot.provenance.source_data}",
            f"- **Transformations**: {len(snapshot.provenance.transformations)}",
            "",
            "### Transformation Steps",
            "",
        ]
        
        for i, transform in enumerate(snapshot.provenance.transformations, 1):
            content.append(f"{i}. {transform}")
        
        content.extend([
            "",
            "## Results",
            "",
            "Original experiment results are included in the snapshot file.",
            "",
            "## Citation",
            "",
            "If you use this reproducibility package, please cite:",
            "",
            "```",
            f"Ceramic Armor Discovery Framework - Experiment {experiment_id}",
            f"Generated: {snapshot.parameters.timestamp}",
            "```",
            "",
            "## Contact",
            "",
            "For questions about this reproducibility package, please contact the authors.",
            "",
            "## License",
            "",
            "This reproducibility package is provided for research purposes.",
        ])
        
        with open(readme_path, 'w') as f:
            f.write('\n'.join(content))
    
    def generate_data_dictionary(
        self,
        data: pd.DataFrame,
        column_descriptions: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
        output_filename: str = "data_dictionary"
    ) -> Path:
        """
        Generate data dictionary documentation.
        
        Args:
            data: DataFrame to document
            column_descriptions: Dictionary mapping column names to descriptions
            units: Dictionary mapping column names to units
            output_filename: Output filename (without extension)
        
        Returns:
            Path to data dictionary file
        """
        column_descriptions = column_descriptions or {}
        units = units or {}
        
        # Analyze data types and statistics
        dictionary_data = []
        
        for col in data.columns:
            col_info = {
                'Column': col,
                'Type': str(data[col].dtype),
                'Description': column_descriptions.get(col, ''),
                'Units': units.get(col, ''),
                'Non-Null Count': data[col].notna().sum(),
                'Null Count': data[col].isna().sum(),
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(data[col]):
                col_info.update({
                    'Min': f"{data[col].min():.3f}" if not pd.isna(data[col].min()) else '',
                    'Max': f"{data[col].max():.3f}" if not pd.isna(data[col].max()) else '',
                    'Mean': f"{data[col].mean():.3f}" if not pd.isna(data[col].mean()) else '',
                    'Std': f"{data[col].std():.3f}" if not pd.isna(data[col].std()) else '',
                })
            else:
                # For categorical columns, show unique values
                n_unique = data[col].nunique()
                col_info['Unique Values'] = n_unique
                if n_unique <= 10:
                    col_info['Values'] = ', '.join(str(v) for v in data[col].unique()[:10])
            
            dictionary_data.append(col_info)
        
        dictionary_df = pd.DataFrame(dictionary_data)
        
        # Export as Excel with formatting
        output_path = self.output_dir / f"{output_filename}.xlsx"
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            dictionary_df.to_excel(writer, sheet_name='Data Dictionary', index=False)
            
            # Add summary sheet
            summary_data = {
                'Total Columns': len(data.columns),
                'Total Rows': len(data),
                'Numeric Columns': len(data.select_dtypes(include=[np.number]).columns),
                'Categorical Columns': len(data.select_dtypes(include=['object', 'category']).columns),
                'Total Missing Values': data.isna().sum().sum(),
                'Generated': datetime.utcnow().isoformat(),
            }
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return output_path
    
    def export_analysis_results(
        self,
        results: Dict[str, Any],
        output_filename: str = "analysis_results",
        format: str = 'json'
    ) -> Path:
        """
        Export analysis results with proper formatting.
        
        Args:
            results: Dictionary containing analysis results
            output_filename: Output filename (without extension)
            format: Export format ('json', 'yaml')
        
        Returns:
            Path to exported file
        """
        # Convert numpy types to native Python types
        serializable_results = self._make_serializable(results)
        
        if format == 'json':
            output_path = self.output_dir / f"{output_filename}.json"
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
        
        elif format == 'yaml':
            output_path = self.output_dir / f"{output_filename}.yaml"
            with open(output_path, 'w') as f:
                yaml.dump(serializable_results, f, default_flow_style=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def create_publication_dataset(
        self,
        data: pd.DataFrame,
        metadata: Dict[str, Any],
        column_descriptions: Dict[str, str],
        units: Dict[str, str],
        license: str = "CC BY 4.0",
        doi: Optional[str] = None,
        output_filename: str = "publication_dataset"
    ) -> Path:
        """
        Create publication-ready dataset with complete documentation.
        
        Args:
            data: DataFrame to publish
            metadata: Dataset metadata (title, authors, description, etc.)
            column_descriptions: Descriptions for each column
            units: Units for each column
            license: Data license
            doi: Digital Object Identifier (if available)
            output_filename: Output filename (without extension)
        
        Returns:
            Path to publication dataset package (ZIP file)
        """
        # Create temporary directory
        temp_dir = self.output_dir / f"temp_{output_filename}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Export main data
            data_path = temp_dir / "data.csv"
            data.to_csv(data_path, index=False)
            
            # Create data dictionary
            dict_data = []
            for col in data.columns:
                dict_data.append({
                    'Column': col,
                    'Description': column_descriptions.get(col, ''),
                    'Units': units.get(col, ''),
                    'Type': str(data[col].dtype),
                })
            
            dict_df = pd.DataFrame(dict_data)
            dict_path = temp_dir / "data_dictionary.csv"
            dict_df.to_csv(dict_path, index=False)
            
            # Create metadata file
            full_metadata = {
                **metadata,
                'license': license,
                'doi': doi,
                'generated': datetime.utcnow().isoformat(),
                'n_rows': len(data),
                'n_columns': len(data.columns),
                'columns': list(data.columns),
            }
            
            metadata_path = temp_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            # Create README
            readme_path = temp_dir / "README.md"
            self._create_publication_readme(
                readme_path,
                metadata,
                license,
                doi
            )
            
            # Create LICENSE file
            license_path = temp_dir / "LICENSE.txt"
            with open(license_path, 'w') as f:
                f.write(f"This dataset is licensed under {license}\n\n")
                if license == "CC BY 4.0":
                    f.write("Creative Commons Attribution 4.0 International License\n")
                    f.write("https://creativecommons.org/licenses/by/4.0/\n")
            
            # Create ZIP archive
            package_path = self.output_dir / f"{output_filename}.zip"
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)
        
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        return package_path
    
    def _create_publication_readme(
        self,
        readme_path: Path,
        metadata: Dict[str, Any],
        license: str,
        doi: Optional[str]
    ) -> None:
        """Create README for publication dataset."""
        content = [
            f"# {metadata.get('title', 'Dataset')}",
            "",
            "## Description",
            "",
            metadata.get('description', 'No description provided.'),
            "",
            "## Authors",
            "",
        ]
        
        authors = metadata.get('authors', [])
        for author in authors:
            content.append(f"- {author}")
        
        content.extend([
            "",
            "## Files",
            "",
            "- `data.csv`: Main dataset",
            "- `data_dictionary.csv`: Column descriptions and metadata",
            "- `metadata.json`: Complete dataset metadata",
            "- `LICENSE.txt`: License information",
            "",
            "## Usage",
            "",
            "```python",
            "import pandas as pd",
            "",
            "# Load data",
            "data = pd.read_csv('data.csv')",
            "",
            "# Load data dictionary",
            "data_dict = pd.read_csv('data_dictionary.csv')",
            "```",
            "",
            "## License",
            "",
            f"This dataset is licensed under {license}.",
            "",
        ])
        
        if doi:
            content.extend([
                "## Citation",
                "",
                f"DOI: {doi}",
                "",
            ])
        
        content.extend([
            "## Contact",
            "",
            "For questions about this dataset, please contact the authors.",
        ])
        
        with open(readme_path, 'w') as f:
            f.write('\n'.join(content))
