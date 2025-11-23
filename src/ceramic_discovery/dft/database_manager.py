"""Database manager for material properties."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ceramic_discovery.dft.database_schema import (
    DFTCalculation,
    ExperimentalData,
    Material,
    MaterialProperty,
    PropertyProvenance,
    ScreeningResult,
    get_session,
)
from ceramic_discovery.dft.unit_converter import UnitConverter


class DatabaseManager:
    """Manager for material property database operations."""

    def __init__(self, database_url: str):
        """
        Initialize database manager.

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.session = get_session(database_url)
        self.unit_converter = UnitConverter()

    def add_material(
        self,
        material_id: str,
        formula: str,
        base_system: Optional[str] = None,
        dopant_element: Optional[str] = None,
        dopant_concentration: Optional[float] = None,
        source: str = "Materials Project",
    ) -> Material:
        """
        Add a new material to the database.

        Args:
            material_id: Unique material identifier
            formula: Chemical formula
            base_system: Base ceramic system (e.g., 'SiC')
            dopant_element: Dopant element if applicable
            dopant_concentration: Dopant concentration (atomic fraction)
            source: Data source

        Returns:
            Material object
        """
        # Check if material already exists
        existing = (
            self.session.query(Material).filter_by(material_id=material_id).first()
        )
        if existing:
            return existing

        material = Material(
            material_id=material_id,
            formula=formula,
            formula_pretty=formula,
            base_system=base_system,
            dopant_element=dopant_element,
            dopant_concentration=dopant_concentration,
            source=source,
        )

        self.session.add(material)
        self.session.commit()

        return material

    def add_property(
        self,
        material_id: str,
        property_name: str,
        value: float,
        unit: str,
        property_category: str,
        property_tier: int,
        source: str,
        uncertainty: Optional[float] = None,
        quality_score: float = 1.0,
        is_derived: bool = False,
        notes: Optional[str] = None,
    ) -> MaterialProperty:
        """
        Add a property to a material with unit standardization.

        Args:
            material_id: Material identifier
            property_name: Property name
            value: Property value
            unit: Property unit
            property_category: Category (structural, mechanical, etc.)
            property_tier: Tier (1, 2, or 3)
            source: Data source
            uncertainty: Uncertainty value
            quality_score: Quality score (0-1)
            is_derived: Whether property is derived from others
            notes: Additional notes

        Returns:
            MaterialProperty object
        """
        # Get material
        material = (
            self.session.query(Material).filter_by(material_id=material_id).first()
        )
        if not material:
            raise ValueError(f"Material {material_id} not found")

        # Standardize unit
        try:
            standardized_value, standard_unit = self.unit_converter.standardize(
                value, unit, property_name
            )
        except ValueError:
            # If standardization fails, use original values
            standardized_value = value
            standard_unit = unit

        # Create property
        prop = MaterialProperty(
            material_id=material.id,
            property_name=property_name,
            property_category=property_category,
            property_tier=property_tier,
            value=standardized_value,
            unit=unit,
            unit_standardized=standard_unit,
            uncertainty=uncertainty,
            source=source,
            quality_score=quality_score,
            is_derived=is_derived,
            notes=notes,
        )

        self.session.add(prop)
        self.session.commit()

        return prop

    def add_dft_calculation(
        self, material_id: str, calculation_data: Dict[str, Any]
    ) -> DFTCalculation:
        """
        Add DFT calculation results to database.

        Args:
            material_id: Material identifier
            calculation_data: Dictionary with DFT results

        Returns:
            DFTCalculation object
        """
        # Get material
        material = (
            self.session.query(Material).filter_by(material_id=material_id).first()
        )
        if not material:
            raise ValueError(f"Material {material_id} not found")

        # Create DFT calculation record
        dft_calc = DFTCalculation(
            material_id=material.id,
            formation_energy_per_atom=calculation_data.get("formation_energy_per_atom"),
            energy_above_hull=calculation_data.get("energy_above_hull"),
            decomposition_energy=calculation_data.get("decomposition_energy"),
            is_stable=calculation_data.get("is_stable"),
            band_gap=calculation_data.get("band_gap"),
            is_metal=calculation_data.get("is_metal"),
            is_magnetic=calculation_data.get("is_magnetic"),
            total_magnetization=calculation_data.get("total_magnetization"),
            efermi=calculation_data.get("efermi"),
            crystal_structure=calculation_data.get("crystal_structure"),
            space_group_number=calculation_data.get("space_group_number"),
            nsites=calculation_data.get("nsites"),
            volume=calculation_data.get("volume"),
            density=calculation_data.get("density"),
            bulk_modulus_vrh=calculation_data.get("bulk_modulus_vrh"),
            shear_modulus_vrh=calculation_data.get("shear_modulus_vrh"),
            elastic_tensor=calculation_data.get("elastic_tensor"),
            functional=calculation_data.get("functional"),
            pseudopotential=calculation_data.get("pseudopotential"),
            calculation_id=calculation_data.get("calculation_id"),
            source_database=calculation_data.get("source_database", "Materials Project"),
            retrieved_at=datetime.utcnow(),
        )

        self.session.add(dft_calc)
        self.session.commit()

        return dft_calc

    def get_material_properties(
        self, material_id: str, property_names: Optional[List[str]] = None
    ) -> List[MaterialProperty]:
        """
        Get properties for a material.

        Args:
            material_id: Material identifier
            property_names: Optional list of specific properties to retrieve

        Returns:
            List of MaterialProperty objects
        """
        material = (
            self.session.query(Material).filter_by(material_id=material_id).first()
        )
        if not material:
            return []

        query = self.session.query(MaterialProperty).filter_by(material_id=material.id)

        if property_names:
            query = query.filter(MaterialProperty.property_name.in_(property_names))

        return query.all()

    def update_property(
        self,
        material_id: str,
        property_name: str,
        new_value: float,
        change_reason: str,
        changed_by: str = "system",
    ) -> MaterialProperty:
        """
        Update a property value with provenance tracking.

        Args:
            material_id: Material identifier
            property_name: Property name
            new_value: New property value
            change_reason: Reason for change
            changed_by: Who made the change

        Returns:
            Updated MaterialProperty object
        """
        # Get material
        material = (
            self.session.query(Material).filter_by(material_id=material_id).first()
        )
        if not material:
            raise ValueError(f"Material {material_id} not found")

        # Get property
        prop = (
            self.session.query(MaterialProperty)
            .filter_by(material_id=material.id, property_name=property_name)
            .first()
        )
        if not prop:
            raise ValueError(f"Property {property_name} not found for material {material_id}")

        # Record provenance
        provenance = PropertyProvenance(
            property_id=prop.id,
            version=prop.version,
            previous_value=prop.value,
            new_value=new_value,
            change_reason=change_reason,
            changed_by=changed_by,
        )

        # Update property
        prop.value = new_value
        prop.version += 1
        prop.updated_at = datetime.utcnow()

        self.session.add(provenance)
        self.session.commit()

        return prop

    def search_materials(
        self,
        base_system: Optional[str] = None,
        dopant_element: Optional[str] = None,
        max_energy_above_hull: Optional[float] = None,
        min_quality_score: Optional[float] = None,
    ) -> List[Material]:
        """
        Search for materials by criteria.

        Args:
            base_system: Base ceramic system
            dopant_element: Dopant element
            max_energy_above_hull: Maximum energy above hull
            min_quality_score: Minimum quality score

        Returns:
            List of Material objects
        """
        query = self.session.query(Material)

        if base_system:
            query = query.filter_by(base_system=base_system)

        if dopant_element:
            query = query.filter_by(dopant_element=dopant_element)

        if max_energy_above_hull is not None:
            # Join with DFT calculations
            query = query.join(DFTCalculation).filter(
                DFTCalculation.energy_above_hull <= max_energy_above_hull
            )

        materials = query.all()

        # Filter by quality score if specified
        if min_quality_score is not None:
            filtered_materials = []
            for material in materials:
                avg_quality = self._calculate_average_quality(material)
                if avg_quality >= min_quality_score:
                    filtered_materials.append(material)
            return filtered_materials

        return materials

    def _calculate_average_quality(self, material: Material) -> float:
        """Calculate average quality score for a material's properties."""
        properties = material.properties
        if not properties:
            return 0.0

        quality_scores = [p.quality_score for p in properties if p.quality_score is not None]
        if not quality_scores:
            return 0.0

        return sum(quality_scores) / len(quality_scores)

    def add_experimental_data(
        self,
        formula: str,
        property_name: str,
        measured_value: float,
        unit: str,
        citation: str,
        uncertainty: Optional[float] = None,
        temperature: Optional[float] = None,
        doi: Optional[str] = None,
    ) -> ExperimentalData:
        """
        Add experimental validation data.

        Args:
            formula: Material formula
            property_name: Property name
            measured_value: Measured value
            unit: Measurement unit
            citation: Literature citation
            uncertainty: Measurement uncertainty
            temperature: Measurement temperature (K)
            doi: DOI reference

        Returns:
            ExperimentalData object
        """
        exp_data = ExperimentalData(
            material_formula=formula,
            property_name=property_name,
            measured_value=measured_value,
            unit=unit,
            uncertainty=uncertainty,
            temperature=temperature,
            citation=citation,
            doi=doi,
        )

        self.session.add(exp_data)
        self.session.commit()

        return exp_data

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_materials": self.session.query(Material).count(),
            "total_properties": self.session.query(MaterialProperty).count(),
            "total_dft_calculations": self.session.query(DFTCalculation).count(),
            "total_experimental_data": self.session.query(ExperimentalData).count(),
            "materials_by_system": {},
        }

        # Count materials by base system
        materials = self.session.query(Material).all()
        for material in materials:
            if material.base_system:
                stats["materials_by_system"][material.base_system] = (
                    stats["materials_by_system"].get(material.base_system, 0) + 1
                )

        return stats

    def close(self) -> None:
        """Close database session."""
        self.session.close()

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
