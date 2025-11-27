"""Experimental validation planning for material synthesis and characterization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SynthesisMethodType(Enum):
    """Types of synthesis methods."""
    
    HOT_PRESSING = "Hot Pressing"
    SOLID_STATE_SINTERING = "Solid State Sintering"
    SPARK_PLASMA_SINTERING = "Spark Plasma Sintering"


class DifficultyLevel(Enum):
    """Difficulty levels for synthesis."""
    
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


@dataclass
class SynthesisMethod:
    """
    Synthesis method specification.
    
    Attributes:
        method: Synthesis method name
        temperature_K: Processing temperature in Kelvin
        pressure_MPa: Processing pressure in MPa
        time_hours: Processing time in hours
        difficulty: Difficulty level (Easy, Medium, Hard)
        cost_factor: Relative cost factor (1.0 = baseline)
        precursors: List of required precursor materials
        equipment: Required equipment
        notes: Additional notes or considerations
    """
    
    method: str
    temperature_K: float
    pressure_MPa: float
    time_hours: float
    difficulty: str
    cost_factor: float
    precursors: List[str] = field(default_factory=list)
    equipment: str = ""
    notes: str = ""


@dataclass
class CharacterizationTechnique:
    """
    Characterization technique specification.
    
    Attributes:
        technique: Technique name
        purpose: What property/feature is being measured
        estimated_time_hours: Estimated time in hours
        estimated_cost_dollars: Estimated cost in dollars
        sample_requirements: Sample preparation requirements
        equipment: Required equipment
        notes: Additional notes
    """
    
    technique: str
    purpose: str
    estimated_time_hours: float
    estimated_cost_dollars: float
    sample_requirements: str = ""
    equipment: str = ""
    notes: str = ""


@dataclass
class ResourceEstimate:
    """
    Resource estimation for experimental work.
    
    Attributes:
        timeline_months: Estimated timeline in months
        estimated_cost_k_dollars: Estimated cost in thousands of dollars
        required_equipment: List of required equipment
        required_expertise: List of required expertise areas
        risk_factors: Identified risk factors
        confidence_level: Confidence in estimates (0-1)
    """
    
    timeline_months: float
    estimated_cost_k_dollars: float
    required_equipment: List[str] = field(default_factory=list)
    required_expertise: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    confidence_level: float = 0.8


class ExperimentalPlanner:
    """
    Experimental validation planner for material synthesis and characterization.
    
    Generates experimental design recommendations including:
    - Synthesis protocols with multiple methods
    - Characterization plans with appropriate techniques
    - Resource estimates (time, cost, equipment)
    """
    
    def __init__(self):
        """Initialize experimental planner."""
        self._initialize_synthesis_database()
        self._initialize_characterization_database()
    
    def _initialize_synthesis_database(self) -> None:
        """Initialize database of synthesis method templates."""
        self.synthesis_templates = {
            SynthesisMethodType.HOT_PRESSING: {
                "equipment": "Hot press with graphite dies",
                "typical_temp_range": (1800, 2200),
                "typical_pressure_range": (20, 50),
                "typical_time_range": (0.5, 2.0),
                "difficulty": DifficultyLevel.MEDIUM,
                "cost_factor": 1.0,
                "advantages": "Good density, relatively fast",
                "disadvantages": "Limited shape complexity"
            },
            SynthesisMethodType.SOLID_STATE_SINTERING: {
                "equipment": "High-temperature furnace",
                "typical_temp_range": (1700, 2300),
                "typical_pressure_range": (0, 0),  # Pressureless
                "typical_time_range": (2.0, 8.0),
                "difficulty": DifficultyLevel.EASY,
                "cost_factor": 0.6,
                "advantages": "Simple, low cost, scalable",
                "disadvantages": "Longer time, may need sintering aids"
            },
            SynthesisMethodType.SPARK_PLASMA_SINTERING: {
                "equipment": "SPS/FAST system",
                "typical_temp_range": (1600, 2100),
                "typical_pressure_range": (30, 80),
                "typical_time_range": (0.1, 0.5),
                "difficulty": DifficultyLevel.HARD,
                "cost_factor": 1.5,
                "advantages": "Very fast, fine microstructure",
                "disadvantages": "Expensive, limited sample size"
            }
        }
    
    def _initialize_characterization_database(self) -> None:
        """Initialize database of characterization techniques."""
        self.characterization_techniques = {
            "XRD": {
                "purpose": "Phase identification and crystal structure",
                "time_hours": 2.0,
                "cost_dollars": 200,
                "equipment": "X-ray diffractometer",
                "sample_prep": "Polished surface or powder",
                "priority": "essential"
            },
            "SEM": {
                "purpose": "Microstructure and morphology",
                "time_hours": 3.0,
                "cost_dollars": 300,
                "equipment": "Scanning electron microscope",
                "sample_prep": "Polished and coated sample",
                "priority": "essential"
            },
            "TEM": {
                "purpose": "Atomic structure and defects",
                "time_hours": 8.0,
                "cost_dollars": 1000,
                "equipment": "Transmission electron microscope",
                "sample_prep": "Thin foil preparation",
                "priority": "optional"
            },
            "Hardness Testing": {
                "purpose": "Vickers hardness measurement",
                "time_hours": 1.0,
                "cost_dollars": 150,
                "equipment": "Vickers hardness tester",
                "sample_prep": "Polished surface",
                "priority": "essential"
            },
            "Fracture Toughness": {
                "purpose": "Fracture toughness (KIC)",
                "time_hours": 2.0,
                "cost_dollars": 300,
                "equipment": "Indentation or SENB testing",
                "sample_prep": "Polished sample or machined bars",
                "priority": "important"
            },
            "Thermal Conductivity": {
                "purpose": "Thermal conductivity measurement",
                "time_hours": 4.0,
                "cost_dollars": 500,
                "equipment": "Laser flash or hot disk system",
                "sample_prep": "Disk sample with specific dimensions",
                "priority": "important"
            },
            "Density Measurement": {
                "purpose": "Bulk density and porosity",
                "time_hours": 0.5,
                "cost_dollars": 50,
                "equipment": "Archimedes method or pycnometer",
                "sample_prep": "Clean sample",
                "priority": "essential"
            },
            "XPS": {
                "purpose": "Surface chemistry and oxidation states",
                "time_hours": 4.0,
                "cost_dollars": 600,
                "equipment": "X-ray photoelectron spectrometer",
                "sample_prep": "Clean surface",
                "priority": "optional"
            },
            "Raman Spectroscopy": {
                "purpose": "Phase identification and stress analysis",
                "time_hours": 2.0,
                "cost_dollars": 250,
                "equipment": "Raman spectrometer",
                "sample_prep": "Polished surface",
                "priority": "optional"
            },
            "Thermal Analysis": {
                "purpose": "Thermal stability (TGA/DSC)",
                "time_hours": 3.0,
                "cost_dollars": 300,
                "equipment": "TGA/DSC system",
                "sample_prep": "Powder or small sample",
                "priority": "optional"
            }
        }
    
    def design_synthesis_protocol(
        self,
        formula: str,
        properties: Dict[str, float]
    ) -> List[SynthesisMethod]:
        """
        Design synthesis protocols for a material.
        
        Recommends 3 synthesis methods based on material properties:
        - Hot pressing (standard method)
        - Solid-state sintering (cost-effective)
        - Spark plasma sintering (high-quality)
        
        Args:
            formula: Chemical formula
            properties: Dictionary of material properties
        
        Returns:
            List of SynthesisMethod objects
        """
        synthesis_methods = []
        
        # Extract relevant properties
        melting_point = properties.get("melting_point", properties.get("melting_point_K", 2500))
        density = properties.get("density", properties.get("density_g_cm3", 3.0))
        
        # Determine precursors from formula
        precursors = self._determine_precursors(formula)
        
        # Method 1: Hot Pressing
        hp_template = self.synthesis_templates[SynthesisMethodType.HOT_PRESSING]
        hp_temp = min(melting_point * 0.75, hp_template["typical_temp_range"][1])
        hp_temp = max(hp_temp, hp_template["typical_temp_range"][0])
        
        hot_pressing = SynthesisMethod(
            method=SynthesisMethodType.HOT_PRESSING.value,
            temperature_K=hp_temp,
            pressure_MPa=35.0,
            time_hours=1.0,
            difficulty=hp_template["difficulty"].value,
            cost_factor=hp_template["cost_factor"],
            precursors=precursors,
            equipment=hp_template["equipment"],
            notes=f"{hp_template['advantages']}. {hp_template['disadvantages']}"
        )
        synthesis_methods.append(hot_pressing)
        
        # Method 2: Solid State Sintering
        ss_template = self.synthesis_templates[SynthesisMethodType.SOLID_STATE_SINTERING]
        ss_temp = min(melting_point * 0.80, ss_template["typical_temp_range"][1])
        ss_temp = max(ss_temp, ss_template["typical_temp_range"][0])
        
        solid_state = SynthesisMethod(
            method=SynthesisMethodType.SOLID_STATE_SINTERING.value,
            temperature_K=ss_temp,
            pressure_MPa=0.0,
            time_hours=4.0,
            difficulty=ss_template["difficulty"].value,
            cost_factor=ss_template["cost_factor"],
            precursors=precursors,
            equipment=ss_template["equipment"],
            notes=f"{ss_template['advantages']}. {ss_template['disadvantages']}"
        )
        synthesis_methods.append(solid_state)
        
        # Method 3: Spark Plasma Sintering
        sps_template = self.synthesis_templates[SynthesisMethodType.SPARK_PLASMA_SINTERING]
        sps_temp = min(melting_point * 0.70, sps_template["typical_temp_range"][1])
        sps_temp = max(sps_temp, sps_template["typical_temp_range"][0])
        
        spark_plasma = SynthesisMethod(
            method=SynthesisMethodType.SPARK_PLASMA_SINTERING.value,
            temperature_K=sps_temp,
            pressure_MPa=50.0,
            time_hours=0.25,
            difficulty=sps_template["difficulty"].value,
            cost_factor=sps_template["cost_factor"],
            precursors=precursors,
            equipment=sps_template["equipment"],
            notes=f"{sps_template['advantages']}. {sps_template['disadvantages']}"
        )
        synthesis_methods.append(spark_plasma)
        
        logger.info(f"Generated {len(synthesis_methods)} synthesis protocols for {formula}")
        
        return synthesis_methods
    
    def _determine_precursors(self, formula: str) -> List[str]:
        """
        Determine likely precursor materials from formula.
        
        Args:
            formula: Chemical formula
        
        Returns:
            List of precursor material names
        """
        precursors = []
        
        # Common element to precursor mapping
        precursor_map = {
            "Si": "Silicon powder or SiO2",
            "C": "Carbon black or graphite",
            "Ti": "Titanium powder or TiO2",
            "Zr": "Zirconium powder or ZrO2",
            "Hf": "Hafnium powder or HfO2",
            "B": "Boron powder or B2O3",
            "Al": "Aluminum powder or Al2O3",
            "N": "Nitrogen gas or metal nitrides",
            "O": "Oxygen or metal oxides"
        }
        
        # Extract elements from formula (simple approach)
        for element, precursor in precursor_map.items():
            if element in formula:
                precursors.append(precursor)
        
        return precursors if precursors else ["Metal powders", "Carbon source"]
    
    def design_characterization_plan(
        self,
        formula: str,
        target_properties: List[str]
    ) -> List[CharacterizationTechnique]:
        """
        Design characterization plan for a material.
        
        Recommends characterization techniques based on target properties.
        Always includes essential techniques (XRD, SEM, density, hardness).
        Adds additional techniques based on target properties.
        
        Args:
            formula: Chemical formula
            target_properties: List of target property names
        
        Returns:
            List of CharacterizationTechnique objects
        """
        techniques = []
        
        # Essential techniques (always included)
        essential_techniques = ["XRD", "SEM", "Density Measurement", "Hardness Testing"]
        
        for tech_name in essential_techniques:
            if tech_name in self.characterization_techniques:
                tech_data = self.characterization_techniques[tech_name]
                technique = CharacterizationTechnique(
                    technique=tech_name,
                    purpose=tech_data["purpose"],
                    estimated_time_hours=tech_data["time_hours"],
                    estimated_cost_dollars=tech_data["cost_dollars"],
                    sample_requirements=tech_data["sample_prep"],
                    equipment=tech_data["equipment"],
                    notes=f"Priority: {tech_data['priority']}"
                )
                techniques.append(technique)
        
        # Property-specific techniques
        property_technique_map = {
            "fracture_toughness": "Fracture Toughness",
            "thermal_conductivity": "Thermal Conductivity",
            "thermal_conductivity_25C": "Thermal Conductivity",
            "thermal_conductivity_1000C": "Thermal Conductivity",
            "band_gap": "XPS",
            "oxidation": "XPS",
            "phase_stability": "Thermal Analysis",
            "stress": "Raman Spectroscopy"
        }
        
        # Add techniques based on target properties
        added_techniques = set(essential_techniques)
        for prop in target_properties:
            prop_lower = prop.lower()
            for key, tech_name in property_technique_map.items():
                if key in prop_lower and tech_name not in added_techniques:
                    if tech_name in self.characterization_techniques:
                        tech_data = self.characterization_techniques[tech_name]
                        technique = CharacterizationTechnique(
                            technique=tech_name,
                            purpose=tech_data["purpose"],
                            estimated_time_hours=tech_data["time_hours"],
                            estimated_cost_dollars=tech_data["cost_dollars"],
                            sample_requirements=tech_data["sample_prep"],
                            equipment=tech_data["equipment"],
                            notes=f"Priority: {tech_data['priority']}"
                        )
                        techniques.append(technique)
                        added_techniques.add(tech_name)
        
        # Add TEM if high-quality microstructure analysis is needed
        if "TEM" not in added_techniques and len(techniques) >= 4:
            tech_data = self.characterization_techniques["TEM"]
            technique = CharacterizationTechnique(
                technique="TEM",
                purpose=tech_data["purpose"],
                estimated_time_hours=tech_data["time_hours"],
                estimated_cost_dollars=tech_data["cost_dollars"],
                sample_requirements=tech_data["sample_prep"],
                equipment=tech_data["equipment"],
                notes=f"Priority: {tech_data['priority']} - for detailed microstructure"
            )
            techniques.append(technique)
        
        logger.info(f"Generated characterization plan with {len(techniques)} techniques for {formula}")
        
        return techniques
    
    def estimate_resources(
        self,
        synthesis_methods: List[SynthesisMethod],
        characterization_plan: List[CharacterizationTechnique]
    ) -> ResourceEstimate:
        """
        Estimate resources required for experimental work.
        
        Calculates timeline and cost based on:
        - Synthesis method complexity and time
        - Number of characterization techniques
        - Equipment requirements
        
        Args:
            synthesis_methods: List of synthesis methods
            characterization_plan: List of characterization techniques
        
        Returns:
            ResourceEstimate object
        """
        # Calculate synthesis time (assume trying 2-3 methods)
        synthesis_time_hours = sum(m.time_hours for m in synthesis_methods[:2])
        # Add setup and optimization time
        synthesis_time_hours *= 3  # Account for multiple attempts and optimization
        
        # Calculate characterization time
        char_time_hours = sum(t.estimated_time_hours for t in characterization_plan)
        
        # Total timeline in months (assume 40 hours/week productive time)
        total_hours = synthesis_time_hours + char_time_hours
        timeline_months = total_hours / (40 * 4)  # 40 hours/week, 4 weeks/month
        # Add buffer for sample preparation, analysis, and reporting
        timeline_months *= 1.5
        # Minimum 2 months for any experimental work
        timeline_months = max(timeline_months, 2.0)
        
        # Calculate costs
        # Synthesis costs
        synthesis_cost = 0
        for method in synthesis_methods[:2]:  # Assume trying 2 methods
            base_cost = 2.0  # Base cost in k$
            synthesis_cost += base_cost * method.cost_factor
        
        # Characterization costs
        char_cost = sum(t.estimated_cost_dollars for t in characterization_plan) / 1000.0
        
        # Add materials and consumables (20% of total)
        materials_cost = (synthesis_cost + char_cost) * 0.2
        
        # Total cost
        total_cost = synthesis_cost + char_cost + materials_cost
        
        # Collect equipment requirements
        equipment = list(set([m.equipment for m in synthesis_methods] + 
                            [t.equipment for t in characterization_plan]))
        
        # Determine required expertise
        expertise = [
            "Ceramic processing and sintering",
            "Materials characterization",
            "Microstructure analysis"
        ]
        
        # Add specific expertise based on methods
        if any("Spark Plasma" in m.method for m in synthesis_methods):
            expertise.append("SPS/FAST operation")
        
        if any("TEM" in t.technique for t in characterization_plan):
            expertise.append("TEM sample preparation and analysis")
        
        # Identify risk factors
        risk_factors = []
        
        # High temperature risks
        max_temp = max(m.temperature_K for m in synthesis_methods)
        if max_temp > 2000:
            risk_factors.append(f"High processing temperature ({max_temp:.0f} K) may require specialized equipment")
        
        # Difficulty risks
        if any(m.difficulty == DifficultyLevel.HARD.value for m in synthesis_methods):
            risk_factors.append("Complex synthesis methods may require multiple optimization cycles")
        
        # Cost risks
        if total_cost > 15:
            risk_factors.append("High cost may require external funding or phased approach")
        
        # Timeline risks
        if timeline_months > 12:
            risk_factors.append("Extended timeline may face resource availability challenges")
        
        # Determine confidence level
        confidence = 0.8
        if len(risk_factors) > 2:
            confidence = 0.6
        elif len(risk_factors) == 0:
            confidence = 0.9
        
        estimate = ResourceEstimate(
            timeline_months=timeline_months,
            estimated_cost_k_dollars=total_cost,
            required_equipment=equipment,
            required_expertise=expertise,
            risk_factors=risk_factors,
            confidence_level=confidence
        )
        
        logger.info(f"Resource estimate: {timeline_months:.1f} months, ${total_cost:.1f}k")
        
        return estimate
    
    def validate_protocol(
        self,
        synthesis_methods: List[SynthesisMethod],
        characterization_plan: List[CharacterizationTechnique]
    ) -> Dict[str, Any]:
        """
        Validate experimental protocol for completeness and feasibility.
        
        Checks:
        - At least one synthesis method
        - At least one characterization technique
        - Essential characterization techniques present
        - Reasonable temperature and pressure ranges
        
        Args:
            synthesis_methods: List of synthesis methods
            characterization_plan: List of characterization techniques
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0
        }
        
        # Check synthesis methods
        if not synthesis_methods:
            validation_results["is_valid"] = False
            validation_results["errors"].append("No synthesis methods specified")
        else:
            # Check temperature ranges
            for method in synthesis_methods:
                if method.temperature_K < 1000 or method.temperature_K > 3500:
                    validation_results["warnings"].append(
                        f"{method.method}: Temperature {method.temperature_K:.0f} K is outside typical range"
                    )
                
                if method.pressure_MPa < 0 or method.pressure_MPa > 200:
                    validation_results["warnings"].append(
                        f"{method.method}: Pressure {method.pressure_MPa:.0f} MPa is outside typical range"
                    )
        
        # Check characterization plan
        essential = ["XRD", "SEM", "Hardness Testing"]
        
        if not characterization_plan:
            validation_results["is_valid"] = False
            validation_results["errors"].append("No characterization techniques specified")
        else:
            # Check for essential techniques
            present_techniques = [t.technique for t in characterization_plan]
            
            for essential_tech in essential:
                if essential_tech not in present_techniques:
                    validation_results["warnings"].append(
                        f"Essential technique '{essential_tech}' not included in plan"
                    )
        
        # Calculate completeness score
        score = 0.0
        
        # Synthesis methods (30%)
        if synthesis_methods:
            score += 0.3 * min(len(synthesis_methods) / 3.0, 1.0)
        
        # Characterization techniques (40%)
        if characterization_plan:
            score += 0.4 * min(len(characterization_plan) / 6.0, 1.0)
        
        # Essential techniques (30%)
        if characterization_plan:
            essential_count = sum(1 for t in characterization_plan if t.technique in essential)
            score += 0.3 * (essential_count / len(essential))
        
        validation_results["completeness_score"] = score
        
        if score < 0.5:
            validation_results["warnings"].append(
                f"Protocol completeness score is low ({score:.2f}). Consider adding more techniques."
            )
        
        return validation_results
    
    def generate_experimental_protocol(
        self,
        formula: str,
        properties: Dict[str, float],
        target_properties: List[str]
    ) -> Dict[str, Any]:
        """
        Generate complete experimental protocol for a material.
        
        Convenience method that combines all planning steps:
        1. Design synthesis protocols
        2. Design characterization plan
        3. Estimate resources
        4. Validate protocol
        
        Args:
            formula: Chemical formula
            properties: Dictionary of material properties
            target_properties: List of target property names
        
        Returns:
            Dictionary containing complete experimental protocol
        """
        # Design synthesis protocols
        synthesis_methods = self.design_synthesis_protocol(formula, properties)
        
        # Design characterization plan
        characterization_plan = self.design_characterization_plan(formula, target_properties)
        
        # Estimate resources
        resource_estimate = self.estimate_resources(synthesis_methods, characterization_plan)
        
        # Validate protocol
        validation = self.validate_protocol(synthesis_methods, characterization_plan)
        
        protocol = {
            "formula": formula,
            "synthesis_methods": synthesis_methods,
            "characterization_plan": characterization_plan,
            "resource_estimate": resource_estimate,
            "validation": validation,
            "is_valid": validation["is_valid"],
            "completeness_score": validation["completeness_score"]
        }
        
        logger.info(f"Generated complete experimental protocol for {formula}")
        logger.info(f"Validation: {'PASS' if validation['is_valid'] else 'FAIL'}, "
                   f"Completeness: {validation['completeness_score']:.2f}")
        
        return protocol
