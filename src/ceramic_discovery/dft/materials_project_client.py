"""Materials Project API client with rate limiting and retry logic."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class MaterialData:
    """Container for material data from Materials Project."""

    material_id: str
    formula: str
    crystal_structure: Dict[str, Any]
    formation_energy: float
    energy_above_hull: float
    band_gap: Optional[float]
    density: Optional[float]
    properties: Dict[str, Any]
    metadata: Dict[str, Any]


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, rate_limit_per_second: int):
        """
        Initialize rate limiter.

        Args:
            rate_limit_per_second: Maximum requests per second
        """
        self.rate_limit = rate_limit_per_second
        self.min_interval = 1.0 / rate_limit_per_second
        self.last_request_time = 0.0

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class MaterialsProjectClient:
    """Client for Materials Project API with rate limiting and retry logic."""

    BASE_URL = "https://api.materialsproject.org"
    API_VERSION = "v1"

    def __init__(
        self,
        api_key: str,
        rate_limit_per_second: int = 5,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Materials Project client.

        Args:
            api_key: Materials Project API key
            rate_limit_per_second: Maximum requests per second
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        if not api_key:
            raise ValueError("Materials Project API key is required")

        self.api_key = api_key
        self.timeout = timeout_seconds
        self.rate_limiter = RateLimiter(rate_limit_per_second)

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,  # Exponential backoff: 2, 4, 8 seconds
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Set default headers
        self.session.headers.update(
            {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        )

    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting and error handling.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            requests.exceptions.RequestException: On API errors
        """
        self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}/{self.API_VERSION}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise requests.exceptions.RequestException(
                f"Request to {endpoint} timed out after {self.timeout}s"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise requests.exceptions.RequestException("Invalid API key")
            elif e.response.status_code == 429:
                raise requests.exceptions.RequestException("Rate limit exceeded")
            else:
                raise requests.exceptions.RequestException(f"HTTP error: {e}")

    def get_material_by_id(self, material_id: str) -> MaterialData:
        """
        Get material data by Materials Project ID.

        Args:
            material_id: Materials Project ID (e.g., 'mp-149')

        Returns:
            MaterialData object with material properties

        Raises:
            requests.exceptions.RequestException: On API errors
        """
        endpoint = f"materials/{material_id}"
        data = self._make_request(endpoint)

        if not data.get("data"):
            raise ValueError(f"No data found for material ID: {material_id}")

        material = data["data"][0]

        return MaterialData(
            material_id=material.get("material_id", material_id),
            formula=material.get("formula_pretty", ""),
            crystal_structure=material.get("structure", {}),
            formation_energy=material.get("formation_energy_per_atom", 0.0),
            energy_above_hull=material.get("energy_above_hull", 0.0),
            band_gap=material.get("band_gap"),
            density=material.get("density"),
            properties=self._extract_properties(material),
            metadata={
                "source": "Materials Project",
                "api_version": self.API_VERSION,
                "retrieved_at": time.time(),
            },
        )

    def search_materials(
        self,
        formula: Optional[str] = None,
        elements: Optional[List[str]] = None,
        max_energy_above_hull: Optional[float] = None,
        limit: int = 100,
    ) -> List[MaterialData]:
        """
        Search for materials by formula or elements.

        Args:
            formula: Chemical formula (e.g., 'SiC')
            elements: List of elements to include
            max_energy_above_hull: Maximum energy above hull in eV/atom
            limit: Maximum number of results

        Returns:
            List of MaterialData objects

        Raises:
            requests.exceptions.RequestException: On API errors
        """
        params: Dict[str, Any] = {"limit": limit}

        if formula:
            params["formula"] = formula
        if elements:
            params["elements"] = ",".join(elements)
        if max_energy_above_hull is not None:
            params["energy_above_hull"] = f"<{max_energy_above_hull}"

        endpoint = "materials/summary"
        data = self._make_request(endpoint, params=params)

        materials = []
        for material in data.get("data", []):
            materials.append(
                MaterialData(
                    material_id=material.get("material_id", ""),
                    formula=material.get("formula_pretty", ""),
                    crystal_structure=material.get("structure", {}),
                    formation_energy=material.get("formation_energy_per_atom", 0.0),
                    energy_above_hull=material.get("energy_above_hull", 0.0),
                    band_gap=material.get("band_gap"),
                    density=material.get("density"),
                    properties=self._extract_properties(material),
                    metadata={
                        "source": "Materials Project",
                        "api_version": self.API_VERSION,
                        "retrieved_at": time.time(),
                    },
                )
            )

        return materials

    def _extract_properties(self, material: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract standardized properties from material data.

        Args:
            material: Raw material data from API

        Returns:
            Dictionary of extracted properties
        """
        properties = {}

        # Structural properties
        if "structure" in material:
            structure = material["structure"]
            properties["lattice_parameters"] = structure.get("lattice", {})
            properties["space_group"] = structure.get("space_group", {})
            properties["nsites"] = structure.get("nsites")

        # Thermodynamic properties
        properties["formation_energy_per_atom"] = material.get("formation_energy_per_atom")
        properties["energy_above_hull"] = material.get("energy_above_hull")
        properties["is_stable"] = material.get("is_stable")

        # Electronic properties
        properties["band_gap"] = material.get("band_gap")
        properties["is_metal"] = material.get("is_metal")
        properties["is_magnetic"] = material.get("is_magnetic")

        # Mechanical properties (if available)
        properties["density"] = material.get("density")
        properties["density_atomic"] = material.get("density_atomic")

        # Elastic properties (if available)
        if "elasticity" in material:
            elasticity = material["elasticity"]
            properties["bulk_modulus"] = elasticity.get("k_vrh")
            properties["shear_modulus"] = elasticity.get("g_vrh")
            properties["elastic_tensor"] = elasticity.get("elastic_tensor")

        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        return properties

    def get_ceramic_systems(
        self, ceramic_formulas: List[str], max_energy_above_hull: float = 0.1
    ) -> Dict[str, List[MaterialData]]:
        """
        Get materials for multiple ceramic systems.

        Args:
            ceramic_formulas: List of ceramic formulas (e.g., ['SiC', 'B4C'])
            max_energy_above_hull: Maximum energy above hull in eV/atom

        Returns:
            Dictionary mapping formula to list of materials

        Raises:
            requests.exceptions.RequestException: On API errors
        """
        results = {}

        for formula in ceramic_formulas:
            try:
                materials = self.search_materials(
                    formula=formula, max_energy_above_hull=max_energy_above_hull
                )
                results[formula] = materials
            except requests.exceptions.RequestException as e:
                # Log error but continue with other formulas
                results[formula] = []
                print(f"Error fetching {formula}: {e}")

        return results

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self) -> "MaterialsProjectClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
