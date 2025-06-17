"""
Strata Utilities Module

This module provides centralized functions for handling strata abbreviations
across the AIND dashboard. It ensures consistent mapping and display of
strata names throughout all components.
"""


def get_strata_abbreviation(strata_name):
    """
    Convert full strata name to abbreviated forms for UI display

    This function provides a centralized way to abbreviate strata names
    used across the dashboard for consistent display in charts and tables.

    Parameters:
        strata_name (str): The full strata name (e.g., "Uncoupled Baiting_ADVANCED_v3")

    Returns:
        str: The abbreviated strata (e.g., "UBA3")

    Examples:
        >>> get_strata_abbreviation("Uncoupled Baiting_ADVANCED_v3")
        'UBA3'
        >>> get_strata_abbreviation("Coupled Without Baiting_BEGINNER_v1")
        'CWBB1'
        >>> get_strata_abbreviation("")
        ''
    """
    # Return empty string if no strata found
    if not strata_name:
        return ""

    # Hard coded mappings for common terms
    strata_mappings = {
        "Uncoupled Baiting": "UB",
        "Coupled Baiting": "CB",
        "Uncoupled Without Baiting": "UWB",
        "Coupled Without Baiting": "CWB",
        "BEGINNER": "B",
        "INTERMEDIATE": "I",
        "ADVANCED": "A",
        "v1": "1",
        "v2": "2",
        "v3": "3",
    }

    # Split the strata name
    parts = strata_name.split("_")

    # Handle different strata formats
    if len(parts) >= 3:
        # Format: curriculum_Stage_Version (e.g., "Uncoupled Baiting_ADVANCED_v3")
        curriculum = "_".join(parts[:-2])
        stage = parts[-2]
        version = parts[-1]

        # Get abbreviations from mappings
        curriculum_abbr = strata_mappings.get(curriculum, curriculum[:2].upper())
        stage_abbr = strata_mappings.get(stage, stage[0])
        version_abbr = strata_mappings.get(version, version[-1])

        # Combine abbreviations
        return f"{curriculum_abbr}{stage_abbr}{version_abbr}"
    else:
        return strata_name.replace(" ", "")


def get_strata_mappings():
    """
    Get the standard strata mappings dictionary

    Returns:
        dict: Dictionary mapping full strata terms to their abbreviations
    """
    return {
        "Uncoupled Baiting": "UB",
        "Coupled Baiting": "CB",
        "Uncoupled Without Baiting": "UWB",
        "Coupled Without Baiting": "CWB",
        "BEGINNER": "B",
        "INTERMEDIATE": "I",
        "ADVANCED": "A",
        "v1": "1",
        "v2": "2",
        "v3": "3",
    }
