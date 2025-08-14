"""
Natural sorting utilities for Indonesian legal document hierarchy.

Provides natural sorting for legal unit children to replace database-level
ordinal sorting. Handles various numbering systems used in Indonesian legal
documents including Roman numerals, Arabic numerals, and alphabetical ordering.
"""

import re
from typing import List, Union, Tuple, Any
from dataclasses import dataclass


@dataclass
class SortKey:
    """Structured sort key for natural ordering."""

    numeric_part: int
    alpha_part: str
    suffix_part: str
    original: str

    def __lt__(self, other: 'SortKey') -> bool:
        """Compare sort keys for natural ordering."""
        # Primary sort by numeric part
        if self.numeric_part != other.numeric_part:
            return self.numeric_part < other.numeric_part

        # Secondary sort by alphabetical part
        if self.alpha_part != other.alpha_part:
            return self.alpha_part < other.alpha_part

        # Tertiary sort by suffix
        return self.suffix_part < other.suffix_part


class NaturalSorter:
    """
    Natural sorter for Indonesian legal document numbering systems.

    Handles:
    - Arabic numerals: 1, 2, 3, 10, 11
    - Roman numerals: I, II, III, IV, V, X, XI
    - Alphabetical: a, b, c, aa, ab
    - Mixed formats: 1a, 2b, 1bis, 2ter
    - Parenthetical: (1), (2), (3)
    """

    # Roman numeral mapping
    ROMAN_VALUES = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25,
        'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30
    }

    # Indonesian legal suffixes
    LEGAL_SUFFIXES = {
        'bis': 1,
        'ter': 2,
        'quater': 3,
        'quinquies': 4,
        'sexies': 5,
        'septies': 6,
        'octies': 7,
        'novies': 8,
        'decies': 9
    }

    def __init__(self):
        """Initialize natural sorter with compiled patterns."""
        # Pattern for various number formats
        self.patterns = {
            # Parenthetical numbers: (1), (2), (3)
            'parenthetical': re.compile(r'^\s*\(\s*(\d+)\s*\)\s*$'),

            # Roman numerals: I, II, III, IV
            'roman': re.compile(r'^([IVX]+)(?:\s+(.*))?$'),

            # Arabic with suffix: 1bis, 2ter, 3quater
            'arabic_suffix': re.compile(r'^(\d+)([a-z]+)$'),

            # Arabic with letter: 1a, 2b, 3c
            'arabic_letter': re.compile(r'^(\d+)([a-z])$'),

            # Pure arabic: 1, 2, 3, 10, 11
            'arabic': re.compile(r'^(\d+)$'),

            # Pure alphabetical: a, b, c, aa, bb
            'alpha': re.compile(r'^([a-z]+)$'),

            # Mixed complex: 1a.bis, 2b.ter
            'complex': re.compile(r'^(\d+)([a-z]+)\.?([a-z]+)?$')
        }

    def create_sort_key(self, number_label: str) -> SortKey:
        """
        Create a structured sort key from a number label.

        Args:
            number_label: The number label to parse (e.g., "1", "a", "II", "1bis")

        Returns:
            SortKey object for natural comparison
        """
        if not number_label:
            return SortKey(0, "", "", "")

        label = number_label.strip()
        original = label

        # Try parenthetical format first: (1), (2), (3)
        match = self.patterns['parenthetical'].match(label)
        if match:
            num = int(match.group(1))
            return SortKey(num, "", "", original)

        # Try Roman numerals: I, II, III, IV
        match = self.patterns['roman'].match(label.upper())
        if match:
            roman = match.group(1)
            if roman in self.ROMAN_VALUES:
                num = self.ROMAN_VALUES[roman]
                suffix = match.group(2) or ""
                return SortKey(num, "", suffix, original)

        # Try Arabic with suffix: 1bis, 2ter
        match = self.patterns['arabic_suffix'].match(label.lower())
        if match:
            num = int(match.group(1))
            suffix = match.group(2)
            suffix_weight = self.LEGAL_SUFFIXES.get(suffix, 999)
            return SortKey(num * 1000 + suffix_weight, "", suffix, original)

        # Try Arabic with letter: 1a, 2b
        match = self.patterns['arabic_letter'].match(label.lower())
        if match:
            num = int(match.group(1))
            letter = match.group(2)
            letter_value = ord(letter) - ord('a')
            return SortKey(num * 100 + letter_value, letter, "", original)

        # Try pure Arabic: 1, 2, 3, 10
        match = self.patterns['arabic'].match(label)
        if match:
            num = int(match.group(1))
            return SortKey(num, "", "", original)

        # Try pure alphabetical: a, b, c, aa, bb
        match = self.patterns['alpha'].match(label.lower())
        if match:
            alpha = match.group(1)
            # Convert alphabetical to numeric equivalent
            num_value = self._alpha_to_numeric(alpha)
            return SortKey(num_value, alpha, "", original)

        # Default: treat as string
        return SortKey(999999, label.lower(), "", original)

    def _alpha_to_numeric(self, alpha: str) -> int:
        """
        Convert alphabetical string to numeric equivalent for sorting.

        Examples:
        a = 1, b = 2, ..., z = 26
        aa = 27, ab = 28, ..., az = 52
        ba = 53, etc.
        """
        if not alpha:
            return 0

        result = 0
        for i, char in enumerate(alpha):
            char_value = ord(char.lower()) - ord('a') + 1
            result = result * 26 + char_value

        return result

    def sort_legal_units(self, units: List[Any], key_attr: str = 'number_label') -> List[Any]:
        """
        Sort legal units using natural ordering on number_label.

        Args:
            units: List of legal unit objects
            key_attr: Attribute name containing the number label

        Returns:
            Sorted list of legal units
        """
        def sort_key_func(unit):
            number_label = getattr(unit, key_attr, "") or ""
            return self.create_sort_key(str(number_label))

        return sorted(units, key=sort_key_func)

    def sort_dict_list(self, items: List[dict], key_name: str = 'number_label') -> List[dict]:
        """
        Sort list of dictionaries using natural ordering.

        Args:
            items: List of dictionaries
            key_name: Dictionary key containing the number label

        Returns:
            Sorted list of dictionaries
        """
        def sort_key_func(item):
            number_label = item.get(key_name, "") or ""
            return self.create_sort_key(str(number_label))

        return sorted(items, key=sort_key_func)

    def sort_strings(self, labels: List[str]) -> List[str]:
        """
        Sort list of number label strings using natural ordering.

        Args:
            labels: List of number label strings

        Returns:
            Sorted list of strings
        """
        return sorted(labels, key=self.create_sort_key)


# Module-level convenience functions
_sorter = None

def get_natural_sorter() -> NaturalSorter:
    """Get singleton natural sorter instance."""
    global _sorter
    if _sorter is None:
        _sorter = NaturalSorter()
    return _sorter


def natural_sort_legal_units(units: List[Any], key_attr: str = 'number_label') -> List[Any]:
    """Sort legal units using natural ordering."""
    return get_natural_sorter().sort_legal_units(units, key_attr)


def natural_sort_dicts(items: List[dict], key_name: str = 'number_label') -> List[dict]:
    """Sort dictionaries using natural ordering."""
    return get_natural_sorter().sort_dict_list(items, key_name)


def natural_sort_strings(labels: List[str]) -> List[str]:
    """Sort strings using natural ordering."""
    return get_natural_sorter().sort_strings(labels)


def test_natural_sort():
    """Test function to validate natural sorting behavior."""
    test_cases = [
        # Test case 1: Arabic numerals
        {
            'input': ['10', '2', '1', '3', '11', '20'],
            'expected': ['1', '2', '3', '10', '11', '20']
        },

        # Test case 2: Alphabetical
        {
            'input': ['c', 'a', 'b', 'aa', 'z', 'ab'],
            'expected': ['a', 'b', 'c', 'z', 'aa', 'ab']
        },

        # Test case 3: Roman numerals
        {
            'input': ['X', 'II', 'I', 'IV', 'V', 'III'],
            'expected': ['I', 'II', 'III', 'IV', 'V', 'X']
        },

        # Test case 4: Parenthetical
        {
            'input': ['(10)', '(2)', '(1)', '(3)'],
            'expected': ['(1)', '(2)', '(3)', '(10)']
        },

        # Test case 5: Mixed with suffixes
        {
            'input': ['1bis', '1', '2', '1ter', '3'],
            'expected': ['1', '1bis', '1ter', '2', '3']
        },

        # Test case 6: Arabic with letters
        {
            'input': ['1b', '1a', '2', '1c', '1'],
            'expected': ['1', '1a', '1b', '1c', '2']
        }
    ]

    sorter = NaturalSorter()

    for i, test_case in enumerate(test_cases, 1):
        result = sorter.sort_strings(test_case['input'])
        if result == test_case['expected']:
            print(f"✓ Test case {i}: PASS")
        else:
            print(f"✗ Test case {i}: FAIL")
            print(f"  Input:    {test_case['input']}")
            print(f"  Expected: {test_case['expected']}")
            print(f"  Got:      {result}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_natural_sort()
