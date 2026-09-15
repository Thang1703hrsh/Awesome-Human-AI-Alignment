import unittest

from human_alignment.catalog import categories, validate_catalog
from human_alignment.registry import default_registry


class CatalogTests(unittest.TestCase):
    def test_catalog_has_four_dimensions_and_23_categories(self):
        report = validate_catalog()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.dimension_count, 4)
        self.assertEqual(report.category_count, 23)

    def test_all_registry_categories_exist(self):
        known = {row["id"] for row in categories()}
        for registered in default_registry().list():
            self.assertTrue(set(registered.metadata.taxonomy_categories) <= known)


if __name__ == "__main__":
    unittest.main()
