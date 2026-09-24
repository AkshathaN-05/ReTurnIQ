"""
Unit and integration tests for catalog completeness, product-specific recommendations,
and checkout quantity warnings.
"""

from collections import Counter
import pytest
from dashboard.app import (
    PRODUCT_CATEGORIES,
    SIMULATED_CATALOG,
    CATEGORY_TO_API,
    get_quantity_warning,
    generate_customer_recommendation,
)
from api.schemas import ProductCategoryEnum


def test_product_categories_count():
    assert len(PRODUCT_CATEGORIES) == 9
    expected_cats = [
        "Clothing",
        "Footwear",
        "Electronics",
        "Beauty / Makeup",
        "Kitchen",
        "Sports",
        "Toys & Games",
        "Books",
        "Home",
    ]
    assert PRODUCT_CATEGORIES == expected_cats


def test_catalog_product_counts():
    counts = Counter(p["category"] for p in SIMULATED_CATALOG)
    for cat in PRODUCT_CATEGORIES:
        assert cat in counts, f"Missing category {cat} in catalog"
        assert counts[cat] >= 3, f"Category {cat} has {counts[cat]} items, expected >= 3"
    assert len(SIMULATED_CATALOG) >= 27


def test_category_api_enums():
    valid_enums = {e.value for e in ProductCategoryEnum}
    for cat in PRODUCT_CATEGORIES:
        api_cat = CATEGORY_TO_API[cat]
        assert api_cat in valid_enums, f"{api_cat} is not in ProductCategoryEnum"


def test_quantity_warnings():
    # Rules: Qty 1 -> None, Qty 2 -> None, Qty 3..5 -> Warning
    assert get_quantity_warning(1) is None
    assert get_quantity_warning(2) is None
    assert get_quantity_warning(3) == "Please double-check the quantity before placing your order. You have selected 3 items."
    assert get_quantity_warning(4) == "Please double-check the quantity before placing your order. You have selected 4 items."
    assert get_quantity_warning(5) == "Please double-check the quantity before placing your order. You have selected 5 items."


def test_recommendations_product_specific():
    for prod in SIMULATED_CATALOG:
        rec = generate_customer_recommendation(prod)
        assert len(rec) > 15

        # Check category cross-contamination rules
        if prod["category"] in ["Toys & Games", "Kitchen", "Electronics", "Books", "Home"]:
            assert "shoe" not in rec.lower()
            assert "waist" not in rec.lower()

        if prod["category"] in ["Footwear"]:
            assert "shoe" in rec.lower() or "boot" in rec.lower() or "fit" in rec.lower()

        # Specific product tests
        if "bat" in prod["name"].lower():
            assert "bat" in rec.lower()
        elif "football" in prod["name"].lower():
            assert "football" in rec.lower() or "ball" in rec.lower()
        elif "yoga" in prod["name"].lower():
            assert "yoga" in rec.lower() or "dimensions" in rec.lower() or "mat" in rec.lower()
        elif "board game" in prod["name"].lower() or "catan" in prod["name"].lower():
            assert "board game" in rec.lower() or "edition" in rec.lower()
        elif "building set" in prod["name"].lower():
            assert "building set" in rec.lower() or "piece" in rec.lower()
        elif "monster truck" in prod["name"].lower():
            assert "remote-control" in rec.lower() or "specifications" in rec.lower() or "scale" in rec.lower()
        elif prod["category"] == "Books":
            assert "book" in rec.lower() or "title" in rec.lower() or "edition" in rec.lower()
        elif prod["category"] == "Kitchen":
            assert "capacity" in rec.lower() or "length" in rec.lower() or "specifications" in rec.lower()
        elif prod["category"] == "Home":
            assert "dimensions" in rec.lower() or "intended use" in rec.lower()


def test_recommendations_with_signals():
    # Sized product with size change history
    jeans = next(p for p in SIMULATED_CATALOG if p["id"] == "PROD-101")
    profile_size_issues = {"size_change_history": 3.0, "previous_return_rate": 0.1}
    rec_size = generate_customer_recommendation(jeans, profile=profile_size_issues)
    assert "fit" in rec_size.lower() or "size" in rec_size.lower()

    # Product with elevated return rate
    blazer = next(p for p in SIMULATED_CATALOG if p["id"] == "PROD-103")
    rec_blazer = generate_customer_recommendation(blazer)
    assert "double-check" in rec_blazer.lower() or "carefully" in rec_blazer.lower()
