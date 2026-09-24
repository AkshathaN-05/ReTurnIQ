"""
Comprehensive End-to-End Test for Streamlit UI using Streamlit's official AppTest.
Verifies all 15 requirements (A through O) in live simulated runtime.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest
from dashboard.app import PRODUCT_CATEGORIES, SIMULATED_CATALOG


def test_streamlit_app_full_flow():
    # 1. Initialize and run AppTest
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"App crashed on startup: {at.exception}"

    # A. All 9 categories load
    cat_select = at.selectbox(key="sf_cat_filter")
    assert cat_select is not None, "Category selectbox not found"
    available_cats = [opt for opt in cat_select.options if opt != "All Categories"]
    assert len(available_cats) == 9, f"Expected 9 categories, got {len(available_cats)}"
    for cat in PRODUCT_CATEGORIES:
        assert cat in available_cats, f"Category {cat} missing from selectbox options"

    # B. All 9 categories contain >= 3 products
    for cat in PRODUCT_CATEGORIES:
        cat_select.select(cat).run()
        assert not at.exception, f"Crash after selecting category {cat}: {at.exception}"
        prod_select = at.selectbox(key=f"sf_prod_select_{cat}")
        assert prod_select is not None, f"Product selectbox missing for category {cat}"
        assert len(prod_select.options) >= 3, f"Category {cat} has fewer than 3 products: {len(prod_select.options)}"

    # C. Sports works
    cat_select.select("Sports").run()
    assert not at.exception, f"Crash on selecting Sports: {at.exception}"
    sports_prod_select = at.selectbox(key="sf_prod_select_Sports")
    assert len(sports_prod_select.options) >= 3

    # D. Toys & Games works
    cat_select.select("Toys & Games").run()
    assert not at.exception, f"Crash on selecting Toys & Games: {at.exception}"
    toys_prod_select = at.selectbox(key="sf_prod_select_Toys & Games")
    assert len(toys_prod_select.options) >= 3

    # E. Repeated category switching: Clothing -> Sports -> Toys & Games -> Clothing -> Sports -> Toys & Games
    for cycle in range(2):
        for target in ["Clothing", "Sports", "Toys & Games"]:
            cat_select.select(target).run()
            assert not at.exception, f"Crash during cycle {cycle} switching to {target}: {at.exception}"



    # F. Different products can be selected within category
    cat_select.select("Sports").run()
    sports_select = at.selectbox(key="sf_prod_select_Sports")

    # Select Cricket Bat
    sports_select.select("PROD-601").run()
    assert not at.exception
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "bat" in markdown_texts.lower(), "Cricket bat recommendation missing"

    # Select Football
    sports_select.select("PROD-602").run()
    assert not at.exception
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "football" in markdown_texts.lower() or "ball" in markdown_texts.lower(), "Football recommendation missing"

    # Select Yoga Mat
    sports_select.select("PROD-603").run()
    assert not at.exception
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "yoga" in markdown_texts.lower() or "dimensions" in markdown_texts.lower(), "Yoga mat recommendation missing"

    # G. Quantity 1 -> NO warning
    sports_select.select("PROD-601").run()
    qty_input = at.number_input(key="sf_qty")
    qty_input.set_value(1).run()
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Please double-check the quantity before placing your order" not in markdown_texts

    # H. Quantity 2 -> NO warning
    qty_input.set_value(2).run()
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Please double-check the quantity before placing your order" not in markdown_texts

    # I. Quantity 3 -> WARNING
    qty_input.set_value(3).run()
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Please double-check the quantity before placing your order. You have selected 3 items." in markdown_texts

    # J. Quantity 4 -> WARNING
    qty_input.set_value(4).run()
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Please double-check the quantity before placing your order. You have selected 4 items." in markdown_texts
    # Both recommendation AND warning appear
    assert "bat" in markdown_texts.lower()
    assert "Please double-check the quantity before placing your order. You have selected 4 items." in markdown_texts

    # K. Clothing recommendation is relevant to clothing
    cat_select.select("Clothing").run()
    clothing_select = at.selectbox(key="sf_prod_select_Clothing")
    clothing_select.select("PROD-101").run()
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "waist" in markdown_texts.lower() or "fit" in markdown_texts.lower()

    # L. Sports recommendation is relevant to selected sports product
    cat_select.select("Sports").run()
    sports_select = at.selectbox(key="sf_prod_select_Sports")
    sports_select.select("PROD-601").run()
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "bat" in markdown_texts.lower()

    # M. Toys & Games recommendation is relevant to selected toy/game
    cat_select.select("Toys & Games").run()
    toys_select = at.selectbox(key="sf_prod_select_Toys & Games")
    toys_select.select("PROD-701").run()
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "board game" in markdown_texts.lower() or "edition" in markdown_texts.lower()

    # N. No unrelated size recommendation appears for toys
    assert "shoe" not in markdown_texts.lower()
    assert "waist" not in markdown_texts.lower()

    # O. Customer order appears identically in Operations
    # Enter customer name
    name_input = at.text_input(key="sf_customer_name_input")
    name_input.set_value("Akshatha").run()

    # Select Cricket bat, Qty 4
    cat_select.select("Sports").run()
    sports_select = at.selectbox(key="sf_prod_select_Sports")
    sports_select.select("PROD-601").run()
    qty_input = at.number_input(key="sf_qty")
    qty_input.set_value(4).run()

    # Place order
    place_btn = at.button(key="sf_place_order_btn")
    place_btn.click().run()
    assert not at.exception, f"Crash on order placement: {at.exception}"

    # Verify order was recorded in session state
    assert len(at.session_state.orders_history) == 1
    placed_order = at.session_state.orders_history[0]
    assert placed_order["customer_name"] == "Akshatha"
    assert placed_order["product_name"] == "MasterStroke English Willow Cricket Bat"
    assert placed_order["quantity"] == 4
    assert placed_order["price"] == 159.99
    assert placed_order["recommendation"] != ""
    assert placed_order["order_status"] == "Preparing for Packing"
    assert "Please double-check the quantity before placing your order" in placed_order["quantity_warning"]

    # Now switch to Operations Center view
    nav_radio = at.sidebar.radio[0]
    nav_radio.set_value("⚙️ Operations Center").run()
    assert not at.exception, f"Crash switching to Operations Center: {at.exception}"

    # Verify Operations view renders the exact same order
    ops_markdown = " ".join([m.value for m in at.markdown])
    assert placed_order["order_id"] in ops_markdown
    assert "Akshatha" in ops_markdown
    assert "MasterStroke English Willow Cricket Bat" in ops_markdown
    assert "159.99" in ops_markdown
    assert placed_order["recommendation"] in ops_markdown
    assert "Preparing for Packing" in ops_markdown
