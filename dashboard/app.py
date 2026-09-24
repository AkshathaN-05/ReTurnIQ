"""
ReTurnIQ — AI-Powered E-Commerce Return Risk Prediction and Prevention System
=============================================================================

Production-grade dual-mode application showcasing:
1. 🛍️ Customer Storefront (NovaCart)
   - Real-world e-commerce shopping experience (typed customer name, product, size, delivery, payment).
   - Transparent background return risk evaluation via ReTurnIQ API.
   - Non-technical, customer-friendly pre-dispatch guidance.
   - Realistic multi-step order progression.

2. ⚙️ Operations Center
   - Real-time Operations Overview (KPI cards from active session).
   - Live Recent Orders queue synchronized directly from customer checkouts.
   - In-depth Selected Order Details with grounded risk attribution & interventions.
   - What-If Risk Analysis for interactive scenario comparison via API.
   - In-Session Operational Analytics backed strictly by real order data.
   - Secondary Advanced Evaluation Tools (manual form & presets) for viva/demo.
   - Collapsed Technical / Evaluator Details section.

Architectural Principles:
- Single source of truth: One prediction call on checkout, stored in session state.
- Customer name serves as order identity, not an ML feature.
- Zero ML jargon on customer storefront.
- Full theme-aware styling supporting both light and dark modes.
"""

from __future__ import annotations

import datetime
import os
import random
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# =====================================================================
# Configuration & API Utilities
# =====================================================================

DEFAULT_API_BASE_URL = os.getenv("RETURNIQ_API_URL", "http://127.0.0.1:8001")


def get_api_base_url() -> str:
    """Detect active API server (checks configured port first, then fallbacks)."""
    candidates = [DEFAULT_API_BASE_URL]
    if "8001" in DEFAULT_API_BASE_URL:
        candidates.append("http://127.0.0.1:8000")
    elif "8000" in DEFAULT_API_BASE_URL:
        candidates.append("http://127.0.0.1:8001")

    for base in candidates:
        try:
            r = requests.get(f"{base}/health", timeout=0.8)
            if r.status_code == 200:
                return base
        except Exception:
            continue
    return DEFAULT_API_BASE_URL


PRODUCT_CATEGORIES = [
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

CATEGORY_TO_API = {
    "Clothing": "Clothing",
    "Footwear": "Footwear",
    "Electronics": "Electronics",
    "Beauty / Makeup": "Beauty & Personal Care",
    "Beauty & Personal Care": "Beauty & Personal Care",
    "Kitchen": "Home & Kitchen",
    "Home & Kitchen": "Home & Kitchen",
    "Sports": "Sports & Outdoors",
    "Sports & Outdoors": "Sports & Outdoors",
    "Toys & Games": "Toys & Games",
    "Books": "Books",
    "Home": "Home & Kitchen",
}

FEATURE_DISPLAY_NAMES = {
    "complaint_x_return_rate": "Customer complaints & previous return history",
    "discount_percentage": "High discount",
    "previous_orders": "Previous order history",
    "previous_returns": "Previous return history",
    "previous_return_rate": "Customer return history",
    "product_return_rate": "Category return rate",
    "size_change_history": "Size exchange history",
    "size_change_x_high_risk_cat": "High-risk category size exchanges",
    "return_rate_x_product_rate": "Customer & product return risk interaction",
    "discount_amount": "High discount amount",
    "price_discount_interaction": "Price & discount interaction",
    "complaint_rate": "Customer complaint history",
    "customer_complaint_count": "Customer complaint history",
    "product_category_Clothing": "Clothing category",
    "product_category_Footwear": "Footwear category",
    "payment_method_Cash on Delivery": "Cash on Delivery payment",
    "delivery_distance_km": "Long delivery distance",
    "order_value": "High order value",
    "quantity": "High order quantity",
    "is_high_risk_category": "High-risk category",
}

ORDER_DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

PAYMENT_METHODS = [
    "Credit Card",
    "Debit Card",
    "UPI",
    "Cash on Delivery",
]

# =====================================================================
# Catalog Benchmarks (Minimum 3 products per category, 27 total)
# =====================================================================

SIMULATED_CATALOG = [
    # -----------------------------------------------------------------
    # 1. Clothing (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-101",
        "name": "Slim-Fit Stretch Denim Jeans",
        "category": "Clothing",
        "price": 69.99,
        "rating": 4.3,
        "sizes": ["28", "30", "32", "34", "36"],
        "product_return_rate": 0.28,
        "icon": "👖",
        "desc": "Premium stretch denim with tailored fit and reinforced seams.",
        "guidance": "Please double-check the selected waist size and fit before placing your order.",
        "guidance_topic": "waist size and fit",
    },
    {
        "id": "PROD-102",
        "name": "Classic Organic Cotton Crewneck T-Shirt",
        "category": "Clothing",
        "price": 24.99,
        "rating": 4.6,
        "sizes": ["S", "M", "L", "XL", "XXL"],
        "product_return_rate": 0.12,
        "icon": "👕",
        "desc": "100% breathable organic cotton, pre-shrunk for everyday comfort.",
        "guidance": "Please double-check the selected shirt size and fit before placing your order.",
        "guidance_topic": "shirt size and fit",
    },
    {
        "id": "PROD-103",
        "name": "Designer Tailored Evening Blazer",
        "category": "Clothing",
        "price": 189.99,
        "rating": 4.1,
        "sizes": ["38R", "40R", "42R", "44R"],
        "product_return_rate": 0.35,
        "icon": "🧥",
        "desc": "Structured single-breasted blazer with peaked lapels and satin lining.",
        "guidance": "Please double-check the selected blazer chest size and tailored fit before placing your order.",
        "guidance_topic": "blazer chest size and tailored fit",
    },
    # -----------------------------------------------------------------
    # 2. Footwear (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-201",
        "name": "AeroStep Pro Running Sneakers",
        "category": "Footwear",
        "price": 129.99,
        "rating": 4.5,
        "sizes": ["US 7", "US 8", "US 9", "US 10", "US 11", "US 12"],
        "product_return_rate": 0.22,
        "icon": "👟",
        "desc": "Cushioned responsive road running shoes with breathable mesh upper.",
        "guidance": "Please confirm your shoe size and fit before placing your order.",
        "guidance_topic": "running shoe size and fit",
    },
    {
        "id": "PROD-202",
        "name": "Italian Handcrafted Leather Penny Loafers",
        "category": "Footwear",
        "price": 89.99,
        "rating": 4.4,
        "sizes": ["US 8", "US 9", "US 10", "US 11"],
        "product_return_rate": 0.19,
        "icon": "👞",
        "desc": "Handcrafted leather slip-ons designed for all-day office comfort.",
        "guidance": "Please confirm your shoe size and fit before placing your order.",
        "guidance_topic": "loafer shoe size and fit",
    },
    {
        "id": "PROD-203",
        "name": "TrailGrip All-Weather Waterproof Hiking Boots",
        "category": "Footwear",
        "price": 149.99,
        "rating": 4.7,
        "sizes": ["US 8", "US 9", "US 10", "US 11", "US 12"],
        "product_return_rate": 0.16,
        "icon": "🥾",
        "desc": "Rugged waterproof nubuck leather boots with high-traction Vibram soles.",
        "guidance": "Please confirm your shoe size and fit, allowing room for hiking socks, before placing your order.",
        "guidance_topic": "boot size and fit",
    },
    # -----------------------------------------------------------------
    # 3. Electronics (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-301",
        "name": "QuietPulse Wireless ANC Headphones",
        "category": "Electronics",
        "price": 149.99,
        "rating": 4.7,
        "sizes": [],
        "product_return_rate": 0.08,
        "icon": "🎧",
        "desc": "Active noise cancellation, 40-hour battery life, and spatial audio support.",
        "guidance": "Please confirm the selected model and audio specifications before placing your order.",
        "guidance_topic": "audio model and device compatibility",
    },
    {
        "id": "PROD-302",
        "name": "SmartPulse Pro Fitness Smartwatch",
        "category": "Electronics",
        "price": 79.99,
        "rating": 4.2,
        "sizes": [],
        "product_return_rate": 0.11,
        "icon": "⌚",
        "desc": "Continuous heart rate, SpO2 monitoring, GPS tracking, and sleep analysis.",
        "guidance": "Please confirm the selected smartwatch model and smartphone OS compatibility before placing your order.",
        "guidance_topic": "smartwatch model and smartphone OS compatibility",
    },
    {
        "id": "PROD-303",
        "name": "UltraView 27-Inch 4K USB-C Monitor",
        "category": "Electronics",
        "price": 329.99,
        "rating": 4.6,
        "sizes": [],
        "product_return_rate": 0.09,
        "icon": "🖥️",
        "desc": "IPS panel with 99% sRGB coverage, 65W power delivery, and ergonomic stand.",
        "guidance": "Please confirm the display resolution, ports, and desk setup specifications before placing your order.",
        "guidance_topic": "display resolution and ports",
    },
    # -----------------------------------------------------------------
    # 4. Beauty / Makeup (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-401",
        "name": "HydraGlow Vitamin C Daily Serum",
        "category": "Beauty / Makeup",
        "price": 28.99,
        "rating": 4.5,
        "sizes": [],
        "product_return_rate": 0.05,
        "icon": "✨",
        "desc": "Antioxidant facial serum with hyaluronic acid and ferulic acid.",
        "guidance": "Please confirm the serum formula and skin-type suitability before placing your order.",
        "guidance_topic": "serum formula and skin-type suitability",
    },
    {
        "id": "PROD-402",
        "name": "Velvet Matte Longwear Foundation (30ml)",
        "category": "Beauty / Makeup",
        "price": 34.50,
        "rating": 4.3,
        "sizes": ["Fair Neutral", "Medium Warm", "Tan Olive", "Deep Rich"],
        "product_return_rate": 0.14,
        "icon": "💄",
        "desc": "24-hour transfer-resistant full coverage liquid foundation with SPF 15.",
        "guidance": "Please confirm the selected shade or variant before placing your order.",
        "guidance_topic": "foundation shade and undertone",
    },
    {
        "id": "PROD-403",
        "name": "SilkRadiance Botanical Cleansing Oil (150ml)",
        "category": "Beauty / Makeup",
        "price": 22.00,
        "rating": 4.7,
        "sizes": [],
        "product_return_rate": 0.04,
        "icon": "🧴",
        "desc": "Gentle emulsifying facial cleanser infused with camellia and jojoba oils.",
        "guidance": "Please confirm the bottle size and skin sensitivity suitability before placing your order.",
        "guidance_topic": "cleansing oil formula and skin sensitivity",
    },
    # -----------------------------------------------------------------
    # 5. Kitchen (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-501",
        "name": "Artisan French Press Coffee Maker (1L)",
        "category": "Kitchen",
        "price": 34.99,
        "rating": 4.6,
        "sizes": [],
        "product_return_rate": 0.07,
        "icon": "☕",
        "desc": "Heat-resistant borosilicate glass with 4-level stainless steel filtration.",
        "guidance": "Please confirm the selected capacity and specifications before placing your order.",
        "guidance_topic": "1L brewing capacity and counter dimensions",
    },
    {
        "id": "PROD-502",
        "name": "ProChef 8-Inch Japanese Damascus Chef Knife",
        "category": "Kitchen",
        "price": 69.99,
        "rating": 4.8,
        "sizes": [],
        "product_return_rate": 0.06,
        "icon": "🔪",
        "desc": "67-layer VG-10 Damascus steel blade with ergonomic pakkawood handle.",
        "guidance": "Please confirm the selected capacity and specifications before placing your order.",
        "guidance_topic": "8-inch blade length and knife-care requirements",
    },
    {
        "id": "PROD-503",
        "name": "Digital Precision Dual-Basket Air Fryer 8L",
        "category": "Kitchen",
        "price": 119.99,
        "rating": 4.5,
        "sizes": [],
        "product_return_rate": 0.12,
        "icon": "🍳",
        "desc": "Dual independent cooking zones with 8 presets and dishwasher-safe baskets.",
        "guidance": "Please confirm the selected capacity and specifications before placing your order.",
        "guidance_topic": "8L capacity and kitchen counter dimensions",
    },
    # -----------------------------------------------------------------
    # 6. Sports (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-601",
        "name": "MasterStroke English Willow Cricket Bat",
        "category": "Sports",
        "price": 159.99,
        "rating": 4.4,
        "sizes": ["Short Handle", "Long Handle", "Harrow"],
        "product_return_rate": 0.15,
        "icon": "🏏",
        "desc": "Hand-crafted Grade 1 English willow with thick edges and balanced pickup.",
        "guidance": "Please confirm the selected bat size/specification before placing your order.",
        "guidance_topic": "cricket bat size and handle specification",
    },
    {
        "id": "PROD-602",
        "name": "Aerodynamic Pro Match Football",
        "category": "Sports",
        "price": 39.99,
        "rating": 4.6,
        "sizes": ["Size 4 (Youth)", "Size 5 (Official Match)"],
        "product_return_rate": 0.08,
        "icon": "⚽",
        "desc": "FIFA-quality thermal-bonded PU match football with textured outer casing.",
        "guidance": "Please confirm the selected football ball size before placing your order.",
        "guidance_topic": "football ball size",
    },
    {
        "id": "PROD-603",
        "name": "EcoGrip Non-Slip Pro Yoga Mat (6mm)",
        "category": "Sports",
        "price": 48.00,
        "rating": 4.7,
        "sizes": ["Standard (72x24 in)", "Extra-Long (84x26 in)"],
        "product_return_rate": 0.07,
        "icon": "🧘",
        "desc": "High-density natural tree rubber with textured anti-tear alignment lines.",
        "guidance": "Please confirm the yoga mat dimensions and thickness before placing your order.",
        "guidance_topic": "yoga mat dimensions and thickness",
    },
    # -----------------------------------------------------------------
    # 7. Toys & Games (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-701",
        "name": "Catan Trade & Settle Strategy Board Game",
        "category": "Toys & Games",
        "price": 44.99,
        "rating": 4.8,
        "sizes": [],
        "product_return_rate": 0.05,
        "icon": "🎲",
        "desc": "Classic resource trading and tactical settlement building board game for ages 10+.",
        "guidance": "Please confirm the board game edition and player age suitability before placing your order.",
        "guidance_topic": "board game edition and player age suitability",
    },
    {
        "id": "PROD-702",
        "name": "Galactic Cruiser Architecture Building Set",
        "category": "Toys & Games",
        "price": 89.99,
        "rating": 4.7,
        "sizes": [],
        "product_return_rate": 0.07,
        "icon": "🧱",
        "desc": "1,248-piece advanced interlocking brick space cruiser model kit with mini figures.",
        "guidance": "Please confirm the building set piece count and age suitability before placing your order.",
        "guidance_topic": "building set piece count and age suitability",
    },
    {
        "id": "PROD-703",
        "name": "TurboDrift High-Speed RC Monster Truck",
        "category": "Toys & Games",
        "price": 59.99,
        "rating": 4.3,
        "sizes": [],
        "product_return_rate": 0.13,
        "icon": "🏎️",
        "desc": "1:16 scale 4WD all-terrain remote control truck with 2.4GHz controller and dual batteries.",
        "guidance": "Please confirm the remote-control model specifications before placing your order.",
        "guidance_topic": "remote-control model scale and battery specifications",
    },
    # -----------------------------------------------------------------
    # 8. Books (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-801",
        "name": "Machine Learning Engineering in Production",
        "category": "Books",
        "price": 42.99,
        "rating": 4.9,
        "sizes": ["Hardcover", "Paperback", "Kindle Edition"],
        "product_return_rate": 0.03,
        "icon": "📚",
        "desc": "Hardcover practical guide to deploying and monitoring ML pipelines in production.",
        "guidance": "Please confirm the book title and edition before placing your order.",
        "guidance_topic": "book title and hardcover edition",
    },
    {
        "id": "PROD-802",
        "name": "Designing Data-Intensive Applications",
        "category": "Books",
        "price": 49.99,
        "rating": 4.9,
        "sizes": ["Paperback", "Hardcover"],
        "product_return_rate": 0.02,
        "icon": "📖",
        "desc": "The definitive guide to the architecture of data systems, storage, and processing.",
        "guidance": "Please confirm the book title and edition before placing your order.",
        "guidance_topic": "book title and paperback edition",
    },
    {
        "id": "PROD-803",
        "name": "Clean Code: A Handbook of Agile Craftsmanship",
        "category": "Books",
        "price": 38.50,
        "rating": 4.6,
        "sizes": ["Paperback", "Spiral-Bound"],
        "product_return_rate": 0.04,
        "icon": "📘",
        "desc": "Foundational software engineering principles for writing clean, maintainable code.",
        "guidance": "Please confirm the book title and format before placing your order.",
        "guidance_topic": "book title and format",
    },
    # -----------------------------------------------------------------
    # 9. Home (3 items)
    # -----------------------------------------------------------------
    {
        "id": "PROD-901",
        "name": "Ergonomic Memory Foam Lumbar Support Cushion",
        "category": "Home",
        "price": 36.99,
        "rating": 4.5,
        "sizes": [],
        "product_return_rate": 0.10,
        "icon": "🛋️",
        "desc": "High-density molded memory foam pillow with breathable 3D mesh cover and strap.",
        "guidance": "Please confirm the cushion dimensions and intended use before placing your order.",
        "guidance_topic": "cushion dimensions and intended chair fit",
    },
    {
        "id": "PROD-902",
        "name": "Blackout Thermal Insulated Window Curtains (Pair)",
        "category": "Home",
        "price": 42.99,
        "rating": 4.4,
        "sizes": ["52x63 in", "52x84 in", "52x96 in"],
        "product_return_rate": 0.16,
        "icon": "🪟",
        "desc": "Triple-weave blackout fabric with grommet top for light blocking and thermal privacy.",
        "guidance": "Please confirm the curtain dimensions and intended use before placing your order.",
        "guidance_topic": "window measurements and curtain drop length",
    },
    {
        "id": "PROD-903",
        "name": "Dimmable LED Touch Bedside Table Lamp",
        "category": "Home",
        "price": 29.99,
        "rating": 4.6,
        "sizes": [],
        "product_return_rate": 0.06,
        "icon": "💡",
        "desc": "3-way dimmable touch control lamp with dual USB charging ports and warm white bulb.",
        "guidance": "Please confirm the lamp dimensions and intended use before placing your order.",
        "guidance_topic": "lamp dimensions and intended placement space",
    },
]

# Presets for manual evaluation tab in Operations Center
PRESET_HIGH_RISK = {
    "customer_age": 25,
    "customer_tenure_months": 3.0,
    "previous_orders": 6,
    "previous_returns": 4,
    "previous_return_rate": 0.667,
    "customer_complaint_count": 3.0,
    "average_previous_order_value": 120.0,
    "customer_account_verified": True,
    "product_category": "Clothing",
    "product_price": 149.99,
    "product_rating": 3.6,
    "product_return_rate": 0.35,
    "discount_percentage": 45.0,
    "quantity": 1,
    "order_day": "Friday",
    "order_hour": 22,
    "delivery_distance_km": 42.0,
    "payment_method": "Cash on Delivery",
    "order_value": 82.49,
    "is_gift_order": False,
    "is_expedited_shipping": False,
    "size_change_history": 4.0,
    "category_return_history": 0.60,
    "previous_category_orders": 3.0,
    "previous_category_returns": 2.0,
}

PRESET_STANDARD = {
    "customer_age": 34,
    "customer_tenure_months": 14.5,
    "previous_orders": 10,
    "previous_returns": 1,
    "previous_return_rate": 0.10,
    "customer_complaint_count": 0.0,
    "average_previous_order_value": 75.5,
    "customer_account_verified": True,
    "product_category": "Electronics",
    "product_price": 89.99,
    "product_rating": 4.2,
    "product_return_rate": 0.18,
    "discount_percentage": 15.0,
    "quantity": 1,
    "order_day": "Monday",
    "order_hour": 14,
    "delivery_distance_km": 12.5,
    "payment_method": "Credit Card",
    "order_value": 76.49,
    "is_gift_order": False,
    "is_expedited_shipping": True,
    "size_change_history": 1.0,
    "category_return_history": 0.20,
    "previous_category_orders": 4.0,
    "previous_category_returns": 1.0,
}

PRESET_LOW_RISK = {
    "customer_age": 48,
    "customer_tenure_months": 36.0,
    "previous_orders": 35,
    "previous_returns": 1,
    "previous_return_rate": 0.029,
    "customer_complaint_count": 0.0,
    "average_previous_order_value": 65.0,
    "customer_account_verified": True,
    "product_category": "Books",
    "product_price": 29.99,
    "product_rating": 4.8,
    "product_return_rate": 0.03,
    "discount_percentage": 5.0,
    "quantity": 2,
    "order_day": "Wednesday",
    "order_hour": 11,
    "delivery_distance_km": 8.0,
    "payment_method": "Debit Card",
    "order_value": 56.98,
    "is_gift_order": False,
    "is_expedited_shipping": False,
    "size_change_history": 0.0,
    "category_return_history": 0.0,
    "previous_category_orders": 12.0,
    "previous_category_returns": 0.0,
}

# =====================================================================
# Theme-Aware Styling (Light & Dark Mode)
# =====================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --theme-card-bg: #ffffff;
    --theme-card-border: #e2e8f0;
    --theme-card-hover-border: #cbd5e1;
    --theme-alt-bg: #f8fafc;
    --theme-alt-border: #e2e8f0;
    --theme-text-primary: #0f172a;
    --theme-text-secondary: #334155;
    --theme-text-muted: #64748b;
    --theme-accent: #4f46e5;
    --theme-accent-light: rgba(99, 102, 241, 0.12);
    --theme-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);

    /* Recommendations Light */
    --recom-high-bg: #fefce8;
    --recom-high-border: #eab308;
    --recom-high-title: #854d0e;
    --recom-high-text: #713f12;

    --recom-med-bg: #eff6ff;
    --recom-med-border: #3b82f6;
    --recom-med-title: #1d4ed8;
    --recom-med-text: #1e3a8a;

    --recom-low-bg: #f0fdf4;
    --recom-low-border: #22c55e;
    --recom-low-title: #15803d;
    --recom-low-text: #14532d;
}

@media (prefers-color-scheme: dark) {
    :root {
        --theme-card-bg: #1e293b;
        --theme-card-border: #334155;
        --theme-card-hover-border: #475569;
        --theme-alt-bg: #0f172a;
        --theme-alt-border: #334155;
        --theme-text-primary: #f8fafc;
        --theme-text-secondary: #cbd5e1;
        --theme-text-muted: #94a3b8;
        --theme-accent: #818cf8;
        --theme-accent-light: rgba(129, 140, 248, 0.18);
        --theme-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);

        /* Recommendations Dark */
        --recom-high-bg: #422006;
        --recom-high-border: #f59e0b;
        --recom-high-title: #fde047;
        --recom-high-text: #fef08a;

        --recom-med-bg: #172554;
        --recom-med-border: #60a5fa;
        --recom-med-title: #93c5fd;
        --recom-med-text: #dbeafe;

        --recom-low-bg: #052e16;
        --recom-low-border: #4ade80;
        --recom-low-title: #86efac;
        --recom-low-text: #dcfce7;
    }
}

/* Streamlit Theme Adaptations */
[data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Top App Header */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #0f172a 100%);
    border-radius: 16px;
    padding: 1.6rem 2.0rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(129, 140, 248, 0.3);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-header-title {
    color: #f8fafc !important;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header-subtitle {
    color: #cbd5e1 !important;
    font-size: 1.0rem;
    margin-top: 0.3rem;
    font-weight: 500;
}
.app-header-badge {
    background: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(129, 140, 248, 0.6);
    color: #e0e7ff;
    padding: 0.45rem 1.0rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

/* Storefront Banner */
.storefront-banner {
    background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
    color: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25);
}
.storefront-banner h2 {
    color: #ffffff !important;
    margin: 0 0 0.3rem 0;
    font-size: 1.5rem;
    font-weight: 800;
}
.storefront-banner p {
    margin: 0;
    color: #e0e7ff !important;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Cards & Layout */
.product-display-card {
    background: var(--theme-card-bg);
    border: 1px solid var(--theme-card-border);
    border-radius: 16px;
    padding: 1.4rem;
    box-shadow: var(--theme-shadow);
    margin-bottom: 1.0rem;
}
.product-icon-large {
    font-size: 3.2rem;
    text-align: center;
    margin-bottom: 0.4rem;
}
.product-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--theme-text-primary);
    margin-bottom: 0.3rem;
}
.product-category-chip {
    display: inline-block;
    background: var(--theme-accent-light);
    color: var(--theme-accent);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.6rem;
}
.product-price-tag {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--theme-accent);
}
.product-desc {
    color: var(--theme-text-secondary);
    font-size: 0.88rem;
    line-height: 1.5;
    margin-top: 0.4rem;
}

.checkout-summary-card {
    background: var(--theme-alt-bg);
    border: 1px solid var(--theme-alt-border);
    border-radius: 16px;
    padding: 1.4rem;
    box-shadow: var(--theme-shadow);
}
.summary-title {
    margin: 0 0 0.8rem 0;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--theme-text-primary);
}
.summary-line {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    font-size: 0.95rem;
    color: var(--theme-text-secondary);
}
.summary-total-line {
    display: flex;
    justify-content: space-between;
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--theme-text-primary);
    padding-top: 0.5rem;
    border-top: 1px solid var(--theme-card-border);
    margin-top: 0.5rem;
}

/* Order Progression */
.order-step-container {
    background: var(--theme-card-bg);
    border: 1px solid var(--theme-card-border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--theme-shadow);
}
.order-step-item {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--theme-text-primary);
}
.order-step-item:last-child {
    margin-bottom: 0;
}
.order-step-check {
    font-size: 1.15rem;
    margin-right: 0.7rem;
}

/* Customer Order Detail Card */
.order-confirmed-card {
    background: var(--theme-card-bg);
    border: 1px solid var(--theme-card-border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1.0rem 0;
    box-shadow: var(--theme-shadow);
}
.order-confirmed-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--theme-text-primary);
    margin-bottom: 0.5rem;
}
.order-confirmed-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.5rem;
    font-size: 0.92rem;
    color: var(--theme-text-secondary);
}

/* Customer-Facing Recommendations */
.customer-recom-box {
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin: 1.0rem 0;
    border-width: 2px;
    border-style: solid;
}
.recom-high {
    background: var(--recom-high-bg);
    border-color: var(--recom-high-border);
    color: var(--recom-high-text);
}
.recom-high h4 {
    color: var(--recom-high-title) !important;
    margin: 0 0 0.4rem 0;
    font-size: 1.15rem;
    font-weight: 700;
}
.recom-high p {
    color: var(--recom-high-text) !important;
    font-size: 0.95rem;
    line-height: 1.55;
    margin: 0;
}

.recom-medium {
    background: var(--recom-med-bg);
    border-color: var(--recom-med-border);
    color: var(--recom-med-text);
}
.recom-medium h4 {
    color: var(--recom-med-title) !important;
    margin: 0 0 0.4rem 0;
    font-size: 1.15rem;
    font-weight: 700;
}
.recom-medium p {
    color: var(--recom-med-text) !important;
    font-size: 0.95rem;
    line-height: 1.55;
    margin: 0;
}

.recom-low {
    background: var(--recom-low-bg);
    border-color: var(--recom-low-border);
    color: var(--recom-low-text);
}
.recom-low h4 {
    color: var(--recom-low-title) !important;
    margin: 0 0 0.4rem 0;
    font-size: 1.15rem;
    font-weight: 700;
}
.recom-low p {
    color: var(--recom-low-text) !important;
    font-size: 0.95rem;
    line-height: 1.55;
    margin: 0;
}

/* Operations Overview KPI Cards */
.kpi-card {
    background: var(--theme-card-bg);
    border: 1px solid var(--theme-card-border);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    box-shadow: var(--theme-shadow);
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--theme-text-muted);
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
    color: var(--theme-text-primary);
}
.kpi-sub {
    font-size: 0.8rem;
    color: var(--theme-text-muted);
    margin-top: 0.35rem;
}

/* Operational Order Cards */
.ops-order-card {
    background: var(--theme-card-bg);
    border: 1px solid var(--theme-card-border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--theme-shadow);
}
.ops-order-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--theme-card-border);
}
.ops-order-id {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--theme-text-primary);
}
.ops-customer-name {
    margin-left: 0.8rem;
    font-weight: 700;
    color: var(--theme-accent);
}
.ops-order-body {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.6rem;
    font-size: 0.92rem;
    color: var(--theme-text-secondary);
}
.ops-order-recom {
    margin-top: 0.7rem;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    background: var(--theme-alt-bg);
    border-left: 3px solid var(--theme-accent);
    font-size: 0.88rem;
    color: var(--theme-text-primary);
}

/* Risk Badges */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
}
.risk-high {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #f87171;
}
.risk-medium {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fbbf24;
}
.risk-low {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
}

/* Factor Cards & Interventions */
.factor-card {
    background: var(--theme-alt-bg);
    border: 1px solid var(--theme-alt-border);
    border-left: 4px solid var(--theme-accent);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
}
.factor-name {
    font-weight: 700;
    color: var(--theme-text-primary);
    font-size: 0.92rem;
}
.factor-desc {
    color: var(--theme-text-secondary);
    font-size: 0.85rem;
    margin-top: 0.2rem;
    line-height: 1.45;
}

.intervention-box {
    background: var(--theme-alt-bg);
    border: 1px solid var(--theme-card-border);
    border-left: 4px solid #10b981;
    border-radius: 0 12px 12px 0;
    padding: 1.1rem 1.4rem;
    margin-top: 0.8rem;
}
.intervention-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: var(--theme-text-primary);
    margin-bottom: 0.35rem;
}
.intervention-text {
    color: var(--theme-text-secondary);
    font-size: 0.92rem;
    line-height: 1.55;
}

/* What-If Scenario Cards */
.scenario-card {
    background: var(--theme-card-bg);
    border: 2px solid var(--theme-card-border);
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    box-shadow: var(--theme-shadow);
}
.scenario-current { border-color: #6366f1; }
.scenario-modified { border-color: #8b5cf6; }
.scenario-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--theme-text-muted);
    margin-bottom: 0.35rem;
}
.scenario-prob {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
}

.disclaimer-box {
    background: var(--theme-alt-bg);
    border: 1px solid var(--theme-card-border);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin: 1.0rem 0;
    font-size: 0.85rem;
    color: var(--theme-text-muted);
    line-height: 1.5;
}
</style>
"""

# =====================================================================
# API Clients & Helpers
# =====================================================================

def check_backend_health() -> Dict[str, Any]:
    """Check health status of the FastAPI service."""
    base_url = get_api_base_url()
    try:
        resp = requests.get(f"{base_url}/health", timeout=2.5)
        if resp.status_code == 200:
            return {"online": True, "data": resp.json(), "url": base_url}
        return {"online": False, "error": f"HTTP {resp.status_code}", "url": base_url}
    except requests.RequestException as exc:
        return {"online": False, "error": str(exc), "url": base_url}


def call_predict_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send order payload to FastAPI /predict endpoint."""
    base_url = get_api_base_url()
    predict_endpoint = f"{base_url}/predict"
    try:
        t0 = time.perf_counter()
        resp = requests.post(predict_endpoint, json=payload, timeout=6.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 200:
            data = resp.json()
            data["latency_ms"] = round(latency_ms, 1)
            data["api_url"] = base_url
            return {"success": True, "data": data}
        else:
            return {
                "success": False,
                "error": f"Service returned status {resp.status_code}",
            }
    except requests.RequestException:
        return {
            "success": False,
            "error": "Return risk service is currently unavailable. Please try again.",
        }


def get_risk_badge(risk_level: str) -> str:
    level = (risk_level or "LOW").upper()
    css_cls = {
        "HIGH": "risk-high",
        "MEDIUM": "risk-medium",
        "LOW": "risk-low",
    }.get(level, "risk-low")
    icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    return f'<span class="risk-badge {css_cls}">{icons.get(level, "⚪")} {level} RISK</span>'


def get_risk_color(risk_level: str) -> str:
    return {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}.get(
        (risk_level or "").upper(), "#64748b"
    )


def generate_order_id() -> str:
    """Return a unique order ID format ORD-XXXXXX."""
    return f"ORD-{random.randint(100000, 999999)}"


# =====================================================================
# State & Customer Behavioral Feature Generation
# =====================================================================

def init_session_state() -> None:
    if "orders_history" not in st.session_state:
        st.session_state.orders_history = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_placed_order" not in st.session_state:
        st.session_state.last_placed_order = None
    if "preset_data" not in st.session_state:
        st.session_state.preset_data = PRESET_STANDARD.copy()
    if "active_view" not in st.session_state:
        st.session_state.active_view = "🛍️ Customer Storefront"

    for cat in PRODUCT_CATEGORIES:
        key = f"sf_prod_select_{cat}"
        if key not in st.session_state:
            cat_prods = [p["id"] for p in SIMULATED_CATALOG if p["category"] == cat]
            if cat_prods:
                st.session_state[key] = cat_prods[0]


def get_quantity_warning(quantity: int) -> Optional[str]:
    """
    Checkout confirmation warning strictly for quantity >= 3.
    Separate from ML prediction. Does NOT make causal claims.
    """
    if quantity >= 3:
        return f"Please double-check the quantity before placing your order. You have selected {quantity} items."
    return None


def generate_customer_recommendation(
    product: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
    selected_size: Optional[str] = None,
    risk_level: Optional[str] = None,
) -> str:
    """
    Generate product-specific customer recommendation based on:
    1. Actual selected product & category
    2. Relevant training-data signals (product_return_rate, size_change_history,
       category_return_history, previous_return_rate)
    3. Selected size / variant and customer inputs
    4. Model prediction/risk when available

    Zero ML technical jargon, no causal claims. Uses polite guidance phrasing.
    """
    if not product:
        return "Please confirm your order details before placing your order."

    p_name = product.get("name", "")
    p_cat = product.get("category", "")
    p_sizes = product.get("sizes", [])
    has_sizes = bool(p_sizes)
    prod_return_rate = float(product.get("product_return_rate", 0.10))

    prof = profile or {}
    size_changes = float(prof.get("size_change_history", 0.0))
    cat_returns = float(prof.get("category_return_history", 0.0))
    prev_returns = float(prof.get("previous_return_rate", 0.0))

    # Base product-specific recommendation
    base_rec = product.get("guidance", "")
    if not base_rec:
        if p_cat == "Clothing":
            base_rec = "Please double-check the selected size and fit before placing your order."
        elif p_cat == "Footwear":
            base_rec = "Please confirm your shoe size and fit before placing your order."
        elif p_cat == "Electronics":
            base_rec = "Please confirm the selected model and specifications before placing your order."
        elif p_cat == "Beauty / Makeup":
            base_rec = "Please confirm the selected shade or variant before placing your order."
        elif p_cat == "Kitchen":
            base_rec = "Please confirm the selected capacity and specifications before placing your order."
        elif p_cat == "Sports":
            base_rec = "Please confirm the selected equipment size/specification before placing your order."
        elif p_cat == "Toys & Games":
            base_rec = "Please confirm the game edition and age suitability before placing your order."
        elif p_cat == "Books":
            base_rec = "Please confirm the book title and edition before placing your order."
        elif p_cat == "Home":
            base_rec = "Please confirm the product dimensions and intended use before placing your order."
        else:
            base_rec = "Please confirm your product selection and specifications before placing your order."

    # Signal 1: Size change history applies to sized items
    if has_sizes and size_changes >= 2.0:
        if p_cat == "Clothing":
            return "Please double-check the selected size and fit before placing your order to ensure the best fit."
        elif p_cat == "Footwear":
            return "Please confirm your shoe size and fit before placing your order to ensure the best fit."
        elif "bat" in p_name.lower():
            return "Please confirm the selected bat size/specification and handle fit before placing your order."
        elif "curtain" in p_name.lower():
            return "Please double-check your window measurements and curtain drop length before placing your order."
        elif "football" in p_name.lower():
            return "Please confirm the selected football ball size before placing your order."
        elif "yoga" in p_name.lower():
            return "Please confirm the yoga mat dimensions and thickness before placing your order."
        else:
            return f"Please confirm your selected size ({selected_size or 'selected size'}) and fit before placing your order."

    # Signal 2: Elevated product return rate in training data (e.g. >= 0.22)
    if prod_return_rate >= 0.22:
        if has_sizes:
            return "Please double-check the selected size, fit, and product details carefully before placing your order."
        else:
            return f"Please double-check the {p_name} specifications and details carefully before placing your order."

    # Signal 3: Customer with elevated historical return rates
    if cat_returns >= 0.35 or prev_returns >= 0.35:
        if has_sizes:
            return "Consider checking the size guide and fit details carefully before placing your order."
        else:
            return base_rec.replace("Please confirm", "Consider checking")

    # Signal 4: Elevated Model Risk (if evaluated post-prediction)
    if risk_level == "HIGH":
        if has_sizes:
            return "Please double-check the selected size and fit before placing your order."
        else:
            return f"Please confirm the selected {product.get('guidance_topic', 'specifications')} before placing your order."

    return base_rec


def get_or_create_customer_profile(customer_name: str) -> Dict[str, Any]:
    """
    Return realistic baseline customer behavioral features for the ML model.
    Customer Name serves strictly as customer/order identity, NOT an ML feature.
    The ML prediction is driven by the actual order, catalog, and logistical attributes.
    """
    return {
        "customer_id": 1042,
        "age": 30,
        "tenure_months": 14.0,
        "previous_orders": 8,
        "previous_returns": 1,
        "previous_return_rate": 0.125,
        "customer_complaint_count": 0.0,
        "average_previous_order_value": 75.0,
        "customer_account_verified": True,
        "size_change_history": 1.0,
        "category_return_history": 0.10,
        "previous_category_orders": 3.0,
        "previous_category_returns": 0.0,
    }


# =====================================================================
# View 1: Customer Storefront (NovaCart)
# =====================================================================

def render_customer_storefront() -> None:
    st.markdown(
        """
        <div class="storefront-banner">
            <h2>🛍️ NovaCart</h2>
            <p>Smart shopping, with a little extra confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([5, 4], gap="large")

    with col_left:
        st.subheader("1. Customer Information")

        # Customer name input - starts EMPTY as requested
        customer_name_input = st.text_input(
            "Your Name",
            value="",
            placeholder="e.g. Akshatha",
            help="Enter your name to personalize your order.",
            key="sf_customer_name_input",
        )

        customer_name = customer_name_input.strip()
        if customer_name:
            st.markdown(
                f"<div style='font-size: 1.05rem; font-weight: 700; color: #4f46e5; margin-bottom: 0.5rem;'>"
                f"Hello, {customer_name}! 👋</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Please enter your name above to personalize your order.")

        st.markdown("---")
        st.subheader("2. Choose Product")

        # Category Filter with sensible default (Clothing)
        cat_filter = st.selectbox(
            "Shop by Category",
            options=["All Categories"] + PRODUCT_CATEGORIES,
            index=1,  # Default to Clothing
            key="sf_cat_filter",
        )



        # Preserve session_state for inactive category selectboxes so AppTest does not purge them
        for c in PRODUCT_CATEGORIES:
            if c != cat_filter:
                k = f"sf_prod_select_{c}"
                if k in st.session_state:
                    st.session_state[k] = st.session_state[k]

        # Build filtered catalog based on selected category
        filtered_catalog = SIMULATED_CATALOG
        if cat_filter != "All Categories":
            filtered_catalog = [p for p in SIMULATED_CATALOG if p["category"] == cat_filter]

        if not filtered_catalog:
            st.info(f"No products available for category: {cat_filter}")
            selected_prod = None
        else:
            # Mapping from product ID to product for lookup
            id_to_product = {p["id"]: p for p in filtered_catalog}
            prod_ids = list(id_to_product.keys())

            # Category‑specific widget key
            product_widget_key = f"sf_prod_select_{cat_filter}"

            # Determine default index safely without mutating session_state beforehand
            current_selection = st.session_state.get(product_widget_key)
            default_index = (
                prod_ids.index(current_selection)
                if current_selection in prod_ids
                else 0
            )

            # Render product selectbox using the category‑specific key
            selected_id = st.selectbox(
                "Product",
                options=prod_ids,
                index=default_index,
                key=product_widget_key,
            )
            # selected_id is guaranteed to be in prod_ids
            sel_pid = selected_id
            st.session_state["sf_selected_product_id"] = sel_pid
            selected_prod = id_to_product[sel_pid]

        # Display Selected Product Card if a product is selected
        if selected_prod:
            st.markdown(
                f"""
                <div class="product-display-card">
                    <div class="product-icon-large">{selected_prod['icon']}</div>
                    <div class="product-title">{selected_prod['name']}</div>
                    <span class="product-category-chip">{selected_prod['category']}</span>
                    <span class="product-category-chip">⭐ {selected_prod['rating']} / 5.0</span>
                    <div class="product-price-tag">${selected_prod['price']:.2f}</div>
                    <div class="product-desc">{selected_prod['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Sizing and Quantity
            opt_col1, opt_col2 = st.columns(2)
            selected_size = None
            with opt_col1:
                has_sizes = bool(selected_prod.get("sizes"))
                prod_sizes = selected_prod.get("sizes") if has_sizes else ["Standard"]
                cur_size = st.session_state.get("sf_size")
                display_options = prod_sizes if (not cur_size or cur_size in prod_sizes) else [cur_size] + prod_sizes
                size_idx = display_options.index(cur_size) if cur_size in display_options else 0
                sf_size_val = st.selectbox(
                    "Size / Variant" if has_sizes else "Variant",
                    options=display_options,
                    index=size_idx,
                    disabled=not has_sizes,
                    key="sf_size",
                )
                selected_size = sf_size_val if (has_sizes and sf_size_val in prod_sizes) else (prod_sizes[0] if has_sizes else None)
            with opt_col2:
                quantity = st.number_input("Quantity", min_value=1, max_value=5, value=1, step=1, key="sf_qty")

            # Retrieve customer profile for guidance signals
            cust_profile = get_or_create_customer_profile(customer_name or "Guest")

            # Generate product-specific recommendation
            cust_rec = generate_customer_recommendation(
                product=selected_prod,
                profile=cust_profile,
                selected_size=selected_size,
            )
            qty_warning = get_quantity_warning(quantity)

            # Pre-order checks and guidance
            st.markdown(
                f"""
                <div style="background: rgba(79, 70, 229, 0.07); border: 1px solid rgba(79, 70, 229, 0.3); border-radius: 8px; padding: 12px 16px; margin-top: 10px;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: #4338ca; margin-bottom: 3px;">
                        💡 Product-Specific Shopping Guidance
                    </div>
                    <div style="font-size: 0.88rem; color: var(--theme-text-primary, #1e293b);">
                        {cust_rec}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Separate Quantity Warning (Only when quantity >= 3)
            if qty_warning:
                st.markdown(
                    f"""
                    <div style="background: rgba(245, 158, 11, 0.10); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 12px 16px; margin-top: 8px;">
                        <div style="font-weight: 700; font-size: 0.92rem; color: #b45309; margin-bottom: 2px;">
                            ⚠️ Checkout Quantity Notice
                        </div>
                        <div style="font-size: 0.88rem; color: #92400e;">
                            {qty_warning}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_right:
        st.subheader("3. Delivery & Payment")

        if not selected_prod:
            st.info("Please select a product from the catalog to configure delivery and payment.")
            return

        with st.container():
            delivery_opt = st.selectbox(
                "Delivery",
                options=[
                    "Metro Express (1-2 business days)",
                    "Standard Delivery (3-5 business days)",
                ],
                index=0,
                key="sf_delivery_opt",
            )
            is_expedited = "Express" in delivery_opt
            delivery_km = 8.0 if is_expedited else 25.0

            chk_c1, chk_c2 = st.columns(2)
            with chk_c1:
                is_expedited_chk = st.checkbox("⚡ Express Delivery", value=is_expedited, key="sf_express_chk")
            with chk_c2:
                is_gift = st.checkbox("🎁 Gift Packaging", value=False, key="sf_gift")

            promo_choice = st.selectbox(
                "Coupon",
                options=[
                    "None",
                    "SAVE15 (15% Seasonal Promo)",
                    "FLASH40 (40% Clearance Discount)",
                ],
                index=0,
                key="sf_promo_choice",
            )
            discount_pct = 0.0
            if "15%" in promo_choice:
                discount_pct = 15.0
            elif "40%" in promo_choice:
                discount_pct = 40.0

            st.markdown("**Payment**")
            payment_method = st.radio(
                "Payment Method",
                options=PAYMENT_METHODS,
                index=2,  # Default to UPI
                horizontal=True,
                key="sf_payment_method",
                label_visibility="collapsed",
            )

            # Cart Calculation
            subtotal = selected_prod["price"] * quantity
            discount_amount = subtotal * (discount_pct / 100.0)
            shipping_fee = 5.00 if is_expedited_chk else 0.00
            order_total = max(round(subtotal - discount_amount + shipping_fee, 2), 1.00)

            st.markdown(
                f"""
                <div class="checkout-summary-card">
                    <div class="summary-title">Order Summary</div>
                    <div class="summary-line">
                        <span>Items ({quantity}x)</span>
                        <span>${subtotal:.2f}</span>
                    </div>
                    <div class="summary-line" style="color: #16a34a;">
                        <span>Discount ({discount_pct:.0f}%)</span>
                        <span>-${discount_amount:.2f}</span>
                    </div>
                    <div class="summary-line">
                        <span>Delivery</span>
                        <span>${shipping_fee:.2f}</span>
                    </div>
                    <div class="summary-total-line">
                        <span>Total</span>
                        <span>${order_total:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")
            place_order_btn = st.button(
                "🛍️ PLACE ORDER",
                type="primary",
                use_container_width=True,
                key="sf_place_order_btn",
            )

    # -----------------------------------------------------------------
    # Order Placement Flow
    # -----------------------------------------------------------------
    if place_order_btn:
        if not customer_name:
            st.error("Please enter your name above before placing your order.")
            st.stop()

        order_id_str = generate_order_id()
        now = datetime.datetime.now()
        day_of_week = now.strftime("%A")
        hour_of_day = now.hour

        # Silently retrieve customer profile for ML backend
        profile = get_or_create_customer_profile(customer_name)

        payload = {
            "customer_id": profile["customer_id"],
            "product_id": int(selected_prod["id"].split("-")[1]),
            "customer_age": int(profile["age"]),
            "customer_tenure_months": float(profile["tenure_months"]),
            "previous_orders": int(profile["previous_orders"]),
            "previous_returns": int(profile["previous_returns"]),
            "previous_return_rate": float(profile["previous_return_rate"]),
            "customer_complaint_count": float(profile["customer_complaint_count"]),
            "average_previous_order_value": float(profile["average_previous_order_value"]),
            "customer_account_verified": bool(profile["customer_account_verified"]),
            "product_category": CATEGORY_TO_API.get(selected_prod["category"], selected_prod["category"]),
            "product_price": float(selected_prod["price"]),
            "product_rating": float(selected_prod["rating"]),
            "product_return_rate": float(selected_prod["product_return_rate"]),
            "discount_percentage": float(discount_pct),
            "quantity": int(quantity),
            "order_day": day_of_week,
            "order_hour": int(hour_of_day),
            "delivery_distance_km": float(delivery_km),
            "payment_method": payment_method,
            "order_value": float(order_total),
            "is_gift_order": bool(is_gift),
            "is_expedited_shipping": bool(is_expedited_chk),
            "size_change_history": float(profile["size_change_history"]),
            "category_return_history": float(profile["category_return_history"]),
            "previous_category_orders": float(profile["previous_category_orders"]),
            "previous_category_returns": float(profile["previous_category_returns"]),
        }

        with st.spinner("Processing your order..."):
            api_resp = call_predict_api(payload)

        if not api_resp["success"]:
            st.error(f"⚠️ {api_resp['error']}")
        else:
            pred = api_resp["data"]
            prob = float(pred["return_probability"])
            risk_level = str(pred["risk_level"]).upper()
            pred_class = int(pred["predicted_class"])

            # Compute final product-specific recommendation
            final_recom = generate_customer_recommendation(
                product=selected_prod,
                profile=profile,
                selected_size=selected_size,
                risk_level=risk_level,
            )
            qty_warning_text = get_quantity_warning(quantity) or ""

            # Create unified order record (Single source of truth)
            order_record = {
                "order_id": order_id_str,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_name": customer_name,
                "product_name": selected_prod["name"],
                "category": selected_prod["category"],
                "size": selected_size or "Standard",
                "quantity": int(quantity),
                "price": float(selected_prod["price"]),
                "order_value": float(order_total),
                "payment_method": payment_method,
                "delivery_option": delivery_opt,
                "order_status": "Preparing for Packing",
                "risk_level": risk_level,
                "return_probability": prob,
                "predicted_class": pred_class,
                "recommendation": final_recom,
                "quantity_warning": qty_warning_text,
                "top_risk_factors": pred.get("top_risk_factors", []),
                "suggested_intervention": pred.get("suggested_intervention", ""),
                "payload": payload,
                "raw_prediction": pred,
            }

            # Store in session state for shared Operations Center access
            st.session_state.orders_history.append(order_record)
            st.session_state.history.append(
                {
                    "Timestamp": now.strftime("%H:%M:%S"),
                    "Order ID": order_id_str,
                    "Customer": customer_name,
                    "Category": selected_prod["category"],
                    "Order Value ($)": order_total,
                    "Probability": prob,
                    "Risk Level": risk_level,
                    "Predicted Class": pred_class,
                }
            )
            st.session_state.last_placed_order = order_record

    # -----------------------------------------------------------------
    # Post-Order Customer Experience
    # -----------------------------------------------------------------
    if st.session_state.last_placed_order:
        last_order = st.session_state.last_placed_order
        cust_name = last_order["customer_name"]
        ord_id = last_order["order_id"]
        r_level = last_order["risk_level"]
        p_name = last_order["product_name"]
        p_size = last_order["size"]
        p_size_disp = f" (Size: {p_size})" if p_size != "Standard" else ""

        st.markdown("---")
        st.subheader("📦 Order Status")

        # Clean order-status progression
        st.markdown(
            f"""
            <div class="order-step-container">
                <div class="order-step-item">
                    <span class="order-step-check">✓</span>
                    <span>Order Confirmed</span>
                </div>
                <div class="order-step-item">
                    <span class="order-step-check">✓</span>
                    <span>Payment Confirmed</span>
                </div>
                <div class="order-step-item">
                    <span class="order-step-check">✓</span>
                    <span>Return Risk Check Completed</span>
                </div>
                <div class="order-step-item" style="color: #4f46e5;">
                    <span class="order-step-check">●</span>
                    <span>Preparing for Packing</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Order Details Card
        qty_warn_html = (
            f"<div style='margin-top: 6px; font-size: 0.88rem; color: #b45309;'>⚠️ <strong>Quantity Notice:</strong> {last_order['quantity_warning']}</div>"
            if last_order.get("quantity_warning")
            else ""
        )
        st.markdown(
            f"""
            <div class="order-confirmed-card">
                <div class="order-confirmed-title">Order #{ord_id}</div>
                <div class="order-confirmed-grid">
                    <div><strong>Customer:</strong> {cust_name}</div>
                    <div><strong>Product:</strong> {p_name}{p_size_disp}</div>
                    <div><strong>Quantity:</strong> {last_order['quantity']}</div>
                    <div><strong>Price:</strong> ${last_order.get('price', 0.0):.2f}</div>
                    <div><strong>Total:</strong> ${last_order['order_value']:.2f}</div>
                    <div><strong>Current Status:</strong> {last_order['order_status']}</div>
                    <div><strong>Recommendation:</strong> {last_order['recommendation']}</div>
                </div>
                {qty_warn_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Customer-friendly communication based on risk evaluation
        if r_level == "HIGH":
            st.markdown(
                f"""
                <div class="customer-recom-box recom-high">
                    <h4>💡 Before we pack your order</h4>
                    <p>
                        {last_order['recommendation']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if last_order.get("size") != "Standard":
                    st.button("📏 Review Size Guide", key="sf_btn_size_guide")
                else:
                    st.button("🔍 Review Specifications", key="sf_btn_spec_guide")
            with btn_col2:
                st.button("✅ Confirm Order Details", key="sf_btn_confirm_details")

        elif r_level == "MEDIUM":
            st.markdown(
                f"""
                <div class="customer-recom-box recom-medium">
                    <h4>ℹ️ One quick check before packing</h4>
                    <p>
                        {last_order['recommendation']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="customer-recom-box recom-low">
                    <h4>🎉 Your order looks good to go!</h4>
                    <p>
                        {last_order['recommendation']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================================
# View 2: Operations Center
# =====================================================================

def render_operations_center() -> None:
    st.markdown("## ⚙️ Operations Center")
    st.caption("Monitor orders, return risk and recommended interventions.")

    orders = st.session_state.orders_history

    # -----------------------------------------------------------------
    # 1. Operational Overview (KPI Summary Cards)
    # -----------------------------------------------------------------
    st.markdown("### 📊 Operational Overview")
    total_orders = len(orders)
    high_risk_orders = sum(1 for o in orders if o["risk_level"] == "HIGH")
    ready_orders = sum(1 for o in orders if "Packing" in o.get("order_status", ""))
    avg_risk_str = f"{pd.DataFrame(st.session_state.history)['Probability'].mean():.1%}" if st.session_state.history else "0.0%"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Orders Evaluated</div>
                <div class="kpi-value">{total_orders}</div>
                <div class="kpi-sub">Total session volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Return Risk</div>
                <div class="kpi-value" style="color: #4f46e5;">{avg_risk_str}</div>
                <div class="kpi-sub">Session average</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">High-Risk Orders</div>
                <div class="kpi-value" style="color: #dc2626;">{high_risk_orders}</div>
                <div class="kpi-sub">Intervention recommended</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Orders Ready for Packing</div>
                <div class="kpi-value" style="color: #16a34a;">{ready_orders}</div>
                <div class="kpi-sub">In packing queue</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -----------------------------------------------------------------
    # 2. Recent Orders (Customer -> Operator Sync)
    # -----------------------------------------------------------------
    st.markdown("### 📦 Recent Orders")
    if not orders:
        st.info("No incoming orders yet. Place an order from the **Customer Storefront** to see it evaluated and listed here.")
    else:
        # Show recent orders (newest first)
        recent_orders = orders[::-1]
        for ord_data in recent_orders:
            r_badge = get_risk_badge(ord_data["risk_level"])
            size_display = f"(Size: {ord_data['size']})" if ord_data["size"] != "Standard" else ""

            st.markdown(
                f"""
                <div class="ops-order-card">
                    <div class="ops-order-header">
                        <div>
                            <span class="ops-order-id">{ord_data['order_id']}</span>
                            <span class="ops-customer-name">Customer: {ord_data['customer_name']}</span>
                        </div>
                        <div>{r_badge}</div>
                    </div>
                    <div class="ops-order-body">
                        <div><strong>Item:</strong> {ord_data['product_name']} {size_display}</div>
                        <div><strong>Category:</strong> {ord_data['category']}</div>
                        <div><strong>Qty:</strong> {ord_data['quantity']}</div>
                        <div><strong>Unit Price:</strong> ${ord_data.get('price', 0.0):.2f}</div>
                        <div><strong>Order Value:</strong> ${ord_data['order_value']:.2f}</div>
                        <div><strong>Return Risk:</strong> {ord_data['return_probability']:.1%}</div>
                        <div><strong>Status:</strong> {ord_data['order_status']}</div>
                    </div>
                    <div class="ops-order-recom">
                        <strong>Recommendation:</strong> {ord_data['recommendation']}
                    </div>
                    {f"<div style='margin-top: 4px; font-size: 0.82rem; color: #b45309;'>⚠️ <strong>Quantity Notice:</strong> {ord_data['quantity_warning']}</div>" if ord_data.get('quantity_warning') else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # -----------------------------------------------------------------
    # 3. Selected Order Details Inspector
    # -----------------------------------------------------------------
    if orders:
        st.markdown("### 🔍 Selected Order Details")
        order_options = [f"{o['order_id']} — {o['customer_name']} ({o['product_name']})" for o in orders[::-1]]
        selected_order_idx = st.selectbox(
            "Select an order to inspect details:",
            options=range(len(order_options)),
            format_func=lambda i: order_options[i],
            key="ops_inspect_order_select",
        )
        selected_order = orders[::-1][selected_order_idx]

        insp_c1, insp_c2 = st.columns(2)
        with insp_c1:
            st.markdown("#### Order Summary")
            st.markdown(
                f"""
                - **Order ID:** `{selected_order['order_id']}`
                - **Customer Name:** **{selected_order['customer_name']}**
                - **Placed At:** {selected_order['timestamp']}
                - **Product:** {selected_order['product_name']}
                - **Category:** {selected_order['category']}
                - **Size / Variant:** {selected_order['size']}
                - **Quantity:** {selected_order['quantity']}
                - **Unit Price:** ${selected_order.get('price', 0.0):.2f}
                - **Total Value:** ${selected_order['order_value']:.2f}
                - **Payment Method:** {selected_order['payment_method']}
                - **Delivery Option:** {selected_order['delivery_option']}
                - **Status:** {selected_order['order_status']}
                """
            )

        with insp_c2:
            st.markdown("#### Return Risk Assessment")
            qty_notice_line = f"- **Quantity Notice:** {selected_order['quantity_warning']}\n" if selected_order.get('quantity_warning') else ""
            st.markdown(
                f"""
                - **Return Risk:** **{selected_order['return_probability']:.1%}**
                - **Risk Level:** {get_risk_badge(selected_order['risk_level'])}
                - **Recommendation:** {selected_order['recommendation']}
                {qty_notice_line}
                """,
                unsafe_allow_html=True,
            )

            if selected_order.get("top_risk_factors"):
                st.markdown("**Key Risk Contributors:**")
                for factor in selected_order["top_risk_factors"]:
                    raw_name = factor.get("factor") or factor.get("feature", "Risk Factor")
                    f_name = FEATURE_DISPLAY_NAMES.get(raw_name, raw_name.replace("_", " ").title())
                    f_desc = factor.get("description", "")
                    st.markdown(
                        f"""
                        <div class="factor-card">
                            <div class="factor-name">{f_name}</div>
                            <div class="factor-desc">{f_desc}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if selected_order.get("suggested_intervention"):
                interv = selected_order["suggested_intervention"]
                st.markdown(
                    f"""
                    <div class="intervention-box">
                        <div class="intervention-title">🛡️ Recommended Operational Action</div>
                        <div class="intervention-text">{interv}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

    # -----------------------------------------------------------------
    # 4. Session Analytics (Grounded Strictly in Real Session Data)
    # -----------------------------------------------------------------
    st.markdown("### 📈 Session Analytics")
    if not st.session_state.history:
        st.info("Analytics will become more informative as more orders are evaluated.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        sa1, sa2, sa3, sa4 = st.columns(4)
        with sa1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Orders Evaluated</div>
                    <div class="kpi-value">{len(hist_df)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sa2:
            avg_p = hist_df["Probability"].mean()
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Avg Return Risk</div>
                    <div class="kpi-value" style="color: #4f46e5;">{avg_p:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sa3:
            h_count = (hist_df["Risk Level"] == "HIGH").sum()
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">High-Risk Orders</div>
                    <div class="kpi-value" style="color: #dc2626;">{h_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sa4:
            l_count = (hist_df["Risk Level"] == "LOW").sum()
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Low-Risk Orders</div>
                    <div class="kpi-value" style="color: #16a34a;">{l_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        c_ch1, c_ch2 = st.columns(2)
        with c_ch1:
            st.markdown("##### Risk Tier Breakdown")
            risk_dist = hist_df["Risk Level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
            st.bar_chart(risk_dist, color="#4f46e5")

        with c_ch2:
            st.markdown("##### Return Risk Trend (Chronological)")
            st.line_chart(hist_df["Probability"], color="#dc2626")

        st.markdown("##### Prediction History Log")
        st.dataframe(
            hist_df.style.format(
                {
                    "Probability": "{:.1%}",
                    "Order Value ($)": "${:.2f}",
                }
            ),
            use_container_width=True,
        )

        if st.button("🗑️ Clear Session History", key="ops_clear_history"):
            st.session_state.orders_history = []
            st.session_state.history = []
            st.session_state.last_placed_order = None
            st.rerun()

    st.markdown("---")

    # -----------------------------------------------------------------
    # 5. What-If Risk Analysis
    # -----------------------------------------------------------------
    st.markdown("### 🔬 What-If Risk Analysis")
    st.caption("See how changing selected order conditions changes the model's predicted return risk.")

    st.markdown(
        """
        <div class="disclaimer-box">
            Start with an existing order and adjust selected conditions such as discount, product category,
            customer return history, or complaints. ReTurnIQ sends the changed scenario through the same prediction API
            and compares the result with the original order.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Determine baseline order for What-If Explorer
    base_order = None
    if orders:
        base_order = orders[-1]  # Most recent order
    else:
        # Default baseline if no orders exist yet
        base_order = {
            "order_id": "BASE-DEMO",
            "customer_name": "Demo Shopper",
            "product_name": "Slim-Fit Stretch Denim Jeans",
            "category": "Clothing",
            "return_probability": 0.414,
            "risk_level": "MEDIUM",
            "payload": {
                "customer_age": 28,
                "customer_tenure_months": 12.0,
                "previous_orders": 8,
                "previous_returns": 1,
                "previous_return_rate": 0.125,
                "customer_complaint_count": 0.0,
                "average_previous_order_value": 75.0,
                "customer_account_verified": True,
                "product_category": "Clothing",
                "product_price": 69.99,
                "product_rating": 4.3,
                "product_return_rate": 0.28,
                "discount_percentage": 15.0,
                "quantity": 1,
                "order_day": "Friday",
                "order_hour": 18,
                "delivery_distance_km": 15.0,
                "payment_method": "UPI",
                "order_value": 64.49,
                "is_gift_order": False,
                "is_expedited_shipping": False,
                "size_change_history": 1.0,
                "category_return_history": 0.15,
                "previous_category_orders": 3.0,
                "previous_category_returns": 0.0,
            },
        }

    base_payload = base_order.get("payload", PRESET_STANDARD)
    base_prob = base_order.get("return_probability", 0.414)
    base_risk = base_order.get("risk_level", "MEDIUM")

    st.markdown(
        f"**Active Baseline:** Order `{base_order['order_id']}` ({base_order.get('customer_name', 'Customer')}) · "
        f"Item: {base_order.get('product_name', 'Product')} · Risk: **{base_prob:.1%}** · "
        f"{get_risk_badge(base_risk)}",
        unsafe_allow_html=True,
    )

    st.markdown("#### Adjust scenario variables:")

    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        sc_discount = st.slider(
            "Discount percentage",
            min_value=0.0,
            max_value=80.0,
            value=float(base_payload.get("discount_percentage", 15.0)),
            step=5.0,
            help="Adjust the promotional discount applied to the order.",
        )
    with exp_c2:
        sc_ret_rate = st.slider(
            "Customer return history",
            min_value=0.0,
            max_value=1.0,
            value=float(base_payload.get("previous_return_rate", 0.10)),
            step=0.05,
            help="Simulate a customer with different historical return rates.",
        )
    with exp_c3:
        cat_default = base_payload.get("product_category", "Clothing")
        api_to_ui = {v: k for k, v in CATEGORY_TO_API.items()}
        cat_ui_default = api_to_ui.get(cat_default, cat_default)
        cat_idx = PRODUCT_CATEGORIES.index(cat_ui_default) if cat_ui_default in PRODUCT_CATEGORIES else 0
        sc_cat = st.selectbox(
            "Product category",
            options=PRODUCT_CATEGORIES,
            index=cat_idx,
            help="Evaluate how switching category alters predicted risk.",
        )

    exp_c4, exp_c5, exp_c6 = st.columns(3)
    with exp_c4:
        sc_prod_rate = st.slider(
            "Product return benchmark",
            min_value=0.0,
            max_value=0.60,
            value=float(base_payload.get("product_return_rate", 0.20)),
            step=0.02,
            help="Catalog benchmark return rate for the selected item.",
        )
    with exp_c5:
        sc_size_swaps = st.slider(
            "Size-change history",
            min_value=0.0,
            max_value=10.0,
            value=float(base_payload.get("size_change_history", 1.0)),
            step=1.0,
            help="Frequency of past size-swap requests from this customer.",
        )
    with exp_c6:
        sc_complaints = st.slider(
            "Customer complaints",
            min_value=0.0,
            max_value=10.0,
            value=float(base_payload.get("customer_complaint_count", 0.0)),
            step=1.0,
            help="Number of lifetime complaints logged by this customer.",
        )

    if st.button("⚡ Compare Scenario", type="primary", use_container_width=True, key="btn_run_whatif"):
        mod_payload = base_payload.copy()
        mod_payload["discount_percentage"] = float(sc_discount)
        mod_payload["previous_return_rate"] = float(sc_ret_rate)
        mod_payload["product_category"] = CATEGORY_TO_API.get(sc_cat, sc_cat)
        mod_payload["product_return_rate"] = float(sc_prod_rate)
        mod_payload["size_change_history"] = float(sc_size_swaps)
        mod_payload["customer_complaint_count"] = float(sc_complaints)

        with st.spinner("Evaluating scenario..."):
            sc_res = call_predict_api(mod_payload)

        if not sc_res["success"]:
            st.error(f"Scenario evaluation failed: {sc_res['error']}")
        else:
            sc_pred = sc_res["data"]
            sc_prob = float(sc_pred["return_probability"])
            sc_risk = str(sc_pred["risk_level"]).upper()
            delta = sc_prob - base_prob

            st.markdown("---")
            col_b, col_delta, col_m = st.columns([2, 1, 2])

            with col_b:
                st.markdown(
                    f"""
                    <div class="scenario-card scenario-current">
                        <div class="scenario-label">Current Order Risk</div>
                        <div class="scenario-prob" style="color: {get_risk_color(base_risk)};">{base_prob:.1%}</div>
                        <div style="margin-top: 0.5rem;">{get_risk_badge(base_risk)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_delta:
                delta_icon = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                delta_color = "#dc2626" if delta > 0 else "#16a34a"
                sign = "+" if delta > 0 else ""
                st.markdown(
                    f"""
                    <div style="text-align: center; padding-top: 1.2rem;">
                        <div style="font-size: 1.8rem;">{delta_icon}</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: {delta_color};">{sign}{delta:.1%}</div>
                        <div style="font-size: 0.78rem; color: #64748b; font-weight: 600;">Change in Predicted Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_m:
                st.markdown(
                    f"""
                    <div class="scenario-card scenario-modified">
                        <div class="scenario-label">Scenario Risk</div>
                        <div class="scenario-prob" style="color: {get_risk_color(sc_risk)};">{sc_prob:.1%}</div>
                        <div style="margin-top: 0.5rem;">{get_risk_badge(sc_risk)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="disclaimer-box">
                    <strong>Statistical Note:</strong> Scenario comparison shows model output differences.
                    It is not a causal guarantee. The model's predicted risk changed when this scenario was evaluated.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # -----------------------------------------------------------------
    # 6. Advanced Evaluation Tools (Secondary manual evaluator)
    # -----------------------------------------------------------------
    with st.expander("🛠️ Advanced Evaluation (Manual Model Evaluation)"):
        st.caption(
            "This secondary tool allows engineering evaluators to directly test individual feature combinations "
            "against the live FastAPI /predict service."
        )

        st.markdown("##### Quick Demo Presets")
        p_c1, p_c2, p_c3 = st.columns(3)
        if p_c1.button("🔴 Load High Risk Preset", use_container_width=True):
            st.session_state.preset_data = PRESET_HIGH_RISK.copy()
            st.rerun()
        if p_c2.button("🟡 Load Mid Risk Preset", use_container_width=True):
            st.session_state.preset_data = PRESET_STANDARD.copy()
            st.rerun()
        if p_c3.button("🟢 Load Low Risk Preset", use_container_width=True):
            st.session_state.preset_data = PRESET_LOW_RISK.copy()
            st.rerun()

        current_data = st.session_state.preset_data

        with st.form("manual_evaluation_form"):
            st.markdown("###### Customer Attributes")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                m_age = st.number_input("Customer Age", 18, 100, int(current_data["customer_age"]))
            with c2:
                m_tenure = st.number_input("Tenure (months)", 0.0, 240.0, float(current_data["customer_tenure_months"]))
            with c3:
                m_prev_orders = st.number_input("Lifetime Orders", 0, 500, int(current_data["previous_orders"]))
            with c4:
                m_prev_returns = st.number_input("Lifetime Returns", 0, 500, int(current_data["previous_returns"]))

            c5, c6, c7, c8 = st.columns(4)
            with c5:
                m_ret_rate = st.number_input("Return Rate", 0.0, 1.0, float(current_data["previous_return_rate"]))
            with c6:
                m_complaints = st.number_input("Complaints Count", 0.0, 50.0, float(current_data["customer_complaint_count"]))
            with c7:
                m_avg_spend = st.number_input("Avg Spend ($)", 0.0, 5000.0, float(current_data["average_previous_order_value"]))
            with c8:
                m_verified = st.checkbox("Account Verified", value=bool(current_data["customer_account_verified"]))

            st.markdown("###### Product & Logistics")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                cur_cat = current_data.get("product_category", "Clothing")
                api_to_ui = {v: k for k, v in CATEGORY_TO_API.items()}
                cur_ui_cat = api_to_ui.get(cur_cat, cur_cat)
                p_cat_idx = PRODUCT_CATEGORIES.index(cur_ui_cat) if cur_ui_cat in PRODUCT_CATEGORIES else 0
                m_cat = st.selectbox("Category", PRODUCT_CATEGORIES, index=p_cat_idx)
            with p2:
                m_price = st.number_input("Price ($)", 0.01, 10000.0, float(current_data["product_price"]))
            with p3:
                m_rating = st.number_input("Rating", 1.0, 5.0, float(current_data["product_rating"]))
            with p4:
                m_cat_rate = st.number_input("Product Return Rate", 0.0, 1.0, float(current_data["product_return_rate"]))

            p5, p6, p7, p8 = st.columns(4)
            with p5:
                m_discount = st.number_input("Discount (%)", 0.0, 100.0, float(current_data["discount_percentage"]))
            with p6:
                m_qty = st.number_input("Quantity", 1, 100, int(current_data["quantity"]))
            with p7:
                m_day = st.selectbox("Order Day", ORDER_DAYS, index=ORDER_DAYS.index(current_data["order_day"]) if current_data["order_day"] in ORDER_DAYS else 0)
            with p8:
                m_pay = st.selectbox("Payment Method", PAYMENT_METHODS, index=PAYMENT_METHODS.index(current_data["payment_method"]) if current_data["payment_method"] in PAYMENT_METHODS else 0)

            l1, l2, l3, l4 = st.columns(4)
            with l1:
                m_val = st.number_input("Total Value ($)", 0.01, 50000.0, float(current_data["order_value"]))
            with l2:
                m_dist = st.number_input("Distance (km)", 0.0, 5000.0, float(current_data["delivery_distance_km"]))
            with l3:
                m_gift = st.checkbox("Gift Order", value=bool(current_data["is_gift_order"]))
            with l4:
                m_exp = st.checkbox("Expedited Shipping", value=bool(current_data["is_expedited_shipping"]))

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                m_size_swaps = st.number_input("Size Swap History", 0.0, 50.0, float(current_data["size_change_history"]))
            with b2:
                m_cat_hist = st.number_input("Category Return Rate", 0.0, 1.0, float(current_data["category_return_history"]))
            with b3:
                m_cat_orders = st.number_input("Category Orders", 0.0, 500.0, float(current_data["previous_category_orders"]))
            with b4:
                m_cat_returns = st.number_input("Category Returns", 0.0, 500.0, float(current_data["previous_category_returns"]))

            eval_submit = st.form_submit_button("⚡ Run Model Evaluation", type="primary", use_container_width=True)

        if eval_submit:
            if m_prev_returns > m_prev_orders:
                st.error("Validation Error: Lifetime Returns cannot exceed Lifetime Orders.")
                st.stop()
            if m_cat_returns > m_cat_orders:
                st.error("Validation Error: Category Returns cannot exceed Category Orders.")
                st.stop()

            manual_payload = {
                "customer_age": int(m_age),
                "customer_tenure_months": float(m_tenure),
                "previous_orders": int(m_prev_orders),
                "previous_returns": int(m_prev_returns),
                "previous_return_rate": float(m_ret_rate),
                "customer_complaint_count": float(m_complaints),
                "average_previous_order_value": float(m_avg_spend),
                "customer_account_verified": bool(m_verified),
                "product_category": CATEGORY_TO_API.get(m_cat, m_cat),
                "product_price": float(m_price),
                "product_rating": float(m_rating),
                "product_return_rate": float(m_cat_rate),
                "discount_percentage": float(m_discount),
                "quantity": int(m_qty),
                "order_day": m_day,
                "order_hour": 14,
                "delivery_distance_km": float(m_dist),
                "payment_method": m_pay,
                "order_value": float(m_val),
                "is_gift_order": bool(m_gift),
                "is_expedited_shipping": bool(m_exp),
                "size_change_history": float(m_size_swaps),
                "category_return_history": float(m_cat_hist),
                "previous_category_orders": float(m_cat_orders),
                "previous_category_returns": float(m_cat_returns),
            }

            with st.spinner("Scoring via FastAPI /predict..."):
                m_res = call_predict_api(manual_payload)

            if not m_res["success"]:
                st.error(f"Prediction failed: {m_res['error']}")
            else:
                m_pred = m_res["data"]
                m_prob = float(m_pred["return_probability"])
                m_risk = str(m_pred["risk_level"]).upper()
                st.success(f"Scoring Complete: **{m_risk} Risk** (Probability: **{m_prob:.1%}**)")

    # -----------------------------------------------------------------
    # 7. Technical Details (For Evaluators)
    # -----------------------------------------------------------------
    with st.expander("🔬 Technical Details (For Evaluators)"):
        base_url = get_api_base_url()
        health = check_backend_health()

        m_name = "Logistic Regression"
        d_thresh = 0.63
        f_count = 52
        if health["online"]:
            h_data = health["data"]
            m_name = h_data.get("model_name", "Logistic Regression")
            if m_name == "LogisticRegression":
                m_name = "Logistic Regression"
            d_thresh = float(h_data.get("deployment_threshold", 0.63))
            f_count = int(h_data.get("feature_count", 52))

        st.markdown(
            f"""
            **Prediction Service**  
            The backend service that receives order information and returns the ML prediction.  
            *Prediction API:* `{base_url}/predict`

            **ML Model**  
            {m_name} — the trained model currently used to estimate return probability.

            **Decision Threshold**  
            {d_thresh:.2f} — predictions at or above this probability are classified as a predicted return.

            **Processed Features**  
            {f_count} — the number of model-ready features after preprocessing and encoding.
            """
        )

        st.markdown("**Prediction Flow**")
        st.code(
            """Customer Storefront
       ↓
Order information
       ↓
Prediction API
       ↓
Trained ML model
       ↓
Return probability
       ↓
Risk level + guidance""",
            language="text",
        )


# =====================================================================
# Main Application Entrypoint
# =====================================================================

def main() -> None:
    st.set_page_config(
        page_title="ReTurnIQ — Return Risk Prediction & Prevention",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    # -----------------------------------------------------------------
    # Sidebar (Clean & Non-Technical)
    # -----------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 📦 ReTurnIQ")
        st.caption("AI-Powered E-Commerce Return Risk Prediction")

        st.markdown("---")
        st.markdown("**Application**")
        selected_view = st.radio(
            "Navigation",
            options=["🛍️ Customer Storefront", "⚙️ Operations Center"],
            index=0 if st.session_state.active_view == "🛍️ Customer Storefront" else 1,
            key="sidebar_view_nav",
            label_visibility="collapsed",
        )
        st.session_state.active_view = selected_view

        st.markdown("---")
        health = check_backend_health()
        if health["online"]:
            st.markdown(
                "<div style='font-size: 0.85rem; color: #16a34a; font-weight: 600; display: flex; align-items: center;'>"
                "<span style='margin-right: 0.4rem;'>●</span> System Online"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size: 0.85rem; color: #dc2626; font-weight: 600; display: flex; align-items: center;'>"
                "<span style='margin-right: 0.4rem;'>●</span> System Offline"
                "</div>",
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------
    # App Top Branding Banner
    # -----------------------------------------------------------------
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1 class="app-header-title">📦 ReTurnIQ</h1>
                <div class="app-header-subtitle">
                    Smart shopping, safer ordering.
                </div>
            </div>
            <div class="app-header-badge">
                Smart Return Prevention
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Active View
    if st.session_state.active_view == "🛍️ Customer Storefront":
        render_customer_storefront()
    else:
        render_operations_center()


if __name__ == "__main__":
    main()
