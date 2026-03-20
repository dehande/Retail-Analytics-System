# Retail-Analytics-System
### End-to-end data analytics for a fashion retail store in Moscow

A practical analytics project built for a single-location fashion store. Covers sales analysis, external factor correlation, staff performance tracking, and a weekly sales competition tool — all running locally without a database or cloud infrastructure.
---
Parses monthly sales exports from the POS system and builds a feature matrix
Visualizes sales trends, category performance, discount patterns, and staff metrics
Correlates sales with external 4 factors 
Calculates a multi-factor bonus score per sales advisor, separating mandatory discount periods from free-selling periods
Generates a weekly competition dashboard — the supervisor fills a simple HTML form, a leaderboard downloads automatically
Produces a staff training dashboard with upsell guides, natural product pairs, and price tactics
---
## Stack
Python — pandas, matplotlib, scikit-learn, lightgbm, yfinance, requests
HTML / JavaScript / Chart.js — standalone offline dashboards, no server required
---
## Files

| File | Description |
|------|-------------|
| `01_feature_engineering.py` | Parses sales data, builds monthly feature matrix |
| `02_visualization.py` | Sales analysis charts |
| `03_external_vs_sales.py` | External factors vs sales comparison |
| `04_bonus_dashboard.py` | Staff bonus calculation and performance dashboard |
| `05_demand_forecast.py` | Demand forecasting (naive baseline, Ridge, LightGBM) |
| `06_staff_training_dashboard.html` | Offline training dashboard for sales advisors |
| `07_weekly_form.html` | Supervisor fills weekly data → leaderboard downloads |
---
## Notes

- Sales data is anonymized. Real store name and staff names have been replaced.
- Demand forecasting model is intentionally simple — 4 months of data is not enough for reliable ML predictions. The architecture is in place for when more data accumulates.
- All dashboards are self-contained HTML files designed to work offline, as the store has no on-site computer.
---
## Context

Built as a consulting engagement. The store had no analytics infrastructure — data lived in Excel exports. The goal was to build something the manager could actually use, not something that looked impressive in a notebook.
---
## License
MIT — free to use with attribution.
