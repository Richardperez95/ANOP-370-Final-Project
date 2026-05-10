"""
app_live_optimizer.py

Streamlit app for a live McDonald's meal optimization model.

This app loads the pickle files created by prepare_mcdonalds_pickles.py and runs a
new Gurobi solve whenever the user changes constraints and clicks Optimize Meal.

Required files in this same folder, or in a ./data subfolder:
    - df_nutrients.pkl
    - nutritional_targets.pkl
    - meal_plan_results.pkl        optional, used only for default comparison
    - mcdonalds_optimization_bundle.pkl optional

Run:
    python -m streamlit run app_live_optimizer.py
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception as exc:  # keeps app usable enough to explain missing dependency
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False
    GUROBI_IMPORT_ERROR = exc


st.set_page_config(
    page_title="McDonald's Live Meal Optimizer",
    page_icon="🍟",
    layout="wide",
)

PERSON_TYPES = ["male", "female"]
MEAL_TYPES = ["breakfast", "lunch_dinner"]

NUTRIENT_COL_MAP = {
    "Calories": "Calories",
    "Carbohydrates": "Carbohydrates",
    "Protein": "Protein",
    "Total Fat": "Total Fat",
    "Iron (% Daily Value)": "Iron",
    "Sodium": "Sodium",
}

DISPLAY_NUTRIENTS = [
    "Calories",
    "Carbohydrates",
    "Protein",
    "Total Fat",
    "Iron (% Daily Value)",
    "Sodium",
]


@st.cache_data
def load_pickle(path: str) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


def find_file(filename: str) -> Path:
    """Find a file in the current folder, app folder, or ./data folder."""
    candidates = []
    cwd = Path.cwd()
    app_dir = Path(__file__).resolve().parent if "__file__" in globals() else cwd

    for base in [cwd, app_dir, cwd / "data", app_dir / "data"]:
        candidates.append(base / filename)

    for path in candidates:
        if path.exists():
            return path

    st.error(
        f"Could not find `{filename}`. Put it in the same folder as this app "
        "or inside a `data` subfolder."
    )
    st.stop()


@st.cache_data
def load_app_data() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    df_nutrients = load_pickle(str(find_file("df_nutrients.pkl")))
    nutritional_targets = load_pickle(str(find_file("nutritional_targets.pkl")))

    # Optional: used only to compare the original saved solution.
    try:
        meal_plan_results = load_pickle(str(find_file("meal_plan_results.pkl")))
    except Exception:
        meal_plan_results = {}

    return df_nutrients, nutritional_targets, meal_plan_results


def meal_label(meal_type: str) -> str:
    return "Lunch/Dinner" if meal_type == "lunch_dinner" else "Breakfast"


def person_label(person_type: str) -> str:
    return str(person_type).title()


def money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def filter_menu_by_meal_type(df_menu: pd.DataFrame, meal_type: str) -> pd.DataFrame:
    if meal_type == "breakfast":
        return df_menu[df_menu["Category"].astype(str).eq("Breakfast")].copy()
    return df_menu[~df_menu["Category"].astype(str).eq("Breakfast")].copy()


def default_targets_for(
    nutritional_targets: Dict[str, Any], person_type: str, meal_type: str
) -> Dict[str, Dict[str, float]]:
    return nutritional_targets.get(person_type, {}).get(meal_type, {})


def make_targets_dataframe(targets: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for nutrient, bounds in targets.items():
        max_value = bounds.get("max", float("inf"))
        rows.append(
            {
                "Nutrient": nutrient,
                "Minimum": bounds.get("min", 0.0),
                "Maximum": "No upper limit" if math.isinf(float(max_value)) else max_value,
            }
        )
    return pd.DataFrame(rows)


def build_totals_df(totals: Dict[str, float], constraints: Dict[str, Dict[str, Optional[float]]]) -> pd.DataFrame:
    rows = []
    for col in DISPLAY_NUTRIENTS:
        target_key = NUTRIENT_COL_MAP[col]
        min_value = constraints.get(target_key, {}).get("min")
        max_value = constraints.get(target_key, {}).get("max")
        total_value = totals.get(col, 0.0)
        status = "OK"
        if min_value is not None and total_value < min_value - 1e-6:
            status = "Below minimum"
        if max_value is not None and total_value > max_value + 1e-6:
            status = "Above maximum"
        rows.append(
            {
                "Nutrient": col,
                "Total": round(total_value, 2),
                "Minimum": None if min_value is None else round(float(min_value), 2),
                "Maximum": "None" if max_value is None else round(float(max_value), 2),
                "Status": status,
            }
        )
    rows.insert(
        0,
        {
            "Nutrient": "Total Cost",
            "Total": money(totals.get("Total Cost", 0.0)),
            "Minimum": "",
            "Maximum": "",
            "Status": "",
        },
    )
    return pd.DataFrame(rows)


def solve_live_meal_plan(
    df_menu: pd.DataFrame,
    meal_type: str,
    constraints: Dict[str, Dict[str, Optional[float]]],
    objective_choice: str,
    max_quantity_per_item: int,
    max_total_items: Optional[int],
    excluded_categories: list[str],
    excluded_items: list[str],
    required_items: list[str],
    max_budget: Optional[float],
    force_at_least_one_item: bool = True,
) -> Dict[str, Any]:
    if not GUROBI_AVAILABLE:
        return {
            "status": "GUROBI_NOT_AVAILABLE",
            "message": f"Gurobi could not be imported: {GUROBI_IMPORT_ERROR}",
            "selected_items": pd.DataFrame(),
            "nutrition_totals": {},
            "objective_value": None,
        }

    available_menu = filter_menu_by_meal_type(df_menu, meal_type)

    if excluded_categories:
        available_menu = available_menu[~available_menu["Category"].isin(excluded_categories)]
    if excluded_items:
        available_menu = available_menu[~available_menu["Item"].isin(excluded_items)]

    # Remove rows that cannot be optimized safely.
    required_numeric_cols = ["Price"] + DISPLAY_NUTRIENTS
    for col in required_numeric_cols:
        available_menu[col] = pd.to_numeric(available_menu[col], errors="coerce")
    available_menu = available_menu.dropna(subset=required_numeric_cols).copy()

    if available_menu.empty:
        return {
            "status": "NO_ITEMS_AVAILABLE",
            "message": "No menu items remain after your meal/category/item filters.",
            "selected_items": pd.DataFrame(),
            "nutrition_totals": {},
            "objective_value": None,
        }

    model = gp.Model("mcdonalds_live_meal_optimizer")
    model.Params.OutputFlag = 0

    x = model.addVars(
        available_menu.index.tolist(),
        vtype=GRB.INTEGER,
        lb=0,
        ub=max_quantity_per_item,
        name="x",
    )

    if objective_choice == "Minimize Calories":
        objective_col = "Calories"
    elif objective_choice == "Minimize Sodium":
        objective_col = "Sodium"
    elif objective_choice == "Minimize Total Fat":
        objective_col = "Total Fat"
    else:
        objective_col = "Price"

    model.setObjective(
        gp.quicksum(float(available_menu.loc[i, objective_col]) * x[i] for i in available_menu.index),
        GRB.MINIMIZE,
    )

    if force_at_least_one_item:
        model.addConstr(gp.quicksum(x[i] for i in available_menu.index) >= 1, name="AtLeastOneItem")

    if max_total_items is not None:
        model.addConstr(gp.quicksum(x[i] for i in available_menu.index) <= max_total_items, name="MaxTotalItems")

    if max_budget is not None:
        model.addConstr(
            gp.quicksum(float(available_menu.loc[i, "Price"]) * x[i] for i in available_menu.index) <= max_budget,
            name="MaxBudget",
        )

    for item in required_items:
        matching_indices = available_menu.index[available_menu["Item"].eq(item)].tolist()
        if matching_indices:
            model.addConstr(gp.quicksum(x[i] for i in matching_indices) >= 1, name=f"Required_{item[:20]}")

    for df_col, target_key in NUTRIENT_COL_MAP.items():
        nutrient_constraints = constraints.get(target_key, {})
        min_value = nutrient_constraints.get("min")
        max_value = nutrient_constraints.get("max")

        if min_value is not None:
            model.addConstr(
                gp.quicksum(float(available_menu.loc[i, df_col]) * x[i] for i in available_menu.index) >= float(min_value),
                name=f"Min_{target_key}",
            )
        if max_value is not None:
            model.addConstr(
                gp.quicksum(float(available_menu.loc[i, df_col]) * x[i] for i in available_menu.index) <= float(max_value),
                name=f"Max_{target_key}",
            )

    model.optimize()

    status_lookup = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    status = status_lookup.get(model.status, f"STATUS_{model.status}")

    result = {
        "status": status,
        "message": "",
        "selected_items": pd.DataFrame(),
        "nutrition_totals": {},
        "objective_value": None,
        "available_menu": available_menu,
    }

    if model.status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        result["message"] = (
            "No feasible meal was found. Try relaxing maximum limits, increasing budget, "
            "allowing more items, or raising max quantity per item."
        )
        return result

    selected_rows = []
    totals = {col: 0.0 for col in DISPLAY_NUTRIENTS}
    total_cost = 0.0

    for i in available_menu.index:
        quantity = x[i].X
        if quantity <= 1e-6:
            continue

        row = available_menu.loc[i]
        quantity_int = int(round(quantity))
        line_cost = float(row["Price"]) * quantity_int
        total_cost += line_cost

        selected_row = {
            "Category": row["Category"],
            "Item": row["Item"],
            "Quantity": quantity_int,
            "Unit Price": float(row["Price"]),
            "Line Cost": line_cost,
        }
        for col in DISPLAY_NUTRIENTS:
            selected_row[col] = float(row[col])
            totals[col] += float(row[col]) * quantity_int
        selected_rows.append(selected_row)

    totals["Total Cost"] = total_cost
    selected_df = pd.DataFrame(selected_rows)

    # Make display columns intuitive: item nutrients are per unit, not multiplied.
    if not selected_df.empty:
        selected_df = selected_df.sort_values(["Category", "Item"]).reset_index(drop=True)

    result["selected_items"] = selected_df
    result["nutrition_totals"] = totals
    result["objective_value"] = float(model.ObjVal)
    return result


def get_saved_result(meal_plan_results: Dict[str, Any], person_type: str, meal_type: str) -> Optional[Dict[str, Any]]:
    try:
        return meal_plan_results.get(person_type, {}).get(meal_type)
    except Exception:
        return None


def main() -> None:
    st.title("🍟 McDonald's Live Meal Optimizer")
    st.caption(
        "This version does not just display saved pickle results. It loads the menu and target data, "
        "then runs a new Gurobi optimization each time you submit the widget settings."
    )

    df_nutrients, nutritional_targets, meal_plan_results = load_app_data()

    with st.sidebar:
        st.header("Optimization Setup")
        person_type = st.selectbox("Person Type", PERSON_TYPES, format_func=person_label)
        meal_type = st.selectbox("Meal Type", MEAL_TYPES, format_func=meal_label)

        objective_choice = st.selectbox(
            "Objective",
            ["Minimize Cost", "Minimize Calories", "Minimize Sodium", "Minimize Total Fat"],
        )

        max_quantity_per_item = st.slider("Maximum Quantity Per Item", min_value=1, max_value=10, value=2, step=1)
        max_total_items_enabled = st.checkbox("Limit total number of items", value=True)
        max_total_items = None
        if max_total_items_enabled:
            max_total_items = st.slider("Maximum Total Items", min_value=1, max_value=20, value=5, step=1)

        max_budget_enabled = st.checkbox("Use maximum budget", value=False)
        max_budget = None
        if max_budget_enabled:
            max_budget = st.slider("Maximum Budget", min_value=1.0, max_value=50.0, value=15.0, step=0.5)

        st.divider()
        st.subheader("Menu Filters")
        available_for_meal = filter_menu_by_meal_type(df_nutrients, meal_type)
        category_options = sorted(available_for_meal["Category"].dropna().astype(str).unique().tolist())
        item_options = sorted(available_for_meal["Item"].dropna().astype(str).unique().tolist())

        excluded_categories = st.multiselect("Exclude Categories", category_options)
        excluded_items = st.multiselect("Exclude Items", item_options)
        required_items = st.multiselect("Require At Least One Of These Items", item_options)

    defaults = default_targets_for(nutritional_targets, person_type, meal_type)

    st.subheader("Nutrition Constraints")
    st.write("Adjust these values, then click **Optimize Meal**. The result below will be a new solve, not the saved result.")

    with st.form("live_optimizer_form"):
        c1, c2, c3 = st.columns(3)

        constraints: Dict[str, Dict[str, Optional[float]]] = {}

        with c1:
            st.markdown("**Minimums**")
            min_calories = st.number_input("Minimum Calories", min_value=0.0, value=float(defaults.get("Calories", {}).get("min", 0.0)), step=50.0)
            min_carbs = st.number_input("Minimum Carbohydrates", min_value=0.0, value=float(defaults.get("Carbohydrates", {}).get("min", 0.0)), step=5.0)
            min_protein = st.number_input("Minimum Protein", min_value=0.0, value=float(defaults.get("Protein", {}).get("min", 0.0)), step=5.0)
            min_fat = st.number_input("Minimum Total Fat", min_value=0.0, value=float(defaults.get("Total Fat", {}).get("min", 0.0)), step=5.0)
            min_iron = st.number_input("Minimum Iron (% Daily Value)", min_value=0.0, value=float(defaults.get("Iron", {}).get("min", 0.0)), step=1.0)
            min_sodium = st.number_input("Minimum Sodium", min_value=0.0, value=float(defaults.get("Sodium", {}).get("min", 0.0)), step=50.0)

        with c2:
            st.markdown("**Maximums**")
            use_max_calories = st.checkbox("Maximum Calories", value=False)
            max_calories = st.number_input("Max Calories Value", min_value=0.0, value=max(1200.0, float(defaults.get("Calories", {}).get("min", 0.0)) * 1.5), step=50.0, disabled=not use_max_calories)

            use_max_carbs = st.checkbox("Maximum Carbohydrates", value=False)
            max_carbs = st.number_input("Max Carbohydrates Value", min_value=0.0, value=max(150.0, float(defaults.get("Carbohydrates", {}).get("min", 0.0)) * 1.5), step=5.0, disabled=not use_max_carbs)

            use_max_protein = st.checkbox("Maximum Protein", value=False)
            max_protein = st.number_input("Max Protein Value", min_value=0.0, value=max(100.0, float(defaults.get("Protein", {}).get("min", 0.0)) * 2.0), step=5.0, disabled=not use_max_protein)

            use_max_fat = st.checkbox("Maximum Total Fat", value=False)
            max_fat = st.number_input("Max Total Fat Value", min_value=0.0, value=max(80.0, float(defaults.get("Total Fat", {}).get("min", 0.0)) * 1.5), step=5.0, disabled=not use_max_fat)

            use_max_iron = st.checkbox("Maximum Iron", value=False)
            max_iron = st.number_input("Max Iron Value", min_value=0.0, value=max(100.0, float(defaults.get("Iron", {}).get("min", 0.0)) * 2.0), step=1.0, disabled=not use_max_iron)

            use_max_sodium = st.checkbox("Maximum Sodium", value=False)
            max_sodium = st.number_input("Max Sodium Value", min_value=0.0, value=max(2300.0, float(defaults.get("Sodium", {}).get("min", 0.0)) * 2.0), step=50.0, disabled=not use_max_sodium)

        with c3:
            st.markdown("**Run Model**")
            st.info(
                "If the model is infeasible, relax maximums, increase budget, allow more items, "
                "or increase max quantity per item."
            )
            run_model = st.form_submit_button("Optimize Meal", type="primary")

        constraints = {
            "Calories": {"min": min_calories, "max": max_calories if use_max_calories else None},
            "Carbohydrates": {"min": min_carbs, "max": max_carbs if use_max_carbs else None},
            "Protein": {"min": min_protein, "max": max_protein if use_max_protein else None},
            "Total Fat": {"min": min_fat, "max": max_fat if use_max_fat else None},
            "Iron": {"min": min_iron, "max": max_iron if use_max_iron else None},
            "Sodium": {"min": min_sodium, "max": max_sodium if use_max_sodium else None},
        }

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Default Targets From Pickle")
        st.dataframe(make_targets_dataframe(defaults), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Available Menu Preview")
        preview = filter_menu_by_meal_type(df_nutrients, meal_type)
        st.metric("Available Items Before Exclusions", len(preview))
        display_cols = ["Category", "Item", "Price", "Calories", "Protein", "Sodium"]
        st.dataframe(preview[display_cols].head(10), use_container_width=True, hide_index=True)

    if not GUROBI_AVAILABLE:
        st.error(
            "Gurobi is not available in this Python environment. Install it before running live optimization."
        )
        st.code("pip install gurobipy", language="powershell")
        st.stop()

    if run_model:
        with st.spinner("Solving a new meal optimization model..."):
            result = solve_live_meal_plan(
                df_menu=df_nutrients,
                meal_type=meal_type,
                constraints=constraints,
                objective_choice=objective_choice,
                max_quantity_per_item=max_quantity_per_item,
                max_total_items=max_total_items,
                excluded_categories=excluded_categories,
                excluded_items=excluded_items,
                required_items=required_items,
                max_budget=max_budget,
            )

        st.subheader("New Optimized Meal")
        status = result.get("status")
        if status in {"OPTIMAL", "SUBOPTIMAL"}:
            totals = result.get("nutrition_totals", {})
            selected_items = result.get("selected_items", pd.DataFrame())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Status", status)
            m2.metric("Total Cost", money(totals.get("Total Cost", 0.0)))
            m3.metric("Calories", f"{totals.get('Calories', 0.0):,.0f}")
            m4.metric("Protein", f"{totals.get('Protein', 0.0):,.1f} g")

            st.markdown("### Selected Items")
            display_selected = selected_items.copy()
            if not display_selected.empty:
                for col in ["Unit Price", "Line Cost"]:
                    display_selected[col] = display_selected[col].map(lambda x: round(float(x), 2))
            st.dataframe(display_selected, use_container_width=True, hide_index=True)

            st.markdown("### Nutrition Totals vs. Constraints")
            st.dataframe(build_totals_df(totals, constraints), use_container_width=True, hide_index=True)

            csv = selected_items.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Selected Meal as CSV",
                data=csv,
                file_name=f"optimized_{person_type}_{meal_type}.csv",
                mime="text/csv",
            )
        else:
            st.error(f"Model status: {status}")
            st.write(result.get("message", "No solution was returned."))

    else:
        st.subheader("Saved Optimal Meal From Pickle")
        st.write(
            "This is the original saved result. Click **Optimize Meal** above to run a new model using your widget settings."
        )
        saved_result = get_saved_result(meal_plan_results, person_type, meal_type)
        if saved_result:
            saved_items = saved_result.get("selected_items", pd.DataFrame())
            saved_totals = saved_result.get("nutrition_totals", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Saved Status", saved_result.get("status", "N/A"))
            m2.metric("Saved Cost", money(saved_totals.get("Total Cost", 0.0)))
            m3.metric("Saved Calories", f"{saved_totals.get('Calories', 0.0):,.0f}")
            st.dataframe(saved_items, use_container_width=True, hide_index=True)
        else:
            st.info("No saved result was available, but live optimization can still run.")


if __name__ == "__main__":
    main()
